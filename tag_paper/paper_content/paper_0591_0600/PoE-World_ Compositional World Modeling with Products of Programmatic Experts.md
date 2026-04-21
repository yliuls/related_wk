Title: PoE-World: Compositional World Modeling with Products of Programmatic Experts
Abstract: Learning how the world works is central to building AI agents that can adapt to complex environments. Traditional world models based on deep learning demand vast amounts of training data, and do not flexibly update their knowledge from sparse observations. Recent advances in program synthesis using Large Language Models (LLMs) give an alternate approach which learns world models represented as source code, supporting strong generalization from little data. To date, application of program-structured world models remains limited to natural language and grid-world domains. We introduce a novel program synthesis method for effectively modeling complex, non-gridworld domains by representing a world model as an exponentially-weighted product of programmatic experts (PoE-World) synthesized by LLMs. We show that this approach can learn complex, stochastic world models from just a few observations. We evaluate the learned world models by embedding them in a model-based planning agent, demonstrating efficient performance and generalization to unseen levels on Atari's Pong and Montezuma's Revenge. We release our code and display the learned world models and videos of the agent's gameplay at https://topwasu.github.io/poe-world.

Section: Introduction
How should an intelligent agent represent the dynamics of the natural world? We want a representation that is efficiently learnable, yet flexible enough to handle stochasticity and partial observability, and which supports planning and decision-making. Neural network world models such as Dreamer [1] are radically flexible, but demand enormous training data (compared to humans [2]). Symbolic world models such as WorldCoder [3] instead generate a Python program to represent how the world works. These programmatic world models are data-efficient, because program synthesis requires less data than neural network training-but struggle to scale beyond simple gridworlds, as they do a discrete combinatorial search to find a single large program describing everything about how the world works.
We take inspiration from a longstanding view in philosophy and cognitive science of the mind as a community of interacting experts [4,5,6,7]. This modular organization is evident across multiple scales in natural intelligence, from functional specialization of brain systems [8] to the distinct learning trajectories of specific skills [9,10]. We integrate this modular perspective with the computational paradigm of learning as Program Synthesis, which models learned concepts as symbolic programs [11,12,13,14,15,16]. We extend this line of work by proposing a new computational account of learning world models: as the acquisition of context-specific expert programs, which are refined through practice and reused compositionally to support flexible, goal-directed behavior.
Algorithmically, our key idea is to decompose the problem of learning a world program into learning hundreds of small programs. Each of these learned programs encodes a different causal law, which we probabilistically aggregate to predict future observations (Figure 1a). This makes our world knowledge more modular, and also more learnable, because we no longer search for a single monolithic program handling everything at once. The resulting system, which we call PoE-World (Product of programmatic Experts), can build elaborate world models that accurately support planning and reinforcement learning (RL) from even a brief demonstration in complex Atari games, such as Montezuma's Revenge. PoE-World handles stochasticity because the product of programs is probabilistic, and, as we show, further handles partial observability. To the best of our knowledge, this is the first time a symbolic world model has been learned for environments of this complexity.
Importantly, although PoE-World models fine-grained pixel-level movement, it does not model pixellevel visual appearance, instead assuming symbolic observations from an object detector. Unlike model-based reinforcement learning, PoE-World does not attempt efficient exploration, but focuses on faithfully learning from a demonstrated trajectory (Figure 1b). Lastly, while the world models are ultimately used for planning (Figure 1c), PoE-World fundamentally focuses on world modeling, and not on solving the challenging computational problem of planning itself.
Despite these limitations, we view PoE-World as addressing a central learning problem: Given limited demonstrations of a new environment, quickly assemble a working world model that compositionally generalizes to new situations (Figure 1d). We highlight the following contributions:
1. The PoE-World representation and learning algorithm for symbolic world models.
2. An empirical study of PoE-World on two representative Atari games, Pong and Montezuma's Revenge, demonstrating its superior learning efficiency compared to deep RL, and improved scalability compared to state-of-the-art symbolic model-based RL: PoE-World can synthesize 4000+ line programs that generalize zero-shot to novel game levels and game variations. 3. Demonstration of how to use PoE-World's world models for planning-based decision making, and as a simulated pretraining environment for deep RL.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

Section: Background: World Model Learning
A sequential decision-making problem can be described as (O, A, P, R) where O is an observation space, A is an action space, P is an environment dynamics P = p env (o t+1 |o 1:t , a 1:t ) where o ∈ O and a ∈ A, and R is a reward function. This full-history environment formulation is mathematically equivalent to a Partially Observable Markov Decision Process (POMDP) formulation [17]. In the setting where P is unknown to an agent, the agent learns by interacting with the environment and observing transitions (o t , a t , o t+1 , r t ) at timestep t.
We focus on learning the world model P = p model (o t+1 |o 1:t , a 1:t ) which approximates the true, unknown environment dynamics P = p env (o t+1 |o 1:t , a 1:t ) given observed trajectories
D = {τ i } n i=1
where each trajectory is a sequence of observations and actions τ = (o 1:T +1 , a 1:T ) for some T (r 1:T is also a part of a trajectory, but we drop it for simplicity). The learned world model will later be used by a model-based agent, either through lookahead planning or RL training, to act in the environment.
We treat the world modeling problem as an optimization problem, specifically empirical risk minimization:
p * model = arg min p model (o 1:T +1 ,a 1:T )∈D T t=1 ℓ(p model ; o 1:t+1 , a 1:t )(1)
where ℓ is a loss function, such as the negative log likelihood function ℓ(p model ; o 1:t+1 , a 1:t ) = -log p model (o t+1 |o 1:t , a 1:t ).
Previous works in the model-based reinforcement learning literature [18,19,1,20,21,22,23] have used various deep neural network architectures, including convolutional and recurrent neural networks, transformers, and diffusion models, to define p model as a parametric model p model = p θ . Then, θ is optimized via gradient descent. A weakness of such approaches is poor sample efficiency and generalization. For example, Diamond [22] has failure modes such as imagining that the player can walk through a wall or teleport, even after training on almost 100 hours of observations.
Other works leverage Large Language Models (LLMs) to synthesize a code world model [3,24,25], which can be seen as searching for programmatic p model . These LLM-based code generation algorithms are more sample efficient and extrapolate more systematically than deep learning approaches. However, they have yet to succeed beyond simple text-based and gridworld games.
this section cite: ['b16', 'b17', 'b18', 'b0', 'b19', 'b20', 'b21', 'b22', 'b21', 'b2', 'b23', 'b24']

Section: Modeling the World as a Product of Programmatic Experts (PoE-World)
To address the limitations of existing world model learning methods, we propose representing world models as exponentially-weighted Products of programmatic Experts (PoE-World), enabling sampleefficient and scalable learning of probabilistic world models that leverages LLM code generation. Figure 1 visualizes both the representation and learning algorithm discussed in this section.
this section cite: []

Section: World Model Representation: Product of Programmatic Experts (PoE-World)
World models represented as exponentially weighted Products of programmatic Experts (PoE-World) can be described mathematically as follows:
p θ (o t+1 |o 1:t , a 1:t ) ∝ i p expert i (o t+1 |o 1:t , a 1:t ) θi(2)
where p expert i are programs and θ i are their associated scalar weights.
This representation enables modularity and compositionality. It allows composing many small programs into a full world model to capture complicated environmental dynamics (see Figure 2). We can think of each program as an expert which expresses opinions about particular aspects of the world. For example, in the context of modeling a video game environment, one expert might encode "if a player touches a skull, then the player dies," while another encodes "if an action is LEFT when the player is on a platform, the player's x-axis velocity is -2." The former expert does not express opinions on the player's movement, while the latter does not mention conditions for player's death. A factored state representation yields tractable inference. We assume an object-centric state where each object has a bounding box and velocities, each stored as a separate slot, attribute, or feature.
We treat each such feature as conditionally independent, given the history so far. This independence assumption makes Equation ( 2) tractable, because we can compute a separate normalizing constant for each feature. Concretely, let f index the different object features of the next observation o t+1 , and let o f be the indexing of o with feature f . Then,
p θ (o t+1 |o 1:t , a 1:t ) = f 1 Z f i p expert i (o f t+1 |o 1:t , a 1:t ) θi Z f = o f i p expert i (o f |o 1:t , a 1:t ) θi
Benefits of the full-history formulation over POMDP formulation. The full-history environment formulation is formally equivalent to a POMDP which instead compresses the history into a Markov latent state. We use the history formulation because it makes our world models more modular. Learning global latent variables would entangle all the experts, because every expert would condition on the latent state. Therefore, learning a new latent variable (e.g. "how long the player has been falling") changes the input/output space of every expert, necessitating global joint updates to the structure of every program (e.g. an expert for "is the player dead" would need to be updated). The history formulation allows independent learning of independent mechanisms.
this section cite: []

Section: Hard constraints.
Atari (and the real world) is too complex to perfectly simulate with any effectively learnable program. Therefore, our probabilistic model tends to over-approximate the set of possible futures, giving fuzzy approximate predictions. For example, in Montezuma's Revenge, instead of perfectly modeling the physics of falling downward and landing on the ground, we predict a generic downward trajectory. Ideally that trajectory would perfectly enforce the constraint that the player lands flat on the ground, and never sinks into the ground, but a fuzzy stochastic expert for falling downward could violate that constraint. Therefore, to sharpen the model's outputs, we further learn a collection of hard constraints, {c j }, where c j : O → {0, 1}:
p θ (o t+1 |o 1:t , a 1:t ) ∝ i p expert i (o t+1 |o 1:t , a 1:t ) θi • 1   j c j (o t+1 )   (3
)
We further discuss hard constraints and provide concrete examples in Appendix A.1.
Multi-timestep predictions. We represent the programmatic experts and the world as distributions over the next-timestep observations in eq. ( 2).
It is likewise possible to reformulate a multi-timestep expert p expert (o t+1:t+H |o 1:t , a 1:t ) as a product of next-timestep experts by assuming that the predictions of the multi-step expert at different timestep are independent: p expert (o t+1:t+H |o 1:t , a 1:t ) = H k=1 p expert (o t+k |o 1:t , a 1:t )
this section cite: []

Section: World Model Learning: Program Synthesis and Weight Optimization
We begin with a demonstration trajectory, learn a world model, and then begin to act in the world according to that model. As the agent acts, it collects more trajectory data, which it uses to update or "debug" its model. Concretely, learning proceeds as follows:
Step 1: Synthesize programmatic experts {p expert i } m i=1 given observed trajectories D = {τ i } n i=1 Step 2: Fit the weights θ of the experts according to eq. (5) with a gradient-based optimizer Step 3: Remove the experts with weights below threshold δ Step 4: Repeat Step 1-3 every time the observed trajectories get updated Generating program experts. Following previous works [3, 24, 25], we adopt Large Language Models (LLMs) as our Python program generator. We input a small batch of transitions (o t:t+H+1 , a t:t+H ) to the LLM prompt to produce the programmatic experts {p expert i }.
Figure 2 shows how we interpret small Python programs as distributions over the observations. While we could ask LLMs to synthesize probabilistic programs to specify the distributions, we find it much more effective to have LLMs synthesize simple, deterministic Python programs, presumably because generic Python code is far more prevalent in LLM training data. We assume that an observation is represented as a list of objects, where each object has the following attributes: x/y velocity and visibility. Then, a distribution over observations is a distribution over each object's attributes. To interpret a Python program as a distribution, we assume that all object attributes are conditionally independent given full history, as mentioned in Section 3.1, and convert all object attributes set by the program to distributions with single peaks at the given values. We then add noise to the distributions to ensure non-zero probabilities over alternative values. Any object attributes whose values are not set by the program follow uniform distributions over all possible values: consequently, experts with a single if-condition (fig. 2) yield a uniform distribution when the if-condition is not satisfied.
Gradient-based Weights Optimization. Once we have {p expert i } m i=1 , we can perform maximum likelihood estimation to obtain θ:
θ * = arg max θ (o 1:T +1 ,a 1:T )∈D T t=1 log p θ (o t+1 |o 1:t , a 1:t )(5)
This equation instantiates eq. ( 1) by letting p model have a parametric form p model = p θ and choosing negative log likelihood as the loss function. We can use any gradient-based optimizer to optimize the weights. We use L-BFGS [26], which worked better than Adam [27] and SGD, because we have small data and few parameters.
Finally, the experts whose weights are below a threshold δ are removed from the world model. We repeat this loop every time there are new observations. Appendix A.1 contains full algorithm details.
this section cite: ['b25', 'b26']

Section: World Model Usage: RL in Simulation and Planning with World Model
An important goal of world modeling is to aid decision-making. We consider two such ways of using world models. First, a world model can serve a simulator for reinforcement learning. This makes RL policy learning more sample efficient, because we can quickly learn a world model from real environment interactions, which then substitutes the actual environment. Subsequent policy learning can take place in the world model, obviating the need for further interaction with the real environment.
In practice, we also continue reinforcement learning after pretraining in the world model. Formally, we learn a policy π : O * → A that inputs an observation history and outputs an action.
Alternatively, world models can be used for lookahead planning. Given a reward function R, we plan for a horizon of H timesteps by searching for an optimal action sequence, given our previous observations o 1:t and actions a 1:t-1 :
a * t:t+H = arg max a t:t+H E p θ (o t+1:t+H |o1:t,a 1:t+H ) H-1 k=0 R(o t+k+1 ; o 1:t+k , a 1:t+k )
where p θ (o t+1:t+H |o 1:t , a 1:t+H ) = H-1 k=0 p θ (o t+k+1 |o 1:t+k , a 1:t+k ).
To get to the key (goal), the agent needs to perform a very long action sequence (length > 100) with no intermediate reward
Thus, we build an abstract graph. The nodes are defined by object contact, and the edges connect pairs of nodes that a low-level planner can find a plan to traverse between.
The task is much easier if it is broken down into a sequence of sub-tasks (high-level plan).
Thus, we can search in this graph for possible high-level plans. An agent can now choose a high-level plan to follow.
Some high-level plans might not work. When an agent acts in the environment and discovers that the plan does not work, we remove the false edge from the graph and update the world model.
Eventually, we find a high-level plan that works 1. 2. 3.
this section cite: []

Section: 5. 6.
Figure 3: A sequence of illustrations that demonstrates how our hierarchical planner works.
To help our world models guide long-horizon decision-making, we implement a hierarchical planner inspired by task and motion planning (TAMP). It first plans in a high-level abstract state space defined by object contact, then lowers those plans into actual Atari button presses using the learned world model (Figure 3, Appendix A.2). The resulting agent we denote PoE-World + Planner.
this section cite: []

Section: Experimental Results

this section cite: []

Section: Domains and Evaluation.
We evaluate our agent, PoE-World + Planner, against other methods on Atari's Pong and Montezuma's Revenge (MR), using the Arcade Learning Environment [28]. We use OCAtari [29] to parse each image frame as a list of objects, each with an object category, a bounding box, and velocities. 1 Both games are partially observed: the current state and action cannot uniquely determine the next state. A demonstration of fewer than 1000 frames is created for each game, but these demonstrations are not successful gameplays: they serve only to illustrate the core causal mechanics. In Montezuma's Revenge, our demonstration never achieves positive reward.
To test compositional extrapolation , we created alternative versions of both games, called Pong-Alt and Montezuma's Revenge-Alt (Figure 4), both of which recombine and rearrange the types of objects seen in the training demonstration. Pong-Alt increases the number of objects (3 balls and 3 enemies). Montezuma's Revenge-Alt adds more enemies (which the player has to jump over) and ladders, while changing the map to resemble the game Kangaroo (see Appendix A.3 for details). We do not provide demonstrations for these alternative versions of the games.
this section cite: ['b27', 'b28']

Section: Method

this section cite: []

Section: Score

this section cite: []

Section: Pong Pong-Alt MR MR-Alt
Random Agent -20.67 ± 0.33 -20.00 ± 0.58 0.00 ± 0.00 0.00 ± 0.00 PPO @ 100k env steps [31] -21.00 ± 0.00 -18.66 ± 0.88 0.00 ± 0.00 0.00 ± 0.00 LLM as Agent (ReAct) [32] -20.00 ± 0.00 -20.67 ± 0.33 0.00 ± 0.00 0.00 ± 0.00 WorldCoder + Planner [3] -17.00 ± 3.00 -19.00 ± 1.00 0.00 ± 0.00 0.00 ± 0.00 PoE-World + Planner (Ours) -12.33 ± 0.88 -13.67 ± 0.67 100 ± 0.00 100 ± 0.00 PPO @ 20m env steps [31] 17.00 ± 0.58 1.33 ± 2.03 0.00 ± 0.00 0.00 ± 0.00
Table 1: Scores on Pong and Montezuma's Revenge (MR) and their alternate versions. For PoE-World and WorldCoder, brief demonstrations on Pong and MR are given to initialize the world models.
Their agents then train for at most 3k steps before evaluation.
Baselines. PPO [31] is a go-to, widely-used model-free RL algorithm. It optimizes a lower bound of the policy's performance using gradient descent. LLM as Agent (ReAct) [32] directly uses an LLM as a policy. ReAct prompts LLMs to use extra chain-of-thought [33] "thinking" actions before selecting an action. WorldCoder [3] is an LLM agent that models the world as a single Python program. It uses an LLM code generation and repair algorithm called REx [34] to refine its world model to achieve high predictive accuracy on observed trajectories. More details in Appendix A.4.
Agent Results. Figure 5 and Table 1 show the scores of different agents on Pong, Pong-Alt, Montezuma's Revenge, and Montezuma's Revenge-Alt. The scores on Pong and Pong-Alt indicates the difference in points achieved by the player and the enemies when the game ends at 21 points. The scores on Montezuma's Revenge and Montezuma's Revenge-Alt become positive if and only if the agent succeeds in collecting the key. As shown in Table 1, our agent, PoE-World + Planner, performs best across all environments in the low-data regime, particularly when PPO baseline is allowed to have 100, 000 training environment interactions, the standard budget for sample-efficient agents on Atari 2600 [20]. In Figure 5, we keep training PPO for more steps, finding that it takes over a million steps for PPO to surpass PoE-World + Planner. Moreover, PPO with 20M training steps never achieves positive score on Montezuma's Revenge. PoE-World + Planner is the only method that manages to obtain positive reward on Montezuma's Revenge in both base and alternative versions.  5M 10M 15M 20M Training Step 20 0 20 40 Max Score Max Score of PPO w/ and w/o Pre-training (Pong) PPO w/o Pre-training PPO w/ Pre-training on PoE-World's world model Instead of planning, can we use PoE-World to learn a policy? Training a policy avoids the test-time compute cost of planning. In Figure 6, we show that pre-training a policy inside PoE-World's world model accelerates policy learning: we first run PPO "in simulation" (in our world model), and then fine-tune in the actual Atari environment. The fine-tuned PPO achieves significantly higher score than vanilla PPO at most training steps, and while the vanilla PPO takes 1M training steps to do better than a random agent, the fine-tuned PPO takes only 200k training steps. Asymptotically the pretrained and randomly-initialized policies converge to the same value: World-model pretraining is effectively a way of warm-starting policy training.
this section cite: ['b30', 'b31', 'b2', 'b30', 'b30', 'b31', 'b32', 'b33', 'b19']

Section: Montezuma's Revenge-Alt Pong-Alt
Next Observation Prediction Results. Table 2 and Table 3 show next observation and next observation's object attributes prediction accuracies under the symbolic world modeling approaches, respectively. PoE-World outperforms the baselines on most settings, except for the test observation (random frames) for Pong where all methods perform similarly because a random agent rarely succeeds in hitting the ball, and so knowledge of game mechanics is not comprehensively tested.
The role of hard constraints. We investigate the role of hard constraints in our world model representation in Table 4. In practice, hard constraints act to rule out "physically impossible" scenes. PoE-World ended up enforcing constraints just for MR, since most possible scenes in Pong are already physically possible. As shown in Table 4, removing hard constraints causes the agent's performance  Table 4: Planning successes (out of 9 tries), where the goal is for the player to grab the key, and next observation prediction accuracies on 1000 random test frames of our agent with and without hard constraints on Montezuma's Revenge (MR) and its alternate version (MR-Alt).
to drop on both the base and alternative versions of Montezuma's Revenge. Interestingly, we found no significant changes in next observation prediction accuracies. We hypothesize that hard constraints do not necessarily turn bad predictions into good ones. Instead, they perform damage control-they refine poor predictions just enough to make them usable for long-horizon planning.
Qualitative difference between WorldCoder's and PoE-World's world models. World models produced by PoE-World consists of 4000+ lines of code for Montezuma's Revenge compared to WorldCoder's less than 100 lines. This difference is reflected in their ability to capture the underlying causal laws: PoE-World models accurately represent important causal laws like character movement constraints with respect to platforms and ladders, while WorldCoder fails to model these mechanics, predicting that the player can "fly" around the map without any constraints. Moreover, WorldCoder models tend to hallucinate, e.g., imagining nonexistent bullet firing abilities in Montezuma's Revenge, potentially because there is no granular way to downweigh buggy parts of a world model. PoE-World, in contrast, prunes irrelevant experts with low weights, yielding more precise world models.
this section cite: []

Section: Related Work
World models as programs has been explored in several recent works that motivate PoE-World. Similar to our work, WorldCoder [3] and CodeWorldModels [24] use LLMs to write a Python transition function. Unlike our work, they learn a single monolithic program, which severely limits scalability: our world models have an order of magnitude more code needed to model complex environments, as well as handle partial observability and non-determinism. Recent works have also explored synthesizing high-level abstract world models as programs [35,36] to support robotic planning by integrating visual perception with symbolic reasoning. They could synergize with our work, as we have focused on learning low-level world models describing the motion of objects.
Earlier works [37,38] learn world models in restricted (non-Turing complete) domain specific programming languages, performing well on benchmarks co-designed with their Domain Specific Language. Even earlier, Schema Networks learn a conceptually related factor-graph world model [39]. AIXI [40] is a theoretical model of reinforcement learning which considers all possible Turing machines, mathematically related to all these works. Interestingly, AIXI also works with a history space as a conceptually elegant way of modeling partial observability. Other studies generate programs for world models primarily from natural language instructions, rather than from example interactions with the environment [41,42,43,44].
Hierarchical Planning. Our approach to segmenting complex continuous tasks into symbolic states is inspired by hierarchical [45] and task-and motion [46,47] planning. Hierarchical representations in Reinforcement Learning (RL) are often expressed as options-temporal abstractions that allow agents to reason at multiple time scales by grouping sequences of primitive actions [48]. Options can be manually specified by design or learned from experience [49]. Task and Motion Planning (TAMP) automatically discover symbolic states and action abstractions that enable generalizable symbolic plans grounded in continuous motion, however the applications of TAMP to Atari-style domains are still lacking due to complex dynamics. Similar to recent work [29,50], we define a high-level state space in terms of object contact relations-which is key to supporting symbolic abstraction.
Alignment with human cognition. Our system, composed of simpler, specialized programmatic experts that give rise to complex behavior, is inspired by a view of mind as a community of interacting agents --a recurring theme in philosophy and cognitive science [5,6,52,7]. Our modeling approach to modeling objects and actions aligns with empirical studies of event segmentation [53,54] and hierarchical planning [55,56]. Hierarchical state-spaces based on motion cues, such as ours, predict how people draw [13] and interpret social interactions [57], attesting to the cognitive alignment of our approach. Likewise, our programmatic representations of actions align with studies that demonstrate human concept learning to be akin to mental programs [58,59,11,12], which recent work models by LLM-based program synthesis.
this section cite: ['b2', 'b23', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b28', 'b49', 'b4', 'b5', 'b51', 'b6', 'b52', 'b53', 'b54', 'b55', 'b12', 'b56', 'b57', 'b58', 'b10', 'b11']

Section: Discussion and Limitations
Compositionality. Recombining pieces of knowledge to generalize and extrapolate is a core feature of symbolic systems and, arguably, also of human cognition, ranging from natural language to abstract reasoning to everyday thought [60,52,61,62]. Our approach is in this spirit. It factors its knowledge into small experts whose predictions can combine to extrapolate to scenes with more objects recomposed into new arrangements. More formally, our approach generalizes to novel "entity compositions" and "relational compositions," terms used by [63]. It should be noted, however, that our compositional factoring is orthogonal to our use of symbolic code: Programs can be monolithic [3], and neural nets can be factored [64]. Nonetheless, compositionality proved critical to scalably learning the underlying symbolic program. Synthesizing a single monolithic program is, in our view, intractable not just for the real world, but even for Atari.
Limitations. We make important assumptions, and only address part of the full model-based reinforcement learning problem. Symbolic programs expect symbolic inputs: We do not learn straight from pixels. RL involves exploration, decision-making, and reward function learning, but we do not address those problems here. However, we speculate our approach could unlock better methods for exploration: A program-structured world model exposes an interpretable interface for describing beliefs about how the world works, and efficiently exploring the world is analogous to testing the program that encodes the world model. Therefore ideas from software testing and program analysis could, in theory, be brought to bear, enabling new approaches to model-based exploration.
this section cite: ['b59', 'b51', 'b60', 'b61', 'b62', 'b2', 'b63']

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claims made in the abstract and introduction accurately reflect the paper's contributions and scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The paper discusses the limitations of the work performed by the authors.
Guidelines:
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA]
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We have provided OpenAI API cost for our experiments and compute resources used for our experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conducted in the paper conforms, in every aspect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: Our work is foundational research and has no direct societal impact.
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
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: The creators of original data are all properly credited, and the license and terms of use of the data are explicitly mentioned and properly respected.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [Yes] Justification: This paper describes the usage of LLMs in our method.
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: A Appendix

this section cite: []

Section: A.1 PoE-World algorithm details
As mentioned in Section 3.2, our learning algorithm PoE-World alternates between two main steps: programmatic expert synthesis and gradient-based weight optimization. Given a trajectory, we do batch processing of size 10-we first learn p θ (o 11 |o 1:10 , a 1:10 ), then p θ (o 21 |o 1:20 , a 1:20 ), and so on.
this section cite: []

Section: Programmatic expert synthesis.
To synthesize programmatic experts, we first need to turn the observations into natural language observations that can be input into LLMs. A sequence of consecutive observation transitions (o t:t+H+1 , a t:t+H ) can be transformed into a text showing the first input object list, the list of actions, and the changes to the input object list for the following timesteps, all in natural language. Table 5 shows an example text representation.
We note that in the text representation, only changes to a single object type (which is 'player' in Table 5) is displayed at a time. This choice is to enforce modularity between the object types: each experts put non-uniform distribution only on a single attribute of objects with a specific object type. This means Equation ( 2) can be further rewritten as:
p θ (o t+1 |o 1:t , a 1:t ) ∝ obj-type i p obj-type_expert i (o t+1 |o 1:t , a 1:t ) θi(6)
where {p obj-type_expert i } are the experts associated with each object type.
We then implement multiple (10 in total) LLM-based synthesis modules. For each object type, these modules take as input a sequence of consecutive observation transitions and output a set of programmatic experts. The output programmatic experts are pooled into a single set once all modules finish running. Having multiple synthesis modules allows each module to focus on different aspects of the environment, e.g., how objects move passively, how object moves when interacts with objects, how objects get created and deleted, etc. We describe one of our modules, ActionSynthesizer, below.
ActionSynthesizer focuses on synthesizing experts that explain how each action affects objects when the objects are interacting (touching) other objects (see example experts of this kind in Figure 2). It takes in just a single transition (o t:t+1 , a t ). It turns this transition into a text representation as discussed above (Table 5). Then, it prompts a LLM to output causal explanations for the object changes (see the prompt Table 6). An example causal explanation is "the player objects that touch an ladder object set their y-axis velocity to -4". With a set of causal explanations, we prompt a LLM to turn each of them into a program (see the prompt Table 7 and Table 8).
Other modules follow this template of first prompting a LLM for natural language causal explanations and then prompting a LLM to turn the explanations into programs. We refer the reader to our code at https://github.com/topwasu/poe-world for the implementations of other LLM-based module.
The programs synthesized by LLMs use our manually-written helper classes: Obj, ObjList, RandomValues, and SeqValues. Their docstrings are passed to the prompt. Obj provides a method Obj.touches which is the function we manually build in to help determine object contact. Obj Example input list of objects: player object (id = 0) with x-axis velocity = +4, Interaction --player object (id = 0) is touching ladder object (id = 2), Interaction --player object (id = 0) is touching unknown object (id = 4), Example list of actions: NOOP, NOOP, RIGHT Example output list of object changes: -The player object (id = 0) sets x-axis velocity to [+0, +0, +2]
Table 5: Example text representation of a sequence of consecutive transitions.
contains object attributes: object category, x/y position, x/y velocities. It also contains properties including center_x, center_y, left_side, right_side, which are calculated based on the position and velocities, and their setter methods actually modify the velocities under the hood. ObjList is a class that represents an object list. It has a method ObjList.get_objs_by_obj_type and ObjList.create_object. Lastly, RandomValues and SeqValues are the classes used to mark values set by a LLM. The purpose of these two classes is further discussed below.
After the programmatic experts are synthesized (see examples in Table 9), we interpret them as distributions as discussed in Section 3.2. We tell LLMs in the prompt to set attribute values as instances of RandomValues, as opposed to integers. This marks the attributes whose values changed by a LLM so that we can write an program-to-distribution interpreter that puts a single-peak distribution on the attribute whose value is set and uniform distributions on all other attributes. SeqValues is similar to RandomValues, but is used in multi-timestep predictions scenario, as discussed in Section 3.1.
The LLM used in the steps above is gpt-4o-2024-08-06. We implemented a disk cache for the LLM responses to avoid paying multiple times for the same prompts and seeds.
Hard constraints. The hard constraints (Section 3.1) are also synthesized similar to how we synthesize the programmatic experts. Table 10 shows the constraints learned for Montezuma's Revenge. In Equation ( 3), we choose to use a disjunction rather than a conjunction because the physics in video games can be peculiar and unrealistic-a player might have their body overlap with a platform when climbing down a ladder attached to that platform. In the real world, we believe a conjunction would be a better choice.
Gradient-based weight optimization. Once we have the expert distributions {p expert i }, we can optimize their weights {w i } according to Equation (5). We use the L-BFGS optimizer [26] implemented in PyTorch [76] with strong Wolfe line search, learning rate = 1, number of epochs = 4, and without mini-batching. We also include a L1 regularization loss with weight = 1 so that the weights do not get too big.
We note that the weight optimization is done without taking into account the hard constraints since we would like to use gradient-based approaches.
We prune programs with weights lower than δ = 0.01 after the weight optimization is done, and we prune constraints that contradict with the observations or explain less than 1% of the observations.
this section cite: ['b4', 'b25', 'b75']

Section: A.2 Planner
In a game like Montezuma's Revenge, planning in the actual, low-level action space is hard because the number of actions required to get to the first positive reward is very high-at least 100. This means the size of search space is 8 100 as there are 8 possible actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE.
Thus, inspired by task and motion planning (TAMP), our hierarchical planner interleaves planning in the low-level motion action space with planning in a high-level abstract action space. It learns an abstract graph where the nodes are abstract states defined by object contact, and the edges represent whether the world model believes the player can traverse between the two nodes. Then, we search for a path in the abstract graph that takes the player to the goal. In Atari, a goal is usually for a player to touch a goal object (key, ball, platforms, etc.). The discovered path in the abstract graph is a high-level plan-a sequence of subgoals-for a low-level planning agent. A low-level planning agent performs online planning to choose an action and then execute it in the environment.
The planning algorithm can be described step-by-step as follows:
Step 1: Learn an abstract graph by running a low-level motion planner in simulation on all pairs of nodes. A (ordered) pair of nodes has an edge between them if we can find at least one low-level plan to traverse between them. Step 2: Search for a path in the abstract graph. We use breadth-first search (BFS) here to find a path to the goal with the shortest length. The path is our high-level plan. If there is no path to the goal, go back to step 1. Step 3: Attempt to follow the high-level plan, completing each subgoal in order, with a low-level planning agent.
Step 4: If there is a "false" edge, update the world model, remove that edge from the graph, then go back to step 2. Otherwise, the algorithm stops, and the agent has achieved the goal.
Figure 3 shows simplified illustrations of how our hierarchical planner works.
We now discuss the implementation of our low-level motion planner:
Low-level motion planner. We implemented two low-level motion planners. The first is a variant of Monte Carlo Tree Search (MCTS) [77,78]. It follows the same set of procedures as vanilla MCTS with two differences: first, similar to the MCTS algorithm used in [3], it approximates the value of a node using a heuristic function instead of doing a random rollout in the simulate step. The heuristic function is the Manhattan distance between the current position of the player object and the position of a goal object. We find that this function is a good estimate of how good an observation is when trying to achieve a goal. Second, the value of a node is updated as the maximum value of its children nodes, instead of the expected value, in the backpropagation step. Intuitively, using the maximum value encourages the planner to be more optimistic. The exploration parameter for MCTS is initially equal to 1 for Montezuma's Revenge and Montezuma's Revenge-Alt and 10 for Pong and Pong-Alt, and it increases by 10 times every 1000 iterations of MCTS.
We also implement "sticky actions" by extending the action space that MCTS searches on so that it includes repeated sequences of primitive actions with lengths 1, 4, and 8. Thus, the extended action space has 3n actions where n is the number of primitive actions. We include these repeated action chunks in our action space to make planning easier. Playing Atari games rarely requires players to change actions at every timestep, so repeated action chunks can be helpful.
The second low-level motion planner is a greedy search with the same heuristic function used in MCTS. At each iteration, it greedily finds the best repeated action chunk of length 8 and includes it in the plan. It backtracks if the current state leads to death no matter which action chunk of length 8 the planner chooses. It returns a plan when the player achieves the goal (touches the goal object).
Pong, Pong-Alt, and Montezuma's Revenge-Alt agents only use the greedy planner, while Montezuma's Revenge uses both: it first tries to find a plan with MCTS for 4000 iterations and falls back to the greedy planner if MCTS fails.
As discussed earlier, the low-level motion planner is used in two steps: to build the abstract graph by finding a plan to traverse between two abstract nodes in the world model, and to help inform a low-level planning agent that is trying to complete a subgoal.
this section cite: ['b76', 'b77', 'b2']

Section: Low-level planning agent.
The agent performs online planning: it uses the low-level planner to find a plan to the goal, and then it takes a sequence of actions and replans. The agent replans when the ccurent plan no longer takes the agent to the goal, so the agent may take several actions before replanning. We further optimize the agent by letting it replan only 40% of the times when the current plan no longer works in Montezuma's Revenge.
Because of this introduced stochasticity, however, we find that our whole planning pipeline can give different scores in different runs even with the same initial world model, so we treat running the hierarchical planner multiple times with the same initial world model as part of training, and we run the planning algorithm 3 times on the same initial world model to get results on Montezuma's Revenge and Montezuma's Revenge-Alt. Our work focuses on world modeling, and we leave it to future work to increase the efficiency and performance of the hierarchical planner.
this section cite: []

Section: A.3 Domain details
We evaluate our methods on Atari's Pong and Montezuma's Revenge using the Arcade Learning Environment (ALE). The frameskip parameter is set to 3 for both games. We use OCAtari to parse each image frame as a list of objects, each with an object category, a bounding box, and velocities. OCAtari reverses-engineers the RAM values of Atari games to get the bounding boxes of each object.
We modify OCAtari to fix a number of issues in its handling of Pong and Montezuma's Revenge. This includes fixing bugs in object detection so that it fully detects every object in play, correcting bugs in the the bounding boxes so that object interactions correctly correspond to when object bounding boxes touch/overlap, etc. We refer the reader to our code implementation https://github.com/  topwasu/poe-world for full details.
Details on how we create Pong-Alt and Montezuma's Revenge-Alt are below:
Pong-Alt is created by layering three Pong environments. We sync the player's location in all three environments, so that it appears as if we have only one paddle. The enemies and balls, on the other hand, are all at different locations. We end up with one player, three enemies, and three balls.
Montezuma's Revenge-Alt is created by stacking the lower section of the first room in the original Montezuma's Revenge to make three platforms of different heights, connected by stairs. We stack up three lower sections of three different Montezuma's Revenge environments. The first and the third section in the stack are flipped horizontally so that the stairs are on different sides of the room, requiring the player to jump over the skulls to reach the stairs.
ALE code uses GPL-2.0 license, and OCAtari code uses MIT license.
A.4 Baseline details PPO. PPO uses frame stacking = 4. We use the same hyperparameters as the PPO paper [31] and OCAtari [29]. We use the stable-baselines3 implementation of PPO with MLP backbone ('MlpPolicy') [79]. For the Alt environments, we take the PPO model pretrained 20M steps on the base environments and finetune it on the Alt environments. This pretraining process is done for fiar comparison with our method, since our method assumes demonstrations from the base environments when evaluating on the corresponding Alt environments.
ReAct. We write prompts that would alternate between thinking and taking actions, one for Pong and Pong-Alt and another for Montezuma's Revenge and Montezuma's Revenge-Alt. The frame observation is transformed into text where we provide each object's x and y position and a list of all pairwise object interactions. This text observation is input as part of the prompt. In the prompt, we only include the 4 most recent observations (along with the 4 most recent thinking actions and 4 most recent taken actions) as each observation is quite long in text.
WorldCoder. We use the official WorldCoder implementation [3] but replace the existing prompts with new ones that are tailored towards our text representation of object-centric Atari frames. The prompts can be found in our codebase https://github.com/topwasu/poe-world. The instantiation of WorldCoder in [3] actually has its own planner implemented, but for fair comparison with our method, we use only the world modeling part of that system and combine it with our own planner so that the planner is the same for both WorldCoder and our method. We choose to use our own planner instead of theirs since ours is hierarchical.
this section cite: ['b30', 'b28', 'b78', 'b2', 'b2']

Section: WorldCoder code uses MIT license.

this section cite: []

Section: A.5 Compute resources and execution time
Compute Resources. For the world modeling part, our experiments are run on 4 CPUs (Cascade-Lake, IceLake, or SaphireRapids) with 64 GB memory. PoE-World and WorldCoder uses a budget of $20 worth of OpenAI credit per run. For the planner, we also run it mostly on 4 CPUs, but for the part where we need to build an abstract graph by running many low-level planners, we parallelize it on multuple compute jobs on a job scheduling cluster, so we might be using 100 CPUs at a time.
this section cite: []

Section: Execution time.
PoE-World alone without the planner tends to take around 8 hours to run (this includes the time we need to wait for OpenAI LLM requests). The planner running time varies, but most runs finish under 24 hours.
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects Here's an example with player objects: Example input list of objects: player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2, Interaction --player object (id = 0) is touching ladder object (id = 2), Interaction --player object (id = 0) is touching unknown object (id = 4), Example output list of object changes: -The player object (id = 0) sets x-axis velocity to +0 -The player object (id = 0) sets y-axis velocity to -4
Example reasons:
1. The player objects that touch an unknown object set their x-axis velocity to +0 2. The player objects that touch an unknown object set their y-axis velocity to -4 3. The player objects that touch an ladder object set their x-axis velocity to +0 4. The player objects that touch an ladder object set their y-axis velocity to -4 Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.
Input list of objects: {input} Output list of object changes: {effects} Please follow these rules for your output: 1. make sure each reason only talks about one object change 2. do not talk about IDs Table 7: First half of a prompt in ActionSynthesizer used to turn the natural language causal explanations into programs.
Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of action '{action}' following these rules: 1. Each function should make changes to one attribute --this could be the x-axis position, y-axis position, creation of object, or deletion of object. 2. Always use RandomValues to set attribute values. If there are conflicting changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute. 3. Use Obj.touches to check for interactions. 4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop. 5. You can assume the velocities of input objects are integers. 6. Please use if-condition to indicate that the effects only happen because of action '{action}' Format the output as a numbered list.
Table 8: Second half of a prompt in ActionSynthesizer used to turn the natural language causal explanations into programs. Montezuma's Revenge-Alt Pong Pong-Alt Montezuma's Revenge
this section cite: []

Section: 
Justification: The paper does not include theoretical results. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We have fully described the algorithms used in the paper, with details included in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes]
Justification: The code is available at https://anonymous.4open.science/r/  poe-world-neurips-628F/ Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: Algorithm details are described in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes]
Justification: The paper reports error bars computed over 5 seeds.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper currently does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. Table 10: The collection of constraints synthesized for Montezuma's Revenge. Function names have been shortened to save space.
this section cite: []

Section: References
Ref_id:b0 Title: Mastering diverse domains through world models Year: (2023)
Ref_id:b1 Title: Human learning in atari Year: (2017)
Ref_id:b2 Title: Worldcoder, a model-based LLM agent: Building world models by writing code and interacting with the environment Year: (2024)
Ref_id:b3 Title: The modularity of mind Year: (1983)
Ref_id:b4 Title: Society of mind. Simon and Schuster Year: (1986)
Ref_id:b5 Title:  Year: (1993)
Ref_id:b6 Title: The algebraic mind: Integrating connectionism and cognitive science Year: (2003)
Ref_id:b7 Title: Cognitive neuroscience. the biology of the mind Year: (2025)
Ref_id:b8 Title: Acquisition of cognitive skill Year: (1982)
Ref_id:b9 Title: Automated discovery of symbolic laws governing skill acquisition from naturally occurring data Year: (2024)
Ref_id:b10 Title: Human-level concept learning through probabilistic program induction Year: (2015)
Ref_id:b11 Title: People infer recursive visual concepts from just a few examples Year: (2020)
Ref_id:b12 Title: Learning abstract structure for drawing by efficient motor program induction Year: (2020)
Ref_id:b13 Title: Learning evolved combinatorial symbols with a neuro-symbolic generative model Year: (2021)
Ref_id:b14 Title: Symbolic metaprogram search improves learning efficiency and explains rule learning in humans Year: (2024)
Ref_id:b15 Title: Drawing out of distribution with neuro-symbolic generative models Year: (2022)
Ref_id:b16 Title: Optimal control of markov processes with incomplete state information i Year: (1965)
Ref_id:b17 Title:  Year: (2018)
Ref_id:b18 Title: Mastering atari, go, chess and shogi by planning with a learned model Year: (2020)
Ref_id:b19 Title: Model based reinforcement learning for atari Year: (2020)
Ref_id:b20 Title: Transformers are sample-efficient world models Year: (2023)
Ref_id:b21 Title: Diffusion for world modeling: Visual details matter in atari Year: (2024)
Ref_id:b22 Title: Diffusion model predictive control Year: (2024)
Ref_id:b23 Title: Generating code world models with large language models guided by monte carlo tree search Year: (2024)
Ref_id:b24 Title: Synthesizing world models for bilevel planning Year: (2025)
Ref_id:b25 Title: On the limited memory bfgs method for large scale optimization Year: (1989)
Ref_id:b26 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b27 Title: The arcade learning environment: An evaluation platform for general agents Year: (2013-06)
Ref_id:b28 Title: Ocatari: Object-centric atari 2600 reinforcement learning environments Year: (2023)
Ref_id:b29 Title: First return, then explore Year: (2021)
Ref_id:b30 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b31 Title: React: Synergizing reasoning and acting in language models Year: (2022)
Ref_id:b32 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b33 Title: Code repair with llms gives an exploration-exploitation tradeoff Year: (2024)
Ref_id:b34 Title: Visualpredicator: Learning abstract world models with neuro-symbolic predicates for robot planning Year: (2024)
Ref_id:b35 Title: Learning symbolic operators for task and motion planning Year: (2021)
Ref_id:b36 Title: Human-level reinforcement learning through theory-based modeling, exploration, and planning Year: (2021)
Ref_id:b37 Title: Combining functional and automata synthesis to discover causal reactive programs Year: (2023-01)
Ref_id:b38 Title: Schema networks: Zero-shot transfer with a generative causal model of intuitive physics Year: (2017)
Ref_id:b39 Title: Universal artificial intelligence: Sequential decisions based on algorithmic probability Year: (2005)
Ref_id:b40 Title: Factorsim: Generative simulation via factorized representation Year: (2024)
Ref_id:b41 Title: Bootstrapping cognitive agents with a large language model Year: (2024)
Ref_id:b42 Title: Learning adaptive planning representations with natural language guidance Year: (2023)
Ref_id:b43 Title: Leveraging pre-trained large language models to construct and utilize world models for model-based task planning Year: (2023)
Ref_id:b44 Title: Hierarchical task and motion planning in the now Year: (2011)
Ref_id:b45 Title: Partially observable task and motion planning with uncertainty and risk awareness Year: (2024)
Ref_id:b46 Title: Discovering state and action abstractions for generalized task and motion planning Year: (2022)
Ref_id:b47 Title: Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning Year: (1999)
Ref_id:b48 Title: The option-critic architecture Year: (2017)
Ref_id:b49 Title: Deep reinforcement learning via object-centric attention Year: (2025)
Ref_id:b50 Title: Predictive state representations: A new theory for modeling dynamical systems Year: (2012)
Ref_id:b51 Title: Building machines that learn and think like people Year: (2017)
Ref_id:b52 Title: Modeling human activity comprehension at human scale: prediction, segmentation, and categorization Year: (2024)
Ref_id:b53 Title: Event perception and memory Year: (2020)
Ref_id:b54 Title: Approximate planning in spatial search Year: (2024)
Ref_id:b55 Title: Neural mechanisms of hierarchical planning in a virtual subway network Year: (2016)
Ref_id:b56 Title: Adventures in flatland: Perceiving social interactions under physical dynamics Year: (2020)
Ref_id:b57 Title: Cognitive maps are generative programs Year: (2025)
Ref_id:b58 Title: Map induction: Compositional spatial submap learning for efficient exploration in novel environments Year: (2022)
Ref_id:b59 Title: Faith and fate: Limits of transformers on compositionality Year: (2023)
Ref_id:b60 Title: Human-like systematic generalization through a meta-learning neural network Year: (2023)
Ref_id:b61 Title:  Year: (2007)
Ref_id:b62 Title: Neurosymbolic grounding for compositional world models Year: (2023)
Ref_id:b63 Title: International Conference on Learning Representations Year: (2021)
Ref_id:b64 Title: Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents Year: (2018)
Ref_id:b65 Title: Hypothesis search: Inductive reasoning with language models Year: (2024)
Ref_id:b66 Title: Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement Year: (2023)
Ref_id:b67 Title: Doing experiments and revising rules with natural language and probabilistic reasoning Year: (2024)
Ref_id:b68 Title: Overcoming the expressivity-efficiency tradeoff in program induction Year: (2024)
Ref_id:b69 Title: Genie: Generative interactive environments Year: (2024)
Ref_id:b70 Title: Training products of experts by minimizing contrastive divergence Year: (2002)
Ref_id:b71 Title: Integrated task and motion planning Year: (2021)
Ref_id:b72 Title: RoboDreamer: Learning compositional world models for robot imagination Year: (2024-07)
Ref_id:b73 Title: Compositional visual generation with energy based models Year: (2020)
Ref_id:b74 Title: Compositional Diffusion-Based Continuous Constraint Solvers Year: (2023)
Ref_id:b75 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b76 Title: Efficient selectivity and backup operators in monte-carlo tree search Year: (2006)
Ref_id:b77 Title: A survey of monte carlo tree search methods Year: (2012)
Ref_id:b78 Title: Stable-baselines3: Reliable reinforcement learning implementations Year: (2021)
