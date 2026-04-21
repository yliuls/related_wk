Title: Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
Abstract: To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints.

Section: Introduction
Humans intuitively prioritize and retain memories relevant to their current tasks, filtering out irrelevant details such as the color of a house's walls along a route unless it serves as a crucial navigation landmark. Similarly, reinforcement learning (RL) agents must learn to selectively retain and access task-specific memories guided by task objectives rather than predefined rules. This ability is especially critical for long-horizon tasks, where delayed rewards make credit assignment challenging and necessitate efficient training and inference over extended timescales to effectively capture the relationship between past and future events.
Recently, Transformers have become the go-to framework for sequence modeling due to their flexible and expressive encoding, free from the fixed-size constraints of recurrent neural networks (RNNs) [8,14]. However, it assumes access to all past information at each step rather than selectively retaining relevant memories, which can lead to inefficiencies and make it harder to extract task-relevant information for decision-making. Their quadratic attention complexity further limits scalability, making it challenging to process long contexts efficiently, especially when gradients must propagate over large sequences. At test time, Transformers face another practical challenge: storing an
this section cite: ['b7', 'b13']

Section: Related Work
In this section we briefly review research on extending context limits of transformers in language modeling, as well as the state of the art in long sequence modeling and decision making in RL.
this section cite: []

Section: Extending Transformer Contexts in language:
A significant limitation of current large language models is their restricted input context length, prompting research into methods for scaling and generalizing to larger contexts [33,21,13,27]. Key-value (KV) caching can reduce recomputation at training [9] or inference time [23], by storing and reusing self-attention outputs during decoding. [25] further reduce memory by maintaining a compressed cache, though this compression is not completely task-guided.
Beyond compute efficiency, recent work explores dynamic context extension. Recurrent Memory Transformer (RMT) [4] introduces summarization tokens to periodically compress and propagate prior context while discarding older tokens. Autocompressors (AC) [7] extend this by accumulating generated summaries across context windows, avoiding RMT's fixed-size memory but still truncating gradient propagation through summaries. We extend this context summarization approach to the more involved setting of reinforcement learning (RL), where summarization must support decision-making and handle credit assignment over long horizons. Unlike AC, which fine-tunes pretrained models on a supervised token prediction task with limited gradient propagation-and aims to match but does not surpass full-context transformer baselines-we train from scratch, integrating summarization directly into the RL optimization process and propagating gradients across all summaries. This enables more effective task-driven memory formation, surpassing full-context transformer baselines in both efficiency and scalability.
Transformers in RL: Transformers, while widely used in self-supervised language modeling [10,3], have seen slower adoption in reinforcement learning (RL) due to challenges like sparse rewards, optimization instability, and limited batch sizes in on-policy RL. However, as we move to more The frames O 1-3lseg depict the input sensor observations to the agent at timesteps 1 -3l seg . The figure depicts the information flow between three consecutive segments of a much longer context of inputs provided to a transformer during training. The summary tokens bottleneck the information passed on from one chunk of inputs to the next. complex tasks that demand long-horizon planning and reasoning over past experiences, transformers have shown promising initial results, outperforming recurrent models in POMDPs when encoding longer temporal dependencies [22].
To stabilize transformer training in partially observable RL, Parisotto et al. [24] used KV caching while restricting gradient propagation to a small window of time steps, limiting the learning of longterm dependencies. More recently, ReLIC [11] improved convergence of on-policy RL algorithms by unfreezing the historical context within a rollout and applying frequent updates, while AMAGO [12] adapted off-policy RL to train with a shared transformer backbone. While these methods enhance long-horizon learning, ReLIC's high computational cost and slow training hinder scalability, and AMAGO is limited to small state-space observations, reducing its applicability to more complex tasks. Our approach addresses these challenges by enabling efficient memory utilization and leveraging a more scalable transformer architecture for extreme long-horizon decision-making.
this section cite: ['b32', 'b20', 'b12', 'b26', 'b8', 'b22', 'b24', 'b3', 'b6', 'b9', 'b2', 'b21', 'b23', 'b10', 'b11']

Section: Methodology
This section outlines our methodology for memory-augmented in-context reinforcement learning (RL). We first formalize the problem and describe the in-context RL algorithms and the sequence model we consider, in Section 3.1. We then introduce our context summarization mechanism in Section 3.2, and describe key implementation details including the attention masking, positional encoding, segment randomization and cache management scheme we use.
this section cite: []

Section: Problem Setting
In this work, we study tasks that require a model to perform sequential decision-making over long executions while leveraging its experience history to improve efficiency over time. Many diverse reasoning problems fall into this category, ranging from embodied navigation to language modeling. We formalize these tasks within the framework of in-context reinforcement learning (RL), where models adapt and improve their performance using past inferences rather than pre-collected supervised datasets. By treating these tasks as partially observable Markov decision processes (POMDPs) and transforming them into fully observable MDPs through the integration of historical context, it is possible to train policies effectively using both on-policy and off-policy RL methods.
We use a sequence encoder to model the agent's policy, processing past observations into a contextual representation for decision-making. At each timestep t, the sequence encoder receives a sequence of observations, denoted as X t = {o 1 , . . . , o t }, and maps them to hidden representations h t = SeqEnc(X t ). The policy distribution and value estimate are then computed using learnable actor and critic heads as π(a t | h t ), V (h t ), respectively, where a t ∈ A is the action taken by the agent at timestep t, selected according to the policy π over the action space A.
In-Context RL Algorithms: In in-context RL, an agent adapts over time by conditioning on its entire past within a trial. A trial consists of multiple episodes, where each episode resets upon termination, but the agent retains memory across episodes. These episodes share a common underlying context-such as environment dynamics, task structure, or goal distribution-enabling the model to leverage past experiences for more efficient adaptation. Sequence models like Transformers encode history by attending to all previous timesteps within a trial, allowing the agent to infer temporal dependencies and adjust its behavior accordingly. However, as trial lengths increase, attending to all past timesteps becomes computationally infeasible.
To address this, we introduce Memo, a method for constructing and accessing memory representations in Transformers to improve long-term context management in RL, as described in Section 3.2. To demonstrate its efficacy and broad applicability to long-context RL, we integrate Memo with both on-policy and off-policy RL method:
• On-policy RL: We incorporate our memory mechanism into RELIC [11], an adaptation of DD-PPO [30], which applies frequent partial updates during trajectory rollouts to help transformer policies learn more effectively from in-context experiences.
• Off-policy RL: We extend Memo to AMAGO [12], which trains transformer-based policy using a shared sequence encoder and unified actor-critic loss to improve stability and learning efficiency.
this section cite: ['b10', 'b29', 'b11']

Section: Context Summarization in Memo
Following [7], we introduce context summarization as a learned sub-task that enables the transformer encoder to compress and retain task-relevant information from long observation histories. This mechanism allows the model to efficiently process past experiences while optimizing for task rewards in RL. Our approach uses learnable summary embeddings to prompt the transformer to generate summary tokens at predefined intervals, ensuring efficient memory utilization in long-horizon tasks.
Instead of attending to the full sequence history, the transformer periodically compresses past context into summary tokens, which are then fed back into the model in future timesteps. We partition long input sequences into segments of length l seg , and generate l sum summary tokens at the end of each segment. These tokens act as compact memory representations, allowing the model to condition on past experiences without requiring access to raw observations. The encoding and integration of summary tokens into future segments are illustrated in Figure 1, and the pseudocoe is provided in Appendix A.4. The summarization mechanism is trained end-to-end through the RL objective, allowing the model to attend over and refine all previously generated summary tokens. Gradients propagate through the attention mechanism, ensuring that memory updates remain task-driven.
this section cite: ['b6']

Section: Attention Masking:
We use causal masking which includes all the previous summary tokens as well as the previous observations within the segment being processed at time step t. This excludes any previous observation that contributed to and that was processed before the most recent summarization. This creates a bottleneck for the flow of historical information to be solely through summary tokens.
Positional Encoding: To add positional awareness, we assign indices to an observation at time step t based on its relative position within a segment. We use the same positional encoding method (linear or ROPE) used by the baseline implementation which we will be comparing to. The position labeling scheme in a naive full-context transformer input has indices which run from 0 to the length of the context window. In Memo, the input context at time step t in the rollout consists of n * l sum summary tokens and the latest segment's observations, where n = ⌊t/l seg ⌋. The position indices thus range from 0 to n * l sum -1 for the summaries and then start from n * l sum and go to t -n * l seg + n * l sum for the most recent observations in the latest segment. The summary embeddings at the end of a segment are assigned indices following those of the last segment observation.
Segment length randomization: Following Chevalier et al. [7], we randomize segment lengths during training by sampling uniformly within ±20% of a fixed l seg . This approach, which previously showed small improvements in perplexity for token prediction, will be ablated in 4.8. A fixed l seg is used during data collection and evaluation, while training uses randomized segment lengths.
this section cite: ['b6']

Section: Maintaining the KV Cache:
In on-policy RL, the KV cache stored during action selection can become stale after a model update, as the new weights would encode different representations for the same past context. This divergence is problematic since policy decisions depend on context, making consistency crucial. ReLIC addresses this by refreshing and re-encoding the KV cache with the latest model weights after every on-policy update. We extend this approach to memory tokens, ensuring consistency by recomputing the summary vectors alongside the KV cache refresh. This guarantees that both cached representations and learned memory remain aligned with the current policy.
this section cite: []

Section: Experiments
In this section, we present experimental results highlighting different aspects of our proposed method. We first describe the benchmark tasks and baselines used in our experiments, followed by different experimental insights in the following subsections.
this section cite: []

Section: Benchmarks
EXTOBJNAV: The Extended Object Navigation (EXTOBJNAV) task, first introduced in [11], builds on the OBJECTNAV task commonly used in embodied AI research [1,20]. While OBJECTNAV requires an agent to navigate to a single object goal in an episode, EXTOBJNAV focuses on how performance evolves when the agent is repeatedly tasked with reaching different object goals within the same scene. The task evaluates the agent's ability to leverage memory-utilizing information from prior exploration during navigation to reach subsequent goals.
The EXTOBJNAV task uses 37 training and 12 validation scenes from HSSD [16] and includes 20 object instances from the YCB dataset [6]. These objects can be randomly placed on receptacles throughout the scene, with each placement featuring an average of 30 objects. We use 11,100 novel placements during training and 108 during evaluation. A sample scene is shown in Figure 2a.
Agents are trained and evaluated over trials, each consisting of a sequence of episodes with the same object placement in a scene. Each episode mirrors OBJECTNAV in structure: the agent is randomly spawned and tasked with locating an object from a sampled category. Episodes are capped at 500 steps, while trials are limited to 4096 steps during training and 32k steps during evaluation.
The agent is a Fetch robot with a head-mounted camera (256x256 RGB) and an odometry sensor that provides relative position and orientation. It operates in a discrete action space: Forward (0.25m), TurnLeft (0.3°), TurnRight, and Stop. An episode is successful if the agent stops within 2m of the goal with the object in view, occupying at least 10 pixels in the image.
We report success rate (SR) and success weighted by path length (SPL), with SPL measuring navigation efficiency relative to the shortest path. Results are computed over 32k interaction steps on the validation set and plotted at intervals of 500 steps. All experiments use 10 random seeds. Further details are in Appendix A.1. We include results on another multi-goal 3D maze navigation task (Memory Maze) in the appendix.
this section cite: ['b10', 'b0', 'b19', 'b15', 'b5']

Section: Dark-Key-To-Door:
Dark-Key-To-Door [18] is a Meta-RL [2] benchmark where an agent must find an invisible key to open an invisible door in a 9x9 2D gridworld. The agent receives +1 reward for finding the key and the door. It only observes its (x, y) position at each timestep and must remember key and door locations based on previous reward signals.
Each episode lasts up to 50 timesteps, with trial duration fixed at 500 steps. We evaluate performance by average reward across 960 trials and 3 validation seeds. An agent that leverages contextual information can complete multiple episodes per trial and achieve total reward well above 50.
this section cite: ['b17', 'b1']

Section: Baselines
Below, we describe the main transformer-based baselines that we compare Memo to in this work.
• The full context transformer baseline corresponding to ReLIC [11] and AMAGO [12]. We refer to it as FCT in the following sections for brevity.
• Transformer without Inter-Episode Attention (no-IEA): We modify the attention masking such that the model can't attend over previous trials in the rollout. This sets a lower bound for what the performance would be if the model didn't do in-context learning and simply used its current trial's observations to explore and find objects.
• Recurrent Memory Transformers (RMT): We also include a recurrent-only version of Memo that does not accumulate summary tokens. This baseline represents an implementation of RMT [5] adapted for the RL setting. The purpose of this comparison is to evaluate the benefits of Memo's summary accumulation against a fixed-size memory constraint. To ensure a fair comparison, we set the fixed memory size in RMT equal to the total number of summary tokens Memo would accumulate over multiple segments (l sum * number-of-segments).
• Autocompressors (AC): Memo with a pre-trained initialization and highly truncated backpropagation through time TBTT. This represents an implementation of AC [7] tailored to the RL setting.
We exclude RL 2 and Tr-XL due to their poor performance in the results of [11,12]. In the following subsections, we present experimental insights examining how Memo's memory creation mechanism and training recipe enhance long-horizon reasoning and in-context learning. Each subsection highlights and focuses on a specific finding and analyzes a subset of baselines drawn from figures that are shared across multiple sections, allowing us to do a more structured comparison.
this section cite: ['b10', 'b11', 'b4', 'b6', 'b10', 'b11']

Section: Summarization Outperforms Full In-Context Access
We first show that Memo achieves higher performance and exhibits better ICL-ability compared to naive full context transformer (FCT) while using significantly less tokens in-context during evaluation.
In Figure 2b, we report the validation SR and SPL on the EXTOBJNAV task over 32k environment steps. All methods are trained using on-policy RL technique proposed by ReLIC. Memo is trained with a segment length of 256, 32 summarization tokens, and a trainable rollout length of 4096. We observe that Memo outperforms FCT achieving 7.5% higher SR and 2.5% higher SPL on average while using 8x fewer tokens (details in Table 1). We see that both Memo and FCT show in-context learning ability and continue to increase their performance up to 10k steps (∼2.5x of the training context length). Beyond this point, both methods experience performance degradation due to limitations in context length extrapolation. Notably, FCT degrades more than Memo in SR, while both show similar degradation in SPL. The single-episode variant (Transformer -no IEA) performs significantly worse, highlighting the disadvantage of not leveraging historical context in this navigation task. We also compare the GPU memory and FLOPs for Memo and FCT in Appendix A.5, and their sample-efficiency in Appendix A. 10.
In Figure 3a, we present a similar comparison on the Dark-Key-To-Door environment, reporting the average total return over trials as training progresses. All methods are trained using the off-policy RL algorithm AMAGO. We see that both Memo and RMT match the FCT baseline, achieving peak performance on validation episodes by 40M training steps. Interestingly, FCT shows a notable performance drop around the 35-40M step mark across all seeds, while Memo remains maintains stable convergence. This is likely due to training instability commonly observed in long-context RL [12,24], highlighting further training challenges with full-context models.
The consistent performance gains observed across both on-policy (with ReLIC) and off-policy (with AMAGO) RL setups demonstrates that our proposal to augment transformer policies with summarization enhances long-context modeling capabilities while being independent of the specific RL algorithm used.
this section cite: ['b9', 'b11', 'b23']

Section: Advantage of Summary Accumulation Over Fixed Recurrent Memory
The summary accumulation mechanism in Memo addresses the limitations of both transformers and recurrent models. While transformers can model full context, they suffer from significant memory overhead with long sequences. In contrast, recurrent models are memory-efficient but struggle to propagate gradients effectively over long horizons due to vanishing or exploding gradients. Summary accumulation mitigates context explosion by periodically summarizing experiences and propagating these summaries forward, allowing earlier memories to directly influence outcomes at later timesteps. Additionally, in purely recurrent models, gradients must pass through all intermediate memory update steps to reach the embedding of a relevant input far back in time. In contrast, periodic summary  generation and accumulation (as in Memo) ensure that each summary vector contributes directly to the loss function at later timesteps without degradation. For example, at a timestep t, all summaries created before are available as an undiminished input to the attention and subsequent optimization process.
We empirically validate this by comparing Memo's summary accumulation to a variant where only newly generated summaries (at the end of the current segment) are propagated to the next segment. This variant is closely aligned to the Recurrent Memory Transformer (RMT) [5]. To ensure fairness, we allow the recurrent model a larger summary size. Even with this adjustment, Memo achieves faster convergence, improving training speed by over 10M steps on the Dark-Key-To-Door task (Figure 3a). We further evaluate RMT on EXTOBJNAV (Figure 2b) using 1× and 2× Memo's summary size (32,64). Although RMT outperforms a purely recurrent baseline such as RL 2 -showing that per-step memory updates are too limiting-it still lags Memo by about 5% success rate. Larger summaries (RMT-128) destabilize training. This indicates that periodic summarization (like in Memo or Memo-RMT) improves expressivity and stability over purely recurrent per-step memory updates, while Memo's accumulation mechanism further enhances both by introducing residual-like gradient shortcuts that enable efficient and stable long-horizon optimization (see Appendix A.7 for further analysis).
this section cite: ['b4', 'b31']

Section: Importance of Long-Horizon Gradient Propagation: Comparison to Autocompressors
Since Memo's summarization approach is inspired by the Autocompressors (AC) method from the language modeling domain, we investigate how following the exact training setup from AC would perform in our setting.This involves pre-training a transformer policy on a shorter context length before fine-tuning it with summarization to extend its effective context length. We first train a baseline transformer with a context size of 1024 for 350M steps (half of the total training horizon for other baselines). This model serves as the weight initialization for our AC model, which is then fine-tuned for 350M steps with segment length l seg = 256, summary length l sum = 32, and a trainable rollout length of 4096. Following AC, we employ Truncated Backpropagation Through Time (TBTT) during training, restricting gradient propagation to summary tokens from only the two preceding segments; we refer to this variant as AC (TBTT). For a fairer comparison to other methods, we also train a version of AC where gradients are propagated across all segments within the rollout, referred to as AC (all segments). This setup is identical to Memo, except it's bootstrapped from a pre-trained model with a shorter context length.
We discover that this limited propagation is insufficient for tasks requiring long-horizon memory, as AC (TBTT) fail to capture dependencies spanning extended horizons. As illustrated in Figure 3b, AC (TBTT) exhibits poorer in-context improvement in object-finding tasks compared to AC (all segments). While AC (all segments) slightly surpasses Memo in SR and matches it in SPL during the first 6-8k steps, its performance degrades significantly over time, eventually converging to AC (TBTT) after 16k steps. This suggests that while a short context length pre-trained checkpoint can offer some initial advantage, it ultimately hinders effective generalization to longer contexts.
this section cite: []

Section: Extrapolation and Experience Subsampling During Evaluation
Avoiding memory explosion due to KV cache accumulation is a challenge for transformers at inference time. We evaluate Memo and the FCT baseline in a streaming setting, where only the most recent T KV cache elements are retained for inference, enforcing a fixed memory limit. To keep our setup simple, we do not update the KV cache to account for shifts in token position indices relative to the retained cache. We set T based on the context length at 6k env steps, as both Memo and FCT exhibit ICL generalization up to at least 6k steps (see Figure 2b). Thus, on EXTOBJNAV, we use T = 6k for Streaming Transformer and T = 1024(= round(6000/256) * 32 + 256) for Streaming Memo, since 6k steps correspond to 24 sets of summary tokens when l sum = 32 and l seg = 256.
We present the results in Figure 4a. Streaming Memo not only maintains performance through the trial but even outperforms Memo with full memory access in terms of ICL trend. In contrast, the Streaming Transformer undergoes a significant performance drop once streaming begins at 6k steps. Although methods like StreamingLLM [31] address the issues faced by the Streaming Transformer, Memo achieves strong performance without requiring such modifications. Additionally, Streaming Memo provides a way to maintain peak ICL performance by identifying the highest-performing context length during evaluation and initiating streaming beyond that.
this section cite: ['b30']

Section: Finetuning
In Figures 2b and 4a, we see a saturation in Memo's ICL performance at ∼60% SR. To investigate whether this limitation stems from suboptimal context length extrapolation, we take Memo's and FCT's checkpoints trained until 1B steps and fine-tune them on a 4× larger context size of 16,384 tokens for 500M steps. As shown in Figure 4b, the ICL performance of both models improves when evaluated over 32k environment steps. Notably, Memo (16k) outperforms Memo (4k) in SPL, indicating that training with longer contexts enables the policy to find more efficient paths. While the 16k full-context transformer (FCT) improves significantly over its 4k variant, it still achieves lower maximum SR and SPL compared to Memo (16k). Interestingly, Memo (16k) generalizes up to 1.5× its training context length after which it sees a slight degradation, whereas FCT (16k) maintains performance up to at least 2×. This necessitates further research into the factors limiting Memo's long-context generalization, which we leave for future work.
this section cite: []

Section: Ablations.
Summary Length Ablation: There is a trade-off between creating or retaining more memory and the compute saved when generating fewer tokens at training or inference time. Additionally, creating more tokens pushes the model further into the "extrapolation" regime, where its positional encodings stop generalizing effectively. We vary the summary length between values 16, 32, and 64 (thereby varying the compression ratio, l seg /l sum , between 16×, 8×, and 4×) on the EXTOBJNAV task and plot the resulting performance in Figure 5a. We find that Memo is highly sensitive to the choice of summary length. The main performance gap emerges after 6k environment steps, where a summary length of 32 achieves the best generalization across longer trajectories. In contrast, a summary length of 64 performs the worst, likely due to summary tokens overfitting to the training context length (see Figure 10a), leading to redundant information storage and a lower signal-to-noise ratio.
this section cite: []

Section: Segment Randomization:
In our main experiments, we adopted the strategy of randomizing segment lengths as in [7] for Memo, AC, and RMT to make token generation more robust to specific environment time steps. We randomized segment lengths between [0.8, 1.2] × 256, resulting in a range of [205,307]. In this ablation, we disable this randomization and evaluate the impact of using a fixed segment length of 256, as shown in Figure 5b. We observe that this variant performs significantly worse, not just during evaluation but in also training (see Figure 10b). We hypothesize that segment length randomization not only improves generalization by preventing overfitting to fixed boundaries but also serves as a form of curriculum. By exposing the model to segments of varying lengths, it naturally encounters both easier (shorter) and harder (longer) compression tasks, fostering a progressive learning effect. We present further results and ablations in Appendices A.6 and A.8.
this section cite: ['b6']

Section: Discussion
In this work, we proposed a simple yet effective approach for training transformers with memory in sequential decision-making tasks that require long-horizon reasoning. By introducing a context summarization mechanism, Memo enables transformers to retain and retrieve task-relevant information efficiently, significantly reducing computational overhead while maintaining long-term dependencies.
Our experiments demonstrated Memo's key advantages over full-context transformer baselines. In terms of performance vs. efficiency, Memo achieves comparable or superior performance while attending to sequences over 8× shorter than the full-context transformer. This improves scalability, enhances extrapolation to longer contexts, and ensures robust performance in streaming settings, where past context must be discarded due to inference constraints.
Furthermore, our analysis highlighted two key design choices: propagating gradients across multiple summarization steps and accumulating summaries rather than using a fixed memory size. We showed that allowing gradients to flow through sequential memory updates significantly enhances longhorizon reasoning, while memory accumulation improves retention of relevant past experiences, leading to more effective decision-making.
this section cite: []

Section: Limitations
Our experiments focus on object navigation in unseen scenes, where agents locate a fixed set of randomly placed objects. We do not evaluate semantic generalization-i.e., navigating to entirely new object categories-since this would conflate memorization with general object recognition capability. Future work could explore leveraging foundation models in a more open-ended setting.
Additionally, while Memo effectively summarizes experience, we do not explore more flexible memory mechanisms, such as memory consolidation, where past summaries are progressively compressed. Investigating these strategies could further improve long-term memory efficiency. We also do not explicitly study length extrapolation. Our results indicate that compressed memory tokens improve robustness to longer contexts and enable streaming inference, but extending generalization beyond training limits remains an open challenge. Lastly, our memory mechanism is trained endto-end via the RL objective. Future work could explore self-supervised objectives, such as future prediction, to enhance data efficiency in training memory representations.
this section cite: []

Section: A Appendix.
A.1 Task Setting and Model architecture: EXTOBJNAV Our policy takes as input an RGB image, the agent's relative position, the previous action, and the current goal object category. The RGB image is encoded using a ViT-B-sized VC-1 [20] visual encoder, finetuned by ReLIC to improve small object detection, a limitation of the original VC-1 model. The finetuned encoder remains frozen in all experiments, and we use VC-1's CLS token downstream, applying an MLP for dimension reduction to size 256. The goal category one-hot vector and previous action are also embedded into 256 dimensions, while the agent's position is converted into a 32-dimensional embedding. All these embeddings are concatenated together and passed to the sequence encoder.
All our sequence encoders (Memo, AC, etc) are based on a much smaller version of the causal auto-regressive transformer architecture used in LLaMA [29] (see details in Table 2). The weights of the model are trained from scratch unless specified otherwise. To accelerate data collection, the transformer's KV cache is maintained between rollout steps. Following ReLIC, we shuffle older episodes within each sequence and update the KV cache after each policy update, ensuring efficient memory reuse.
We use the same reward function for the navigation task as defined in Elawady et al. [11]. The reward consists of three components: (1) Geodesic distance reward: The agent receives a reward r d = -∆d where d is the geodesic distance to the closest object, which can change throughout the episode. ( 2 The SPL (Success weighted by Path Length) metric is computed as:
SPL = 1 N N i=1 S i • L * i max(Li,L * i )
where N is the number of episodes, S i is a binary indicator of success in episode i, L i is the length of the path taken by the agent, L * i is the length of the shortest path to the goal (computed using privileged simulator information). The SPL values reported in the main graphs are computed at intervals of 500 environment steps. Specifically, the SPL at step t is calculated by averaging the SPL of all episodes that completed between steps t -499 and t. So the SPL at step 5000 reflects the performance of all episodes that finished between step 4501 and 5000.
this section cite: ['b19', 'b28', 'b10']

Section: A.2 Task Setting and Model architecture: Dark-Key-To-Door
We refer the reader to the AMAGO implementation [12] (https://github.com/UT-Austin-RPL/amago) for details on the policy architecture used on the Dark-Key-To-Door task. We keep all implementation and training details fixed.
this section cite: ['b11']

Section: A.3 Training Scenes Dataset: EXTOBJNAV
Since the training dataset used in ReLIC only consisted of 333 episodes, we create our own training dataset while using the same validation dataset as them. We use the same 37 train scenes as [11,32]. However we generate 300 episodes per scene each having a different object placement. This leads to a total of 11100 training episode. In each episode we reduce the number of sampled objects to make the navigation task harder since the agent has to traverse longer, more targeted paths to reach objects instead of getting away by guessing (i.e. navigating to rooms with more clutter and many receptacles since the likelihood of finding an object would be higher there). We visualize the difference in sampled object distributions over episode, between the old [11] and new datasets in Figure 6.
this section cite: ['b10', 'b31', 'b10']

Section: A.4 Pseudocode for Memo
We present the algorithmic pseudocode for Memo in Algorithms 1 and 2. The modifications with respect to a distributed PPO implementation (corresponding to ReLIC) are highlighted in blue. We plan to fully open-source our implementation upon publication and include an early release version at https://github.com/Memory-icrl/memo.
Figure 6: Distribution of Objects in the training episodes: We visualize the changed distribution of objects sampled (and placed) randomly in the scenes corresponding to our dataset versus the original training dataset used in [11]. In our training scenes, avoid sampling more than 30 objects per scene to avoid cluttered rooms where multiple target objects could be placed together, to reduce the risk of models overfitting during training.
Algorithm 1 Memo 1: Initialize θ, ϕ, buffer B ← ∅, summary tokens S ← ∅ 2: o 1 ← reset() 3: i ← 0 4: for t = 1 to N do 5: C i:t-1 ← context from B 6: h t ← SeqEnc ϕ (S, C i:t-1 , o t ) 7: a ∼ π θ (a | h t ) 8: (r, o t+1 , done) ← step(a) 9:
if done then 10:
o t+1 ← reset() 11: end if 12: if t % l seg = 0 then ▷ l seg is segment length 13: S ← SeqEnc ϕ (S, C i:t-1 , o t )14
: i ← t 15: end if 16: Append (o t , a, r, o t+1 , done) to B 17: if t % num_steps = 0 then 18: TRAIN(B, t) 19: end if 20: o t ← o t+1 21: end for Algorithm 2 Train Memo 1: procedure TRAIN(B, t) 2: Initialize summary tokens S train ← ∅ 3: n ← t lseg ▷ n is num segments 4:
for j = 0 to n do  To further evaluate our method on long-horizon memory-intensive tasks, we include preliminary results on the Memory Maze benchmark. This environment poses a challenging multi-goal navigation task in a procedurally generated Mujoco-based maze. At the start of each episode, up to six colored goal objects are randomly placed in the maze, and the agent is given a randomly sampled sequence of goals to reach in order. Upon reaching the current goal, the next target is revealed, continuing until the episode ends. We evaluate Memo and the Full-Context Transformer (FCT) on the 9×9 version of the task, which includes two obstacles and has an episode length of 1000 steps. This setting demands both exploration and the ability to recall spatial information about previously encountered goals and maze structure. Figure 8 show that Memo matches the performance of FCT on this task. A.7 Extended Result: Benefits of Memory Accumulation on an Adversarially Long-Context RL Task (T-Maze) To explore memory-reuse, recall that we had included a recurrent-only summarisation (RMT) baseline in our experiments in Section 4.4. While RMT performance improved with larger memory sizes (e.g., RMT-64 outperformed RMT-32), Memo still achieved 5% higher success rate on navigation. Runs with a larger summary size (RMT-128) showed unstable convergence.
We observed an interesting trend across recurrent summarization baseline results on different benchmarks: while recurrent summarization can eventually achieve reasonable performance, they typically require significantly more training due to the need to propagate gradients through many sequential memory updates. In contrast, architectures like Memo-and transformers more generally-enable more efficient credit assignment by allowing earlier memories to receive gradient signals from later timesteps via direct attention mechanisms.To further validate the above intuition, we ran an experiment in the synthetic T-maze gridworld environment [22]. Here, the agent sees a clue ("left" or "right") at timestep 0, which disappears at timestep 1. It then traverses a long corridor with only forward actions, and at the end must choose the left or right room based on the initial clue to receive a reward. The corridor length can be made arbitrarily large (e.g., 10,000 steps). We hypothesized that purely recurrent models would especially struggle in this adversarial setting, as learning to retain the clue requires backpropagating gradients through all the segments corresponding to the 10,000 steps. Memo and FCT, however, can access that information directly at later timesteps-either through full attention (FCT) or through a small number of memory consolidation stages (Memo). Empirically, we observed the expected result: RMT required 10× longer to achieve an average reward of 1.0 for the first time during training, compared to FCT and Memo. In addition, it exhibited much greater instability in learning the correct policy, since it did not manage to maintain this performance (average reward = 1.0) for three consecutive checkpoints at any point during training.
this section cite: ['b10', 'b21']

Section: A.8 Extended Ablation: Direct KV Cache Construction from Summary Embeddings
To assess whether the improved performance of Memo stems from the additional computation it uses (due to summary embeddings being processed through the transformer twice), we perform an ablation that removes the re-encoding step.
In this variant, instead of generating summary tokens and re-encoding them through the transformer, we directly construct the memory (KV cache) from the internal embeddings produced at each layer while encoding the learnable summary embeddings. As in Memo, these summary embeddings are inserted at regular intervals between segments of tokens. However, unlike Memo, their layer-wise representations are extracted directly to serve as the KV cache, without further processing. This approach is more compute-efficient, as each token-including summaries-is passed through the transformer only once. However, as shown in Figure 9, this variant performs poorly and quickly degrades beyond the training context length. We hypothesize that this is not only due to the lack of deeper summarization capacity, but also due to positional embedding generalization issues. Since summary embeddings are appended at the end of each segment, their positional indices lie in the range [l seg , l seg + l sum ] , with offsets accumulating across segments. Consequently, these summary positions -and those of all future tokens -progressively reach larger values, potentially leading to distribution shifts in the positional encodings that the model is unable to generalize over.
This result suggests that Memo's re-encoding step not only adds depth for more expressive summarization but also resets the positional index range for the summary tokens, both of which appear beneficial for stability and generalization.
A.9 Training-Validation Gap  In this section, we visualize the overlaid training and validation performance curves for Memo on the EXTOBJNAV task, which presents two key challenges. First, validation scenes are entirely novel, requiring models to generalize their in-context learning (ICL) solutions. Second, models are evaluated on 4× larger trial sizes, testing their context extrapolation capabilities. To assess generalization, we compare overfitting levels across models, distinguishing whether low validation performance stems from limited encoding expressivity (indicated by low training performance) or poor generalization despite strong training performance. We present train versus validation curves for Memo while ablating the summary lengths in Figure 10a and while ablating the segment length randomization in Figure 10b.
this section cite: []

Section: A.10 Sample Efficiency of Memo
We observed higher sample-efficiency for Memo during training on the EXTOBJNAV and Dark-Key-To-Door tasks compared to the full context transformer. For Dark-Key-To-Door, this is directly observable in Figure 3a since the performance is plotted against training progress. We present the this observation for the EXTOBJNAV task in this section by visualizing the performance of earlier and later checkpoints (700M and 1B training steps) of both FCT and Memo in Appendix A.9. We see that even earlier into training, Memo exhibits higher-ICL success rates compared to FCT which starts to get closer to Memo's SPL only after 300M more steps of training.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We only list the findings from experimental results in our claims.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Provided in Section 6 Guidelines:
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: We do not include theorems and proofs in this work.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: Provided in the Appendix. We also release our code.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification:Provided in Appendix A.4.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
this section cite: []

Section: 
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: Provided in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: All results are averaged over seeds and we indicate error intervals.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: All information required to reproduce the setup can be found in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We have read through the ethics guidelines and conformed to them.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: Our work is related to foundational research on transformer architectures and does not immediately carry broader societal impacts.
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
Answer: [NA] Justification: We do not release any models or datasets.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All works that we build on are cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: The released code is documented.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No crowd-sourcing is conducted. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No research on human subjects is conducted. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: Only used for editing. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: On evaluation of embodied agents navigating to objects Year: (2020)
Ref_id:b1 Title: A survey of meta-reinforcement learning Year: (2023)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: Scaling transformer to 1m tokens and beyond with rmt Year: (2023)
Ref_id:b4 Title: Scaling transformer to 1m tokens and beyond with rmt Year: (2024)
Ref_id:b5 Title: The ycb object and model set: Towards common benchmarks for manipulation research Year: (2015)
Ref_id:b6 Title: Adapting language models to compress contexts Year: (2023-12)
Ref_id:b7 Title: Learning phrase representations using rnn encoder-decoder for statistical machine translation Year: (2014)
Ref_id:b8 Title: Transformer-XL: Attentive language models beyond a fixed-length context Year: (2019-07)
Ref_id:b9 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b10 Title: A Recipe for 64k Steps of In-Context Reinforcement Learning for Embodied AI Year: (2024-10)
Ref_id:b11 Title: AMAGO: Scalable in-context reinforcement learning for adaptive agents Year: (2024)
Ref_id:b12 Title: Hierarchical memory transformer for long context language processing Year: (2024)
Ref_id:b13 Title: Long short-term memory Year: (1997-11)
Ref_id:b14 Title: Deep networks with stochastic depth Year: (2016)
Ref_id:b15 Title: Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation Year: (2024)
Ref_id:b16 Title: A method for stochastic optimization Year: (2014)
Ref_id:b17 Title: -context reinforcement learning with algorithm distillation Year: (2022)
Ref_id:b18 Title: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b19 Title: Where are we in the search for an artificial visual cortex for embodied intelligence? Year: (2023)
Ref_id:b20 Title: Leave no context behind: Efficient infinite context transformers with infini-attention Year: (2024)
Ref_id:b21 Title: When do transformers shine in rl? decoupling memory from credit assignment Year: (2023)
Ref_id:b22 Title: A fast, extensible toolkit for sequence modeling Year: (2019)
Ref_id:b23 Title: Stabilizing transformers for reinforcement learning Year: (2020)
Ref_id:b24 Title: Compressive transformers for long-range sequence modelling Year: (2020)
Ref_id:b25 Title: Glu variants improve transformer Year: (2020)
Ref_id:b26 Title: Efficient attention: Attention with linear complexities Year: (2021)
Ref_id:b27 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b28 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b29 Title: Ddppo: Learning near-perfect pointgoal navigators from 2.5 billion frames Year: (2020)
Ref_id:b30 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b31 Title: Open-vocabulary mobile manipulation Year: (2023)
Ref_id:b32 Title: Long Context Compression with Activation Beacon. arXiv e-prints, art Year: (2024-01)
