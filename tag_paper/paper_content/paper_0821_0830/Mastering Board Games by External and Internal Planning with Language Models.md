Title: Mastering Board Games by External and Internal Planning with Language Models
Abstract: Advancing planning and reasoning capabilities of Large Language Models (LLMs) is one of the key prerequisites towards unlocking their potential for performing reliably in complex and impactful domains. In this paper, we aim to demonstrate this across board games (Chess, Fischer Random / Chess960, Connect Four, and Hex), and we show that search-based planning can yield significant improvements in LLM game-playing strength. We introduce, compare and contrast two major approaches: In external search, the model guides Monte Carlo Tree Search (MCTS) rollouts and evaluations without calls to an external game engine, and in internal search, the model is trained to generate in-context a linearized tree of search and a resulting final choice. Both build on a language model pre-trained on relevant domain knowledge, reliably capturing the transition and value functions in the respective environments, with minimal hallucinations. We evaluate our LLM search implementations against game-specific state-of-the-art engines, showcasing substantial improvements in strength over the base model, and reaching Grandmaster-level performance in chess while operating closer to the human search budget. Our proposed approach, combining search with domain knowledge, is not specific to board games, hinting at more general future applications.

Section: Introduction
While large language models (LLMs) perform fluently on text generation, language understanding and translation, they are prone to hallucinations and reasoning errors, especially in complex contexts (Chang et al., 2024;Hadi et al., 2023). Hence, special attention has been given to the development of planning and reasoning capabilities in LLMs (Minaee et al., 2024). In terms of Kahneman's cognitive theory of two systems (Kahneman, 2011), prior work primarily improved associative System 1 inference in language models, whereas planning and reasoning now focuses on improving the more deliberate System 2 thinking (Plaat et al., 2024).
Planning and reasoning approaches typically fall into one of two distinct categories: In internal planning the LLM develops a plan in context, like Chain-of-Thought prompting (Wei et al., 2022), by autoregressively considering possible steps towards the goal and their consequences. By contrast, external planning uses the LLM to generate steps in a neurosymbolic system, such as in Tree of Thought (Yao et al., 2024), where an outer loop performs explicit search over possible sequences of steps. This paper presents how language models can be trained for internal and external planning to improve reasoning in sequential decisionmaking, using board games as an experimental domain.
Board games have historically played an important role in the development of automated decision-making, with Torres' automaton El Ajedrecista playing three-piece chess endgames before the advent of digital computation (tor, November 1915). Games provide diverse reasoning challenges about both environment dynamics and opponent strategies, and thus have gained significant attention in pushing the boundaries of LLM reasoning capabilities (Hu et al., 2024b;Costarelli et al., 2024;Duan et al., 2024). LLMs have incidentally been struggling with reliably playing common board games, like chess or even tic-tac-toe (Topsakal & Harper, 2024). This may seem somewhat surprising, considering how much headway has been made in other areas. However, as games astutely expose the inability of LLMs to consistently reason over possible futures with world models, they make a great testbed for planning and reasoning.
Making progress in game-playing could inform how to best instill this ability in LLMs going forward. We present several contributions towards this aspirational goal.
Contribution 1: MAV model. We pre-train a Transformer model, the multi action-value model (MAV), capable of playing several board games (Chess, Chess960, Connect Four, Hex) at a strong level. This model is capable of reliably tracking the board state throughout games, and makes (good) legal moves.
this section cite: ['b11', 'b43', 'b79', 'b57', 'b88', 'b126', 'b136', 'b19', 'b24', 'b113']

Section: Contribution 2: External search.
We use MAV within an external MCTS controller, as a value/implicit policy and transition function. Our Async MCTS reaches Grandmaster level with the number of moves considered per decision (∼100 to ∼1k) comparable to human players (for reference, Al-phaZero (Silver et al., 2017) used ∼10k simulations and traditional engines use up to millions).
this section cite: ['b107']

Section: Contribution 3: Internal search.
We distill the search procedure directly into the LLM, generalizing the Stream of Search (SoS) (Gandhi et al., 2024b) approach to a significantly more complex domain. The performance of the resulting agent scales smoothly with the given search budget.
this section cite: []

Section: Multi-Action-Value Model
The multi-action-value (MAV) model is a Transformer model pre-trained exclusively on textual game data that functions simultaneously as a: (i) world model, (ii) value function, and (iii) policy function for multiple perfectinformation board games. Acting as world model further requires the following capabilities: (i) state-tracking (determining the new state of the game after a move has been played in the previous state), (ii) legal move prediction, (iii) terminal state detection. To achieve this, MAV is trained on examples following a flexible format illustrated in Figure 1.
Command specification. MAV input starts with a header containing a series of commands specifying the game being played, the inputs provided and the expected outputs.
this section cite: []

Section: State representation.
Each game uses a different textual format to represent its state, used with the %state and %prev state commands. All of them are designed to be easy for the model to navigate and manipulate: each field on the board is a separate token, and fields maintain constant relative positions in token space. For chess we additionally support using the standard FEN representation (with the %fen command).
Starting state. The starting state for the model can be given in two ways: either directly with the %state command, or as in Figure 1 with the commands %prev state %prev action %state, which instruct the model to take a previous state (%prev state) and a move played in that state (%prev action) and use these to infer and output the starting state (%state). The latter functionality makes the model act as a transition function.
Value function. The %top k command instructs the model to consider the starting state and output the best k legal moves and their action values, in order of preference from highest action value to lowest. k is configurable to allow for varying the amount of inference-time computation MAV performs. If there are fewer legal moves than k, the model simply outputs all of them (this can also be achieved by setting k to "all"). In the case where the starting state is terminal, the %top k command outputs the game outcome instead (e.g., [%top 1 invalid : "1-0"] when the first player has won).
The action value for a move corresponds to the predicted win probability for the player in the current state if this move is taken. For chess, state-action values are represented similarly to Ruoss et al. (2024a): Stockfish centipawn evaluations are mapped to win probabilities using the formula Win % = 50 + 50 2 1 + e -0.00368208•centipawns -1 obtained from https://lichess.org/page/  accuracy. Win probabilities are then mapped to discrete non-overlapping buckets, making the value prediction problem a classification task rather than a regression task. This has proven to be beneficial in other works (Farebrother et al., 2024). We use 64 buckets, each represented by a different special token (e.g., <ctrl28> for bucket 28). Figure 2 shows two win probability distributions predicted by MAV. Connect Four and Hex use game engines Fhourstones (Tromp) and neurobenzene (Gao & Pawlewicz), respectively, to obtain state-action values, which are similarly mapped to the same 64 bucket tokens.
When producing training data we vary k randomly, making sure to include examples with k greater than the number of legal moves. We randomize the order of the moves to: (i) encourage the model to treat the moves independently, and (ii) help to prevent hallucinations. With a fixed order, e.g., lexicographical from a1a2 to h8h7, then (i) if the model skipped a legal move a1a4 and emitted a1a5, it couldn't go back and fix its mistake, and (ii) the model might learn to heavily lean on evaluations of previous moves when considering a move like h8h7, which is late in the order. Using randomised ordering of moves in the training data, the model is steered towards an approximate permutation symmetry (Bronstein et al., 2021).  [%top_5 d7b6:<ctrl28> f6f5:<ctrl33> d7c5:<ctrl28> f8h8:<ctrl29> a5f5:<ctrl29>] [%best_action f6f5] [%FEN r1b1nr2/pp1np1bk/2ppP1p1/q4p2/3P1P2/2NQB3/PPP1B1PP/R4RK1 w --0 14] Game name (one of) chess chess960 hex connect_four Input spec (one of) current state, or previous state + action Output spec (all of) current state (if input was previous state), either top-k or all moves+values, best move and state afterwards inputs in order from spec outputs in order from spec prompt with input and output spec Figure 2.
this section cite: ['b28', 'b8']

Section: Value Definition: Scoring Methods. During inference,
For a safe positional move, such as Re1, MAV assigns little uncertainty and gives a win probability of 40-45% since white is a pawn down. In contrast, the bishop sacrifice BXh7+ (best move according to Stockfish) is a riskier attacking move and thus has substantial probability mass around low, equal, and high win probabilities.
for each move the model produces a distribution over the 64 tokens that correspond to action value buckets. We experiment with two methods to turn this distribution into a final score for the move: max scoring, which corresponds to greedy decoding, uses the mode of the model's distribution, while mean scoring uses the expectation over buckets.
Mean scoring (also used in Ruoss et al. (2024a)) utilizes more of the model's information, and is better able to distinguish actions with equal modes. In an almost tied position, the most likely outcome for many moves may be a draw, but still some moves may have a higher winning chance than others. Mean scoring will be able to pick the best move in this case, while max scoring can not. This is further illustrated in Figure 2.
this section cite: []

Section: Best action prediction.
In positions where the model has a large advantage, it may have many moves available with ca. 100% win probability that don't directly lead to finishing the game by checkmate. This may cause the model to play aimlessly in such positions. This problem was encountered in Ruoss et al. (2024a), which they resolved by using Stockfish to break ties when all top 5 moves lie above a win probability of 99%. To eliminate reliance on a game engine at inference time, we introduce the %best action command, which instructs the model to emit the game engine's chosen action in the position. In this way we teach the model the tie-breaking procedure described above.
Novelty. To summarize, MAV makes four key improvements over the existing state-of-the-art Transformer-based chess engines (Ruoss et al., 2024a;Monroe & Leela Chess Zero Team, 2024;Czech et al., 2024;Farebrother et al., 2024) First, it performs world modeling, policy and actionvalue computation together in one model. Second, it is trained to output the best action at the end of action-value modeling, enabling the model to reliably finish games where it has a decisive advantage. These two improvements enable MAV to play complete games without relying on an external game engine for legal moves or for finishing the game. Third, all of the above steps can be done in a single model call without having to evaluate every action separately-an important feature that guided the design of the MAV format to reduce the cost and infrastructural complexity of inference. Fourth, the amount of compute performed by the model at inference time can be varied dynamically, enabling us to achieve higher quality at the cost of higher latency when performing both internal and external search, where the performance increases as we scale the planning.
this section cite: ['b82', 'b21', 'b28']

Section: Datasets.
We curate a dataset of diverse, relevant positions in four games: Chess, Chess960, Connect Four, and Hex. The statistics and sources for each of these datasets are shown in Table 4 in Appendix B. Each position is used to produce a single training example, randomly varying (i) the k action values in %top k, (ii) the presence of the initial or final state tracking commands, (iii) the use and order of %state or %FEN representations in chess. Further details on the datasets are provided in Appendix B.
this section cite: []

Section: Models.
We train two randomly initialized decoder-only Transformer models using the Gemini architecture (Gemini Team et al., 2024), called MAV and MAV-small,foot_0 with 2.7 billion and 1 billion parameters respectively 2 . They were trained on 1.9 and 1.2 epochs of the dataset. Except for the max scoring result for MAV-small in Table 1, all other experiments and results use the 2.7 billion MAV. In order to efficiently use our models' parameters, the input part of each training example is masked out during loss computation. This means our models do not waste capacity on learning to generate game positions.
this section cite: ['b38']

Section: External Search
External search employs MAV to generate planning steps, and applies a search algorithm to direct and optimise over sequences of planning steps. In this paper, we evaluate planning on top of the previously discussed MAV, using MCTS as the search procedure.
External search is based on an AlphaZero-style MCTS (Silver et al., 2017). There are two key ingredients: a prior function which returns a probability distribution over actions at each state, and a value function returning a numerical value indicating value of a state or state-action pair, both extracted from the MAV output. External search is an adaptation of MCTS; in its most basic form, as in AlphaZero, MCTS relies on an explicit world model, querying a game engine (e.g., a chess implementation in OpenSpiel (Lanctot et al., 2019)) to provide legal actions, state transitions and terminal states. Inspired by MuZero (Schrittwieser et al., 2020), in a subsection below we describe how we remove this dependency on the game engine and instead use only MAV to track states, transitions, and provide legal actions and game outcome during planning.
The external search algorithm guided by a learned world model is summarized in Algorithm 1, with subroutines contained in Appendix A. A search is started at an initial game state s 0 . First, the legal actions a L 0 and associated state-action values Q (i) (s 0 , a L 0 ) from active player's i perspective are obtained from MAV. The prior function is an ε-greedy policy, derived entirely from state-action values comprised of a greedy softmax over the top k values, mixed with a uniform distribution over all actions to encourage exploration. Specifically, let a g,k ⊆ a L be the subset of legal actions a L whose values are among the top k-ranked values, or equal to a L if |a L | ≤ k. Define the probability Algorithm 1 EXTERNAL-MCTS(s 0 ) 1: Input: Initial state s 0 , active player i, top k values, num. simulations M 2: Output: Recommended action a * 3: a L 0 , Q (i) (s 0 , a L 0 ) = MAV(s 0 ) 4: Compute prior P (s 0 , a L 0 ; k) according to Eq. 1. 5: Initialize root node: N 0 (i) ← i N 0 (s) ← s 0 N 0 (a L ) ← a L 0 6: EXPAND(N 0 , P ) 7: for m = 1, . . . , M {this can be async} do 8: SIMULATION(N 0 , MAV, k) 9: end for 10: a * ← FINALMOVESELECTION(N 0 ) 11: return a * of action a in state s under the greedy policy to be
π g,k (s, a; Q (i) ) = exp 1 τ Q (i) (s,a) a ′ ∈a g,k exp 1 τ Q (i) (s,a ′ ) if a ∈ a g,k ; 0 otherwise,
where τ is a temperature parameter. In practice, we dynamically adapt the temperature (Veličković et al., 2024) as a function of the number of moves played and transform stateaction values to win probabilities before computing softmax.
The probability of the uniform policy is π u (s, a;
a L ) = 1 |a L | . The prior probability of taking action a in state s is P (s, a; Q (i) , a L , k) = (1 -ε)π g,k (s, a; Q (i) ) + επ u (s, a; a L ).(1)
The root node N 0 is then initialized with the active player, string description of a state corresponding to the node, and legal actions (respectively: N 0 (i), N 0 (s), N 0 (a L )). The node is then expanded so that a child node is added for each action and a prior probability attached to the action leading to the child. Then simulations are run: each simulation starts at the root node, actions and children nodes are selected according to PUCT until a leaf node is expanding and evaluated, and values are backpropagated through the nodes that the simulation visited. After M such simulations, a final move a * is selected.
this section cite: ['b107', 'b63', 'b101', 'b118']

Section: External Search with State-Tracking MAV
Classical MCTS relies on several components of a game engine to simulate the game. MAV replaces these components by combining state tracking, legal action prediction, action value prediction, and terminal state detection in a single model. This allows us to remove the dependency on a game engine, which results in a few benefits. First, we use the board games to showcase that MCTS achieves astonishing performance even without access to an explicit world model, which is the case for many interesting real-world problems. Second, it is a step closer towards the internal search with LLMs.
We adapt MCTS (see Algorithm 1) to use the learned MAV to predict the state transition from the string description of a parent node N t-1 (s) and action a t-1 to the child node N t , as shown in Figure 3. We use the obtained information to store a (predicted) string description s t of the child node, which is then expanded with legal actions a L t and their associated state-action values Q (i) (s t , a L t ) used to compute a prior according to Equation (1) and the child's value as the maximum over all state-action values, i.e., max at∈a L t Q (i) (s t , a t ). MAV may hallucinate and return responses that are improperly-formatted. To address this, for each game, a parser function is responsible for translating the output into the next state, legal actions, and their values. If the parsing fails -which can only happen if a response violates a predefined format -a special value of -∞ is assigned to the node to avoid future consideration. This procedure makes state-tracking MCTS robust to hallucinations, by explicitly avoiding states where the hallucination occurred.
this section cite: []

Section: Async MCTS and Dynamic Virtual Counts
Due to the heavy cost of inference in LLMs, we use an Asynchronous Monte Carlo tree search (Async MCTS) that queues multiple leaf evaluations simultaneously (i.e., in a "batch" of simulations). To avoid redundant concurrent computation, we employ a commonly-used optimization in parallel MCTS: virtual counts, which temporarily increase the visit counts by some constant n c during the simulation, which are then removed during the backpropagation phase. For more details, please see Appendix A.
With low simulation counts, using the virtual losses accompanied by the fixed virtual counts n c did not strike a satisfactory balance between exploration and exploitation. To address this, we introduce the dynamic virtual counts that dynamically assign more weight to the virtual count values closer to the leaf nodes. Suppose simulation m encounters leaf node N l . We define a virtual count for states-action pairs (s t , a t ) visited in simulation m and leaf
N l n c (m, N l ) = min n min , n max • 2 d N t -d N l ,
where d Nt represents the depth of a child node N t relative to the leaf node N l reached during simulation m and ⌊•⌋ is a floor function. As depicted in Figure 7 in Appendix A, it proved beneficial to exponentially decrease the virtual counts starting with the maximum virtual count n max at the leaf node which is then halved at each parent up to the root node while maintaining a minimum virtual count n min .
this section cite: []

Section: Internal Search
In contrast to external search, in internal search the model does not require an external controller. Instead, the search procedure is distilled into the model so that it is capable of (i) evaluating search nodes (states), (ii) expanding nodes while updating the current state, and (iii) backpropagating the results from leaf nodes back to the root node-all within a single model call. The distillation is done by linearizing search trees into a text format and training the model on those linearized trees.
Data. The prompt for internal search resembles that of MAV, but includes a preamble with search parameters (tree depth and breadth, see Figure 4). The format of the target data is inspired by depth-first order traversal of minimax trees, as an iterative and linearized sequence of minimax tree traversal. Hence, following this format corresponds to an algorithmic execution task, akin to CLRS-Text (Markeeva et al., 2024).
The training data is based on target search trees, which were constructed using depth 3 (N.B., depth-zero is MAV), by annotating states (e.g., chess states being annotated with Stockfish), and expanding the top 5 moves into trees. This results in high quality target search trees, similar to those internally generated by game engines. To diversify training data and enable search budget control for trained models, prompts were composed with diverse search parameters, ranging depth 1-3 and breadth 2-5, and continuations yield corresponding trees that were subsampled to match the parameters while fitting into context size. This necessitated
Minimax breadth=3 depth=1 Evaluation for node: <root> <mav game=chess> %state %top all %best action </mav> [%state 2r3k1/p3p3/4N1Pp/3Q4/8/5Pq1/PPr3P1/1K5R b] [%top all . . . g3g6:<ctrl33> . . . c2e2:<ctrl32> . . . g3d6:<ctrl35> . . . ] [%best action g3d6] Expand from node <root>: <root g3d6> <root g3g6> <root c2e2> Evaluation for node: <root g3d6> <mav game=chess> %prev state %prev action %state %top all %best action </mav> [%prev state 2r3k1/p3p3/4N1Pp/3Q4/8/5Pq1/PPr3P1/1K5R b] [%prev action g3d6] [%state 2r3k1/p3p3/3qN1Pp/3Q4/8/5P2/PPr3P1/1K5R w] [%top all d5d4:<ctrl18> . . . d5b3:<ctrl23> d5d6:<ctrl32> a2a3:<ctrl11> . . . ] [%best action d5d6] Decision for node: <root g3d6> Selecting: d5d6 : "<ctrl32>" Evaluation for node: <root g3g6> <mav game=chess> %prev state %prev action %state %top all %best action </mav> [%prev state 2r3k1/p3p3/4N1Pp/3Q4/8/5Pq1/PPr3P1/1K5R b] [%prev action g3g6] [%state 2r3k1/p3p3/4N1qp/3Q4/8/5P2/PPr3P1/1K5R w] [%top all . . . d5c5:<ctrl1> e6f4:<ctrl22> b2b3:<ctrl1> . . . ] [%best action e6f4] Decision for node: <root g3g6> Selecting: e6f4 : "<ctrl22>" Evaluation for node: <root c2e2> <mav game=chess> %prev state %prev action %state %top all %best action </mav> [%prev state 2r3k1/p3p3/4N1Pp/3Q4/8/5Pq1/PPr3P1/1K5R b] [%prev action c2e2] [%state 2r3k1/p3p3/4N1Pp/3Q4/8/5Pq1/PP2r1P1/1K5R w] [%top all . . . h1h6:<ctrl1> e6f4:<ctrl32> h1c1:<ctrl1> . . . ] [%best action e6f4] Decision for node: <root c2e2> Selecting: e6f4 : "<ctrl32>" Decision for node: <root> g3d6 : "<ctrl33>" g3g5 : "<ctrl43>" c2e2: "<ctrl33>" Selecting g3g6 : "<ctrl43>" Playing g3g6! excluding the biggest parameter combination (i.e., omitting depth 3, breadth 5 examples).
An example internal search trace predicted by the trained internal search MAV (MAV-IS), along with the corresponding search tree, is shown in Figure 4.
this section cite: ['b74']

Section: Training details.
We leveraged the pre-trained MAV and fine-tuned it using a mixture of 60% MAV data and 40% search data. Fine-tuning was run for 20,000 steps, using a batch size of 512. Thus the model saw an order of magnitude more tokens during the MAV pre-training compared to the fine-tuning on internal search traces.
this section cite: []

Section: Experiments

this section cite: []

Section: Evaluation
Evaluating LLMs in general and reasoning specifically is a broad field (Chang et al., 2024), within which games have been established as an evaluation benchmark (Costarelli et al., 2024;Duan et al., 2024). We evaluate MAV language models in a games league which head-to-head match-ups, sampling combinations uniformly at random from a pool of MAV and baseline agents. We report internal Elo ratings (relative Elo only between members of the population) as well as external Elo ratings where possible.
For chess evaluation, we rely on the state-of-the-art engine Stockfish at different playing strengths to estimate the widely used external Elo rating. This is done by calibrating the internal Elo with externally-reported ratings using a linear fit. As an additional, we include the Ext-BoN model (Ruoss et al., 2024a), which showed Grandmaster-level performance on chess with Transformers (see Appendix B for details). We use a set of Top Chess Engine Championship (TCEC) opening positions (tce) (see Table 5 in Appendix G). In each match-up between two agents, a specific opening is used, and agents swap seats to ensure each agent plays each opening both as black and as white. Every instance of Stockfish is run with 2 seconds of search time. We delineate further our evaluation setting together with the closest related works in Appendix B and Appendix C.
An overall comparison of the playing strength of different methods is shown in Table 1. Next, we perform a deeper dive into the performance of the different approaches.
this section cite: ['b11', 'b19', 'b24']

Section: Multi-Action-Value Results
In terms of chess playing strength, MAV reaches an external Elo of 2923 when using mean scoring and 2875 when using In Table 2, we analyze the legal move rate, the precision and recall of the predicted %top all moves, and the accuracy of predicting the next FEN state. The results demonstrate that MAV is able to reliably perform all of these actions.
Generalization. During the opening and endgame, human players often rely on memorized opening lines and endgame theory, while middlegame requires more calculation and intuition. In Appendix D, we report the same pattern in the games of MAV and a concrete position where the model displays a creative play in unseen position. Overall, 10% of the positions played by MAV in evaluation appear in its training data, while between moves 20 and 50, virtually no position has been seen by MAV during training. These results show that in order to avoid losing games during middlegame, MAV is required to generalize.
this section cite: []

Section: External Search Results
For our external-MCTS agents, we tune hyper-parameters using a combination of manual and head-to-head comparisons and report the details in Appendix A. We also include a basic MCTS baseline, which refers to MCTS with a uniform prior and random rollouts run with 100 simulations.
We first analyze the performance of the scoring method for the value function when used within MAV-MCTS. We do this by running a large tournament between various models with the details, including results shown in Figure 9 in Appendix A. MAV-MCTS with mean scoring performs noticeably better in all cases. Hence, for the remainder of the external search experiments, we report results only for the mean scoring method.
We then run an even bigger tournament including Ext-BoN with the results shown in Table 1. The total number of games played in each respective tournament were 14689 for Chess, 4480 for Chess960, 2189 for Connect Four, and 6334 for Hex. In chess, MAV-MCTS with just 100 simulations achieves an internal Elo 68 higher than the searchless MAV and generally Elo performance improves logarithmically as a function of the number of simulations as illustrated in Figure 5 (right).
In the case of Chess960, where every game begins with a random position (among a preset 960 initial positions), we obtain comparable results to Chess. In the case of Connect Four, we notice that MAV-MCTS is particularly helpful in improving upon MAV, with all agents improving by at least 244 Elo. Similarly to Chess, the improvements consistently rise with added simulations, but here we notice a relatively smaller performance gain of MAV over a basic MCTS, and larger improvements of MAV-MCTS over MAV.
Since our final model was not adequately trained with statetracking capabilities for Hex, in the Hex results only a game engine is used to determine legal actions, state transitions, and terminal states. In contrast to Connect Four, we notice a large improvement between basic MCTS and MAV in Hex, and improvements similar to chess from search. We suspect that this is due to the relative complexity of the games, but more research is needed to clarify the reason for these differences. It should be possible to support the engine-free logic in Hex similarly to Chess going forward. Across all games, external MCTS performs consistently better as the number of simulations is increased.
this section cite: []

Section: Internal Search Results
For our internal search experiments, we vary the depth and breadth parameters of the minimax search, up to breadth 4 and depth 2. We map these different configurations into token counts by computing the average length of the prompt + response per configuration in our training data. We also analyzed if the MAV-IS search produces the tree of the requested shape whenever possible, considering the available number of legal moves at each node. MAV-IS constructed an accurately shaped tree 99.6% of the time on a sample 5980 positions taken from existing games.
For an example of how internal search can improve playing strength, see Figure 4. In this example, the initial, MAV section, of the response predicts q d6 as the best move.
However, as the internal search continues, the model is able to find a better move q Xg6 after exploring the top-foot_2 lines one step further. This points to the model's ability to selfcorrect, which is an important capability of LLMs that can reason (Kumar et al., 2024).
this section cite: []

Section: Discussions and Limitations
Despite promising results in the domain of perfect information board games indicating the potential of external and internal planning with LLMs, our initial study makes a number of assumptions that future work may need to address -namely, the ability to acquire or generate large quantities of game play data, as well as the availability of reliable solvers or game engines that can be used to annotate this data in order to create an appropriate training curriculum for the model.
Another important limitation of our MAV models is that they have been trained exclusively on game data, and therefore do not possess the ability to communicate verbally using natural language. However, there should be no fundamental obstacles in achieving the same ability in potentially larger models that may also incorporate natural language data, as we train on the exact same architecture and tokenizer used in classical text-based LLMs.
It remains an open question how to design good value functions for general conversational task, and how to incorporate these value functions or other highly specialized knowledge in training such that the model can draw upon them flexibly at inference time, in a wide variety of conversational contexts.
this section cite: []

Section: Conclusions
This paper demonstrates the capacity of LLMs to learn strong value functions and act as a world model across multiple perfect information games. This enables their use in MCTS, where we observe significant performance gains of approximately +300 Elo points even with a fairly limited search budget. Going further, we find that training on search traces enables the model to learn an effective search procedure that can be executed via a single model call. This adds to the rapidly growing body of literature highlighting the promise of planning and reasoning with LLMs.
Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., and Yang, D. Can large language models transform computational social science? Computational Linguistics, 50(1): 237-291, 2024. +4 +4 +2 +4 +4 +8 +8 +8 Figure 7. Dynamic virtual counts for nmax = 8 and nmin = 2. Virtual counts from separate simulations are added, as is the case in the +4 node with two children (+4 = (+2) + (+2)). 100 250 500 1000 2000 Number of simulations 1350 1400 1450 1500 1550 1600 1650 1700 Internal Elo MAV-MCTS (mean scoring) MAV-MCTS (max scoring) Figure 9. Performance of scoring methods in MCTS among a tournaments between all MCTS agents, various levels of Stockfish, and basic MCTS. The y-axis shows the internal Elo of each agent with basic MCTS set to 0 internal Elo.
this section cite: []

Section: References
Ref_id:b0 Title: Torres and his remarkable automatic devices -he would substitute machinery for the human mind. Scientific American Supplement Year: (1915)
Ref_id:b1 Title: Debunking the chessboard: Confronting gpts against chess engines to estimate elo ratings and assess legal move abilities Year: (2023)
Ref_id:b2 Title: Many-shot in-context learning Year: (2024)
Ref_id:b3 Title: Rest meets react: Selfimprovement for multi-step reasoning llm agent Year: (2023)
Ref_id:b4 Title: Implicit search via discrete diffusion: A study on chess Year: (2024)
Ref_id:b5 Title: A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity Year: (2023)
Ref_id:b6 Title: Graph of thoughts: Solving elaborate problems with large language models Year: (2024)
Ref_id:b7 Title: Reliable reasoning beyond natural language Year: (2024)
Ref_id:b8 Title: Geometric deep learning: Grids, groups, graphs, geodesics, and gauges Year: (2021)
Ref_id:b9 Title: Language models are few-shot learners Year: (2020)
Ref_id:b10 Title: Weak-to-strong generalization Year: (2023)
Ref_id:b11 Title: A survey on evaluation of large language models Year: (2024)
Ref_id:b12 Title: Progressive strategies for monte carlo tree search Year: (2009)
Ref_id:b13 Title: Parallel Monte-Carlo tree search Year: (2008-09-29)
Ref_id:b14 Title: Masked thought: Simply masking partial reasoning steps can improve mathematical reasoning learning of language models Year: (2024)
Ref_id:b15 Title: Plan-on-graph: Self-correcting adaptive planning of large language model on knowledge graphs Year: (2024)
Ref_id:b16 Title: Self-play fine-tuning converts weak language models to strong language models Year: (2024)
Ref_id:b17 Title: When is tree search useful for llm planning? Year: (2024)
Ref_id:b18 Title: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns Year: (2024)
Ref_id:b19 Title: Evaluating strategic reasoning abilities of LLM agents Year: (2024)
Ref_id:b20 Title: Multi-modal retrieval augmented generative commonsense reasoning Year: (2024)
Ref_id:b21 Title: Representation matters for mastering chess: Improved feature representation in alphazero outperforms switching to transformers Year: (2024)
Ref_id:b22 Title: Dynamic planning with a llm Year: (2023)
Ref_id:b23 Title: Everything of thoughts: Defying the law of penrose triangle for thought generation Year: (2023)
Ref_id:b24 Title: Uncovering the strategic reasoning limitations of llms via game-theoretic evaluations Year: (2024)
Ref_id:b25 Title: The proposed uscf rating system, its development, theory, and applications Year: (1967)
Ref_id:b26 Title: Is chess the drosophila of artificial intelligence? a social history of an algorithm Year: (2012)
Ref_id:b27 Title: Large language models are neurosymbolic reasoners Year: (2024)
Ref_id:b28 Title: Stop regressing: Training value functions via classification for scalable deep rl Year: (2024)
Ref_id:b29 Title: Towards revealing the mystery behind chain of thought: a theoretical perspective Year: (2024)
Ref_id:b30 Title: Bridging policy learning and language modeling Year: (2023)
Ref_id:b31 Title: Alphazero-like tree-search can guide large language model decoding and training Year: (2024)
Ref_id:b32 Title: Reasoning robustness of llms to adversarial typographical errors Year: (2024)
Ref_id:b33 Title: Strategic reasoning with language models Year: (2023)
Ref_id:b34 Title: Understanding social reasoning in language models with language models Year: (2024)
Ref_id:b35 Title: Stream of search (SoS): Learning to Year: ()
Ref_id:b36 Title: An improved benzene project for playing and solving hex with the help of deep neural networks Year: ()
Ref_id:b37 Title: Making pre-trained language models better few-shot learners Year: (2020)
Ref_id:b38 Title: A family of highly capable multimodal models Year: (2024)
Ref_id:b39 Title: Puzzle solving using reasoning of large language models: A survey Year: (2024)
Ref_id:b40 Title: Few shot chain-of-thought driven reasoning to prompt llms for open ended medical question answering Year: (2024)
Ref_id:b41 Title: Fourier circuits in neural networks: Unlocking the potential of large language models in mathematical reasoning and modular arithmetic Year: (2024)
Ref_id:b42 Title: Can LLMs solve molecule puzzles? a multimodal benchmark for molecular structure elucidation Year: (2024)
Ref_id:b43 Title: A survey on large language models: Applications, challenges, limitations, and practical usage Year: (2023)
Ref_id:b44 Title: Designing skill-compatible ai: Methodologies and frameworks in chess Year: (2024)
Ref_id:b45 Title: Reasoning with language model is planning with world model Year: (2023)
Ref_id:b46 Title: Clevrskills: Compositional language and visual reasoning in robotics Year: (2024)
Ref_id:b47 Title: Large language models as simulated economic agents: What can we learn from homo silicus? Year: (2023)
Ref_id:b48 Title: Training verifiers for selftaught reasoners Year: (2024)
Ref_id:b49 Title: Self-rewarding tree search for biomedical retrieval-augmented generation Year: (2024-11)
Ref_id:b50 Title: A survey on large language model-based game agents Year: (2024)
Ref_id:b51 Title: Pokellmon: A humanparity agent for pokemon battles with large language models Year: (2024)
Ref_id:b52 Title: Large language models cannot self-correct reasoning yet Year: (2024)
Ref_id:b53 Title: Evidence of learned look-ahead in a chess-playing neural network Year: (2024)
Ref_id:b54 Title: Tree-of-table: Unleashing the power of llms for enhanced large-scale table understanding Year: (2024)
Ref_id:b55 Title: Enhancing llm reasoning with reward-guided tree search Year: (2024)
Ref_id:b56 Title: Cladder: A benchmark to assess causal reasoning capabilities of language models Year: (2024)
Ref_id:b57 Title: Straus and Giroux Year: (2011)
Ref_id:b58 Title: MLLMcompbench: A comparative reasoning benchmark for multimodal LLMs Year: (2024)
Ref_id:b59 Title: Tree search for language model agents Year: (2024)
Ref_id:b60 Title: Large language models are zero-shot reasoners. Advances in neural information processing systems Year: (2022)
Ref_id:b61 Title:  Year: ()
Ref_id:b62 Title: Training language models to self-correct via reinforcement learning Year: (2024)
Ref_id:b63 Title: Openspiel: A framework for reinforcement learning in games Year: (2019)
Ref_id:b64 Title: -context reinforcement learning with algorithm distillation Year: (2022)
Ref_id:b65 Title: Can small language models help large language models reason better? Year: (2024)
Ref_id:b66 Title: Beyond a * : Better planning with transformers via search dynamics bootstrapping Year: (2024)
Ref_id:b67 Title: Codetree: Agent-guided tree search for code generation with large language models Year: (2024)
Ref_id:b68 Title: Large language modelempowered agents for simulating macroeconomic activities Year: (2023)
Ref_id:b69 Title: When LLMs meet cunning texts: A fallacy understanding benchmark for large language models Year: ()
Ref_id:b70 Title: Learning strategic skills by llms via bi-level tree search Year: (2024)
Ref_id:b71 Title: Large language model guided tree-of-thought Year: (2023)
Ref_id:b72 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b73 Title: Chameleon: Plug-andplay compositional reasoning with large language models Year: (2024)
Ref_id:b74 Title: The clrs-text algorithmic reasoning language benchmark Year: (2024)
Ref_id:b75 Title: Tree-of-traversals: A zero-shot reasoning algorithm for augmenting black-box language models with knowledge graphs Year: (2024)
Ref_id:b76 Title: Chess as the drosophila of ai Year: (1990)
Ref_id:b77 Title: Bridging chess mastery and ai innovation: The making of llm-chesscoach Year: (2023)
Ref_id:b78 Title: Whiteboard-ofthought: Thinking step-by-step across modalities Year: (2024)
Ref_id:b79 Title: Large language models: A survey Year: (2024)
Ref_id:b80 Title: An analysis of virtual loss in parallel MCTS Year: (2017)
Ref_id:b81 Title: Compositional chain-of-thought prompting for large multimodal models Year: (2024)
Ref_id:b82 Title: Mastering chess with a transformer model Year: (2024)
Ref_id:b83 Title: One step at a time: Language agents are stepwise planners Year: (2024)
Ref_id:b84 Title: From medprompt to o1: Exploration of run-time strategies for medical challenge problems and beyond Year: (2024)
Ref_id:b85 Title: Automatic multi-step reasoning and tool-use for large language models Year: (2023)
Ref_id:b86 Title: Dynamic strategy planning for efficient question answering with large language models Year: (2024)
Ref_id:b87 Title: Let's think dot by dot: Hidden computation in transformer language models Year: (2024)
Ref_id:b88 Title: Reasoning with large language models, a survey Year: (2024)
Ref_id:b89 Title: Measuring and narrowing the compositionality gap in language models Year: (2022)
Ref_id:b90 Title: Why think step by step? reasoning emerges from the locality of experience Year: (2024)
Ref_id:b91 Title: Agent q: Advanced reasoning and learning for autonomous ai agents Year: (2024)
Ref_id:b92 Title: From r to q * : Your language model is secretly a q-function Year: (2024)
Ref_id:b93 Title: Optimal decision making through scenario simulations using large language models Year: (2024)
Ref_id:b94 Title: Transformer based planning in the observation space with applications to trick taking card games Year: (2024)
Ref_id:b95 Title: Thinking forward and backward: Effective backward planning with large language models Year: (2024)
Ref_id:b96 Title: Amortized planning with large-scale transformers: A case study on chess Year: ()
Ref_id:b97 Title: LMAct: A benchmark for in-context imitation learning with long multimodal demonstrations Year: (2024)
Ref_id:b98 Title: Capabilities of gemini models in medicine Year: (2024)
Ref_id:b99 Title: Language models are greedy reasoners: A systematic formal analysis of chain-of-thought Year: (2022)
Ref_id:b100 Title: Measuring intelligence through games Year: (2011)
Ref_id:b101 Title: Mastering atari, go, chess and shogi by planning with a learned model Year: (2020)
Ref_id:b102 Title: Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning Year: ()
Ref_id:b103 Title: Synthetic prompting: Generating chain-ofthought demonstrations for large language models Year: (2023)
Ref_id:b104 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b105 Title: Language models are multilingual chain-of-thought reasoners Year: (2022)
Ref_id:b106 Title: Math-llava: Bootstrapping mathematical reasoning for multimodal large language models Year: (2024)
Ref_id:b107 Title: Mastering chess and shogi by self-play with a general reinforcement learning algorithm Year: (2017)
Ref_id:b108 Title: Beyond human data: Scaling self-training for problem-solving with language models Year: (2024)
Ref_id:b109 Title: Chain of thoughtlessness? an analysis of cot in planning Year: (2024)
Ref_id:b110 Title: Dawnicl: Strategic planning of problem-solving trajectories for zero-shot in-context learning Year: (2024)
Ref_id:b111 Title: Scaling instruction tuning for mathematical reasoning Year: (2024)
Ref_id:b112 Title: Toward self-improvement of llms via imagination, searching, and criticizing Year: (2024)
Ref_id:b113 Title: Benchmarking large language model (llm) performance for game playing via tic-tac-toe Year: (2024)
Ref_id:b114 Title: The fhourstones benchmark Year: ()
Ref_id:b115 Title: Can large language models play text games well? current state-of-the-art and open questions Year: (2023)
Ref_id:b116 Title: Can large language models play text games well? current state-of-the-art and open questions Year: (2023)
Ref_id:b117 Title: Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting Year: (2024)
Ref_id:b118 Title: softmax is not enough (for sharp out-ofdistribution) Year: (2024)
Ref_id:b119 Title: Cooperative strategic planning enhances reasoning capabilities in large language models Year: (2024)
Ref_id:b120 Title: Can language models solve graph problems in natural language? Year: (2024)
Ref_id:b121 Title: M4u: Evaluating multilingual understanding and reasoning for large multimodal models Year: (2024)
Ref_id:b122 Title: Measuring multimodal mathematical reasoning with mathvision dataset Year: (2024)
Ref_id:b123 Title: Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models Year: (2023)
Ref_id:b124 Title: Self-consistency improves chain of thought reasoning in language models Year: (2022)
Ref_id:b125 Title: Strategic chain-ofthought: Guiding accurate reasoning in llms through strategy elicitation Year: (2024)
Ref_id:b126 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b127 Title: Enhance reasoning for large language models in the game werewolf, 2024a Year: ()
Ref_id:b128 Title: Can graph learning improve planning in LLM-based agents? Year: ()
Ref_id:b129 Title: Conceptmath: A bilingual concept-wise benchmark for measuring mathematical reasoning of large language models Year: (2024)
Ref_id:b130 Title: Whodunitbench: Evaluating large multimodal agents via murder mystery games Year: ()
Ref_id:b131 Title: Monte carlo tree search boosts reasoning via iterative preference learning Year: (2024)
Ref_id:b132 Title: Large language models can learn temporal reasoning Year: (2024)
Ref_id:b133 Title: Evaluating world models with llm for decision making Year: (2024)
Ref_id:b134 Title: Buffer of thoughts: Thoughtaugmented reasoning with large language models Year: (2024)
Ref_id:b135 Title: Synergizing reasoning and acting in language models Year: (2022)
Ref_id:b136 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2024)
Ref_id:b137 Title: The unreliability of explanations in few-shot prompting for textual reasoning Year: (2022)
Ref_id:b138 Title: Bootstrapping reasoning with reasoning Year: (2022)
Ref_id:b139 Title: Quiet-star: Language models can teach themselves to think before speaking Year: (2024)
Ref_id:b140 Title: MR-ben: A meta-reasoning benchmark for evaluating system-2 thinking in LLMs Year: (2024)
Ref_id:b141 Title: Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b Year: (2024)
Ref_id:b142 Title: Transcendence: Generative models can outperform the experts that train them Year: (2024)
Ref_id:b143 Title: A careful examination of large language model performance on grade school arithmetic Year: (2024)
Ref_id:b144 Title: Llm as a mastermind: A survey of strategic reasoning with large language models Year: (2024)
Ref_id:b145 Title: On the diagram of thought Year: (2024)
Ref_id:b146 Title: Automatic chain of thought prompting in large language models Year: (2022)
Ref_id:b147 Title: Multimodal chain-of-thought reasoning in language models Year: (2023)
Ref_id:b148 Title: Large language models as commonsense knowledge for large-scale task planning Year: (2024)
Ref_id:b149 Title: Language agent tree search unifies reasoning acting and planning in language models Year: (2024)
Ref_id:b150 Title: Rh6 Ra1 59. Rh8 Rb1+ 60. Ka6 Rb2 61. Na7 Ra2+ 62. Kb7 Rb2+ 63. Kc8 Rb1 64. Rh6 Kf4 65. Ra6 Rh1 66 Year: ()
Ref_id:b151 Title: Fianchettoing the rook Date: 2024-11-14 White: Stockfish-L20 Black: MAV-MCTS(M = 2000 Year: ()
Ref_id:b152 Title:  Year: ()
Ref_id:b153 Title: Qf4 a5 30. Qf6 Qd7 31 Year: (1944)
Ref_id:b154 Title:  Year: ()
