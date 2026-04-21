Title: Algorithm Development in Neural Networks: Insights from the Streaming Parity Task
Abstract: Even when massively overparameterized, deep neural networks show a remarkable ability to generalize. Research on this phenomenon has focused on generalization within distribution, via smooth interpolation. Yet in some settings neural networks also learn to extrapolate to data far beyond the bounds of the original training set, sometimes even allowing for infinite generalization, implying that an algorithm capable of solving the task has been learned. Here we undertake a case study of the learning dynamics of recurrent neural networks (RNNs) trained on the streaming parity task in order to develop an effective theory of algorithm development. The streaming parity task is a simple but nonlinear task defined on sequences up to arbitrary length. We show that, with sufficient finite training experience, RNNs exhibit a phase transition to perfect infinite generalization. Using an effective theory for the representational dynamics, we find an implicit representational merger effect which can be interpreted as the construction of a finite automaton that reproduces the task. Overall, our results disclose one mechanism by which neural networks can generalize infinitely from finite training experience.

Section: Introduction
Examples of computational algorithms appearing in deep networks are numerous (Olah et al., 2020;Goh et al., 2021;Wang et al., 2022;Power et al., 2022;Nanda et al., 2023;Zhong et al., 2023). Furthermore, recurrent neural networks and transformers have shown a noteworthy ability to generalize (Loula et al., 2018; Lake & Baroni, 2018; Brown et al.,   1 Gatsby Computational Neuroscience Unit, University College London 2 Sainsbury Wellcome Centre, University College London.
Correspondence to: Loek van Rossem <loek.rossem.22@ucl.ac.uk>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). Left: Example sequences of the streaming parity task, from the training dataset (short sequences) and the validation dataset (long sequences). Right: The mean squared loss of a recurrent neural network during training on the streaming parity task. Note that the loss also goes to zero for sequences much longer than found in the training data. This was tested for sequences up to length 10000. 2020) in particular to sequences with lengths not seen in the training data (Dai et al., 2022;Abbe et al., 2023;Cohen-Karlik et al., 2023). These results are surprising as gradient descent provides no clear incentive to generalize beyond the training domain. Why do deep learning systems sometimes develop proper computational algorithms, instead of simply interpolating the data? Understanding this is crucial for the safe application of machine learning models, as for instance, models can appear to generalize initially, but break after moving too far away from the training domain (Anil et al., 2022;Zhou et al., 2024). Similarly, in neuroscience, an important goal is to try to figure out the types of algorithms the brain employs. This is particularly difficult since the brain is so complex, interconnected, and poorly understood (Thompson & Best, 1989;Olshausen & Field, 2006) that it is unclear how to even recognize an algorithm when we see it. Instead, studying the dynamics of algorithm development in the brain might be more tractable (Richards et al., 2019), as learning rules may not be as complex as the learned algorithm itself. Despite attempts to understand these dynamics (Zhou et al., 2021;El-Gaby et al., 2024), it is still a highly challenging problem by itself. Here we hope to provide a better sense of what to look for in the brain, by answering analogous questions in a simpler setting.
Let us consider a relatively simple computational problem for which algorithm development can still be studied: the streaming parity task (Figure 1). Given a sequence of zeros and ones with varying length, the aim of the task is to output a zero when the number of ones is even and output a one when the number of ones is odd. A recurrent neural network trained only on sequences up to some short finite length will sometimes generalize infinitely. It is able to solve the task accurately for any sequence length no matter how large, even for sequences thousands of times longer than shown in the training data.
As longer sequences are not within the same domain as shorter sequences, the generalization cannot simply be explained by interpolation. One can continue feeding in symbols and the network will continue predicting correctly, suggesting it has somehow learned a computational algorithm. It is unclear how such an algorithm can develop during training from gradient descent optimization. The network is trained to reduce its loss only on the shorter sequence dataset; it is not penalized for breaking after a certain length.
The main goal of this paper is to find simple mathematical models able to explain this seemingly surprising behavior. Our main contributions are as follows:
• In Section 3, we provide a local interaction theory for representational learning dynamics in recurrent neural networks.
• In Section 4, we explain how these interactions can result in the development of an algorithm capable of out-of-distribution generalization.
• In Section 5, we find algorithm development occurs in two phases: an initial tree fitting phase, and a secondary generalization phase.
this section cite: ['b39', 'b20', 'b43', 'b37', 'b57', 'b12', 'b0', 'b10', 'b2', 'b61', 'b50', 'b44', 'b58', 'b16']

Section: Automata and Recurrent Neural Networks

this section cite: []

Section: Interpreting Recurrent Neural Networks
The first step to understanding the development of computational algorithms in recurrent neural networks, is to choose the right way of representing the information that defines the network. The model's parameters are a complete but poor representation. They also contain a large amount of redundant information, e.g. swapping the order of neurons does not affect the encoded algorithm in any way.
For analyzing the inner structure of neural networks a better approach is to consider the geometry of the representational structure, i.e. how are the hidden activations corresponding to the data structured in the network (Lin et al., 2019;Williams et al., 2022;Lin & Kriegeskorte, 2023). This is however, still not ideal in the context of computational algorithms. Although understanding how data is represented in the network may be helpful, it does not directly say much about the nature of the computations being performed on that representational space. Moreover, the representational geometry can vary greatly across RNN architectures trained on the same task (Maheswaranathan et al., 2019).
One common approach to interpret RNNs is with dynamical systems (Sussillo & Barak, 2013;Laurent & Brecht, 2016;Can et al., 2020;Driscoll et al., 2024). Here, due to our focus on computational algorithms, we will instead be interpreting the recurrent neural network using deterministic finite automata (DFA), an approach also well known in the RNN literature (Servan-Schreiber et al., 1989;Giles et al., 1992;Omlin & Giles, 1996;Tino et al., 1999;Weiss et al., 2020;Merrill & Tsilivis, 2022;Michaud et al., 2024), which has also seen some applications in a neuroscience context (Turner et al., 2021;Brennan et al., 2023).
A DFA consists of a set of states including an initial state, transitions between the states given input symbols, and an output symbol for each state. An example of such a DFA solving the streaming parity task can be seen in Figure 2.
this section cite: ['b29', 'b56', 'b28', 'b32', 'b49', 'b27', 'b7', 'b14', 'b46', 'b19', 'b42', 'b51', 'b55', 'b34', 'b52', 'b5']

Section: Automaton Extraction
We will use a relatively simple method for constructing an automaton from the representational space of an RNN, as this will be enough for our purposes. Consider an abstract recurrent neural network:
h t = f h (h t-1 , x t ) y t = f y (h t ) ,(1)
where the exact forms of the recurrent map f h and output map f y will depend on the architectural details of the network.
From the hidden representations of this network we can extract a deterministic finite automaton. States are defined to be the hidden representations after receiving each sequence, transitions are determined by following where states go after the network received an input symbol, and outputs are simply the network's output map evaluated at each state. This procedure is illustrated in Figure 3.
When two different sequences of input symbols are assigned the same internal activation vector, an additional application Before the network has seen any input symbols the internal vector of hidden activations has some initial value h0, which we will call the initial state of the automaton. After seeing a symbol, say e.g. 1, the hidden activations change to a different vector f h (h0, 1). We will call this vector the state corresponding to input sequence 1, and say that the initial state has a transition to the state 1 after receiving the input symbol 1. We can repeat this procedure for all possible sequences, to define a state for each one and transitions on each state for every possible input symbol. We can assign an output to each state by evaluating the output map on its corresponding representation.
of the recurrent map will send them to the same internal activation vector for any possible input symbol received. Their activations will remain the same in the future, and the output map will always assign them the same output from that point on. They can no longer be distinguished, so we will consider them to be the same state in the automaton.
If enough representations overlap, it may be the case that all transitions go to already existing states. At this point, we have a finite set of states capable of representing all possible internal states of the network. These states, together with the transitions between them and their outputs, form a discrete computational algorithm capable of producing outputs on input sequences, which match the outputs of the RNN. For more details on automaton extraction, see Appendix A.
this section cite: []

Section: Automata During Training
In order to visualize the development of an algorithm, we extract automata at each epoch from an RNN during training on the streaming parity task (Figure 4). We can see that, due to random small weights, the automaton initially has few states and random outputs. As the model trains, the automaton expands into a complete tree fitting the training data. Then, right when the training loss becomes zero, states in the automaton appear to merge until it becomes finite, and we see generalization on all data. To understand this better, we will first try to model the representational dynamics, and then study the induced dynamics on the automaton.
this section cite: []

Section: Implicit State Merger

this section cite: []

Section: Intuition
Why would states merge during training? The representational space is typically high-dimensional, so it seems statistically unlikely that many different representations end up at the same vector by chance. The key insight here is that, due to continuity, sometimes the fastest way for gradient descent to minimize the loss is to merge nearby representations.
As an example for illustration, suppose that at some point during training, two sequences in the dataset agree on target outputs, and one already has the correct predicted output. Then, if the recurrent map f h adjusts to move the other representation closer, its target output will also move towards the correct prediction, as the output map f y is continuous. Continuity thus gives rise to an interaction effect between nearby representations, potentially resulting in merging.
Implicit bias from gradient descent is a well-studied topic in the deep learning literature (Neyshabur et al., 2015;Gunasekar et al., 2018;Chizat & Bach, 2020;Soudry et al., 2022). However, the relatively simple effect we are considering here will turn out to be particularly interesting in the context of algorithm development in RNNs.
this section cite: ['b38', 'b21', 'b48']

Section: Interaction Model
Let us attempt to formalize this intuition by modeling the interaction between two nearby states, using the modeling approach from (van Rossem & Saxe, 2024), adapted for recurrent networks. Suppose that we have two input sequences
x (1) 1 • • • x (m1) 1 and x (1) 2 • • • x (m2) 2 in our dataset, with corresponding hidden representations h(x (1) 1 • • • x (m1) 1 ), h(x (1) 2 • • • x (m2) 2
). We would like to understand the behavior of these representations when they get near each other during training, in an arbitrary neural network.
Arbitrary architecture with high expressivity. Instead of analyzing this interaction for a specific recurrent architecture, we model it for an arbitrary network with high expressivity, meaning any network large and complex enough to have the freedom to behave like a smooth map. Under this assumption, we replace the effect of the assignment of hidden states by the network's parameters with two arbitrarily optimizable vectors h 1 , h 2 , and replace the effect of the network assigning output prediction to those hidden states with arbitrarily optimizable smooth maps y 1 , . . . , y N (schematized in Figure 5). Note that because we are dealing with a recurrent architecture, there may be multiple data points which have h(x
(1) 1 • • • x (m1) 1 ) or h(x (1) 2 • • • x (m2) 2
) as intermediate hidden states. Thus, we have N output maps, one for each possible sequence after x To simplify the theoretical analysis, we assume here that any sequence x  . The effect of the model assigning their representations is abstracted away with arbitrarily optimizable vectors h1 and h2, and the effect of the model assigning outputs for every of the N subsequent sequences in the training data is abstracted away with arbitrarily optimizable smooth maps y1, . . . , yN .
We will take gradients with respect to these arbitrary smooth maps. While a neural network may not optimize with smooth map dynamics and the gradient will depend on the architectural details, universal approximation theorems (Hornik et al., 1989;Csáji, 2001) tell us that an expressive one is at least able to freely model smooth maps. Therefore, we are modeling a simple behavior that, at least in principle, any expressive network can exhibit. The exact details of the architecture are not required to understand the key mechanisms by which the network learns to solve the task. Abstracting them away may help to find simpler and more intuitive explanations.
Note also that the maps h 1 , h 2 , y 1 , . . . y N may share parameters. They are not independent smooth maps. However, if the network is expressive enough, it can still have enough freedom to effectively optimize each map independently at specific points, so we will choose to ignore potential interaction effects arising from parameter sharing.
Local linear approximation. As we are trying to model the interaction between two representations, we will consider the case where the distance dh := h 2 -h 1 between them is small. We can thus take linear approximations of the maps y 1 , . . . , y N around the representational mean
1 2 (h 2 + h 1 ): y(x (1) α • • • x (nα) α x(1) i • • • x(mi) i ) ≈ ȳi + 1 2 D yi (h α -h ¬α ),(2)
where α ∈ 1, 2 and i ∈ 1, • • • , N .
In general, the dynamics are undefined without specifying the parameterization. However, for the local linearized system, there is a unique choice of parameterindependent dynamics, namely by optimizing with respect to the effective linear parameters of the network h 1 , h 2 , ȳ1 , . . . , ȳN , D y1 , . . . , D y N . For the mean squared loss
L = 1 2 ⟨||ȳ i + 1 2 D yi (h α -h ¬α ) -y * α,i || 2 ⟩ α=1,2,i=1...,N ,(3)
after taking the continuous-time limit and considering solutions where representations either move closer or further away form each other, we find the self-contained 3-scalar system
d dt ||dh|| 2 = - 1 2 1 τ h ⟨w i ⟩ i d dt ⟨||dy i || 2 ⟩ i = - 1 2 ( 1 N τ y ||dh|| 2 + 1 τ h ⟨||dy i || 2 ⟩ i ||dh|| 2 )⟨w i ⟩ i d dt ⟨w i ⟩ i = - 1 4 1 N τ y (3⟨w i ⟩ i -⟨||dy i || 2 ⟩ i + ⟨||y * 2,i -y * 1,i || 2 ⟩ i )||dh|| 2 - 1 4 1 τ h ⟨w i ⟩ i ||dh|| 2 (⟨||dy i || 2 ⟩ i + ⟨w i ⟩ i ),(4)
where ||dh|| 2 is the squared representational distance, ⟨||dy i || 2 ⟩ is the average squared prediction distance, and
⟨w i ⟩ i := ⟨||dy i || 2 -dy ⊤ i (y * 2,i -y * 1,i
)⟩ i the average of an output alignment metric. The constants 1/τ h and 1/τ y are the effective learning rates of the representational map and output map respectively, as we are now optimizing with respect to these smooth maps as opposed to the model's parameters. The derivation and more details can be found in Appendix B.1.
Note that because we are considering the continuous-time limit, no noise has been introduced, and we also did not include any form of regularization. Any results in terms of generalization are from inductive biases in gradient descent alone, e.g. the intuition discussed in Section 3.1. Regularization and noise may also play a crucial role in generalization in some settings (Zhou et al., 2019;Ziyin et al., 2025).
this section cite: ['b24', 'b11', 'b60']

Section: State Merger Condition
The final representational distance is exactly solved in Appendix B.2 to find:
||dh|| 2 = 1 2 A high + 1 4 A 2 high + A 2 low ,(5)
where
A high = ||dh(0)|| 2 - N τ y τ h ⟨ ||dy i (0)|| 2 ||dh(0)|| 2 ⟩ i A low = N τ y τ h • ⟨||y * 2,i -y * 1,i || 2 ⟩ i .(6)
Note that these results are not dependent on the details of the neural network architecture used here. They are the results of an interaction effect locally present in any smooth, expressive machine learning system with hidden representations.
A merger occurs when the final representational distance is zero, which is when
A low = 0 and A high < 0.(7)
Suppose the network's parameters are initialized at some scale G < 1, where G is the average decrease of representational distances when applying the recurrent map. We roughly have
||dh(0)|| 2 ∝ G m ⟨ ||dy i (0)|| 2 ||dh(0)|| 2 ⟩ i ∝ G n ,(8)
where m = min (m 1 , m 2 ) and n = min (n 1 , . . . , n N ) are the minimal sequence lengths of the sequences corresponding to the representations and their potential subsequences respectively. Equation ( 7) then reduces to the condition
∀ i y * 1,i = y * 2,i and C < N • G n-m ,(9)
where C is an unknown constant depending on the network architecture.
We can make three observations from this:
1. The only states that can merge are those which always agree on outputs after receiving both the same sequence.
2. States only merge if the sequences they correspond to are long enough.
3. Mergers only start to occur given enough data and small enough initial weights.
We will study these observations and their implications in more detail in the next section.
this section cite: []

Section: Automaton Development

this section cite: []

Section: System of Interacting Particles
The interaction model requires the states to be close to each other. It does not model the global behavior of the dataset during training, only the local interaction between any two states. Many other things may occur during training, which are potentially more complex and exist on a global scale. We will ignore these effects here and treat the representational learning dynamics as a system of locally interacting particles (Liu et al., 2022;Geshkovski et al., 2023). The aim is to investigate how much of algorithm learning in recurrent neural networks can already be explained from this simple interaction alone.
this section cite: ['b18']

Section: Developing an Algorithm
We can see from Equation ( 9) that two states will only merge if they agree on target outputs for any possible subsequence in the data, i.e. they need not be distinguished in order to solve the task 1 . Indeed, out of all 103460 pairs of representations that ended up merging, all agreed on all possible future outputs.
Once enough of such pairs merge, the automaton will become finite. Because the merging of agreeing pairs does not affect the output of the automaton, it will still predict correctly on the training data. If, such as for the streaming parity task, the task can be expressed with a finite automaton, and the training dataset is large enough, the learned automaton and task automaton must agree on all possible sequences. In particular, as long as the training dataset contains all sequences up to the length of the task automaton's size, it is guaranteed that the automata are equivalent, as all its reachable states can be reached with the training data, on which the two automata agree.
When the learned automaton becomes finite, the behavior of the RNN becomes fixed for any sequence length, and we should see instant generalization on all lengths. This sudden complete generalization can be observed in Figure 6.
this section cite: []

Section: Redundant States
Since weights are initialized small, i.e. G < 1, it follows from Equation ( 9) that the first pairs to merge when decreasing the weight initialization are the ones for which n -m is minimal. Therefore, representations corresponding to shorter sequence pairs may not merge, even when enough longer sequences do. For the training data used here, the smallest possible n is 0, so we should start to see mergers once m reaches a certain threshold, which can be observed in Figure 7. 1 In the language of automata theory, this is equivalent to the absence of a distinguishing extension on the pair of input sequences.  Because not all pairs agreeing on all future outputs merge, redundant states will be present in the learned automaton. In fact, from Figure 7 it can be seen that not even all of the agreeing pairs that reached the minimum length threshold ended up merging. A possible explanation for this is that some pairs that never end up getting close during training, and therefore the effects from the interaction model do not apply.
It is not necessary for all agreeing pairs to merge to fully generalize, only enough for the final automaton to become finite. Redundant states are consequently expected to be learned. The final automaton in Figure 4 has far more states than the minimal two required to solve the task. However, it is still equivalent in computational function to the two-state automaton shown in Figure 2, which can be seen by merging all its redundant states (Figure 8).
this section cite: []

Section: Redundant states in the brain
There is some evidence of redundant states in the brain (Morcos & Harvey, 2016;Marmor et al., 2023). From a functional perspective, it is not so clear why these states may exist. However, as can be seen from the above argument, purely from a learning dynamics perspective, they are expected in the final learned automaton. Similar representations containing seemingly identical information have also been observed in other machine learning settings such as wide feed-forward networks (Doimo et al., 2021).
this section cite: ['b36', 'b33', 'b13']

Section: Full Generalization Phase Transition
Finally, we can see from Equation ( 9) that the merger condition only holds given a small enough initial weight scale G and a large enough number of datapoints N . As either the weight initialization is decreased or the training set size is increased, agreeing states will start to merge. This can be seen on the left side of Figure 9. At some point, enough states will have merged such that the automaton becomes finite, and the RNN will generalize to all sequences. Since no state mergers can occur before the merger condition is reached, we see a sharp boundary in number of states and in particular in the validation accuracy on the right side of Figure 9, splitting the training setting landscape into two regimes: one where it fits the training data with a complete tree, and one where it learns a finite, generalizing representation of the task. Such a separation resembles the phenomenon of rich
Training data fraction Training data fraction Initial weight scale Initial weight scale Validation accuracy Number of states and lazy learning found in RNNs (Schuessler et al., 2024) and other settings (Chizat et al., 2020;Flesch et al., 2021;Atanasov et al., 2022).
this section cite: ['b45', 'b17', 'b4']

Section: Merger Dynamics

this section cite: []

Section: Two Development Phases
Something that remains unclear about the dynamics in Figure 4 is the presence of an initial phase where the automaton expands into a complete tree of all possible sequences, before a phase of mergers resulting in a finite algorithm. Why would the RNN learn to memorize individual outputs per input sequence in the first place, only to collapse into a finite automaton later?
this section cite: []

Section: Fixed Expansion Point Interaction Model
A relatively simple dynamical explanation for this behavior can be found in the representational drift of each interaction pair. If the two representations during an interaction start to drift, and their effective learning rates 1/τ h1 , 1/τ h2 differ, they will drift at different speeds. In this case, their distance may initially start to increase as one outpaces the other.
The local interaction model we have considered does not exhibit this behavior, as the linear expansion point is chosen to be at the moving representational mean, enforcing a representational movement symmetry. In Appendix B.3 the analytical learning trajectory for an agreeing pair is solved for in this model, and their representational distance can be seen to decay exponentially.
To allow for enough freedom in the interaction model for both representations to drift freely, we can instead keep the expansion point fixed:
y(x (1) α • • • x (nα) α x(1) i • • • x(mi) i ) ≈ ȳi + D yi h α . (10
)
The downside of this choice is that as the pair drifts far away from the expansion point, the interaction model may lose accuracy. In Appendix B.4 we use a similar approach as before to reduce these dynamics to a self-contained system of 9 variables.
this section cite: []

Section: Diverging Mergers
We can see from numerical solutions to this system (Figure 10) that agreeing pairs initially diverge before they end up merging. This divergence is only present when the effective learning rates 1/τ h1 , 1/τ h2 differ. Experimentally, we find qualitatively similar divergence behavior in the RNN during training. We also see that the divergence occurs more often in pairs with a higher effective learning rate difference (Appendix D.3). Such an initial divergence of many agreeing pairs, may explain the tree fitting phase. A division in two phases with some similarities has been studied in feed-forward networks in (Shwartz-Ziv & Tishby, 2017).
this section cite: []

Section: Epochs Epochs Epochs
Theory Experiment
Training loss Distance agreeing pair Distance disagreeing pair Loss Figure 10. Left: Numerical solutions of the fixed expansion point interaction model. The representational distance of a pair agreeing on all possible future outputs first diverges before finally merging, whereas a pair which disagrees on some outputs only diverges. Right: Normalized experimentally observed representational distances from a randomly selected pair with agreeing outputs that ended up merging, and a randomly selected pair with disagreeing outputs.
6. Other Settings 6.1. Random Regular Tasks
The ideas considered in this paper should be applicable to any task that can be described by an automaton. Therefore, all experiments were also performed on a set of tasks defined by randomly generated automata. Similar results were found as with the streaming parity task (Appendix D.5).
this section cite: []

Section: Architecture Independence
The local interaction models discussed here are universal with respect to the neural network architecture. They represent intuitions that apply to any model with a smooth, expressive recurrent map on some hidden space and a similarly smooth and expressive output map on this space. To illustrate this, we replaced the ReLU activation function with a hyperbolic tangent and found similar qualitative results (Appendix D.6).
this section cite: []

Section: Transformers
Transformers have been shown to exhibit a similar ability to learn computational algorithms. One may wonder to what extent the ideas considered here for recurrent networks still apply to the transformer architecture.
The interpretation of the internal structure via an automaton is not as clearly applicable to a transformer. The recurrent map was an essential ingredient in the understanding of the formation of a finite automaton, as it allows for mergers to result in automaton transitions going to previous states. Interestingly enough, despite some generalization to larger sequence lengths, transformers fail to fully generalize to sequences of arbitrary length on parity computation (Anil et al., 2022).
However, the intuition behind the local interactions from continuity still applies, and so we may still find similar merg-ing dynamics. To investigate this, we compute the number of states in a transformer during training on the modular subtraction task from (Power et al., 2022). As can be seen from Figure 11, we do not find a clear state merger pattern in the hidden representations of the transformer. However, we do see a state merger pattern in the attention matrix that is reminiscent of the two phases found in recurrent networks. A similar pattern was found for a local complexity measure in (Humayun et al., 2024).
The merging of attention patterns may possibly play an important role in out-of-distribution generalization, similar to representation merger in recurrent neural networks. Other phenomena observed here in RNNs also resemble behaviors in transformers, such as the sudden transition to full generalization (Hoffmann et al., 2024). The exact way in which a transformer can learn an algorithm from mergers is not as clear and requires further study.
this section cite: ['b2', 'b43', 'b25', 'b22']

Section: Discontinuity
One of the assumptions in the interaction model is continuity of the recurrent and output maps. This may not necessarily be a reasonable assumption in the context of neuroscience, as the spiking coupling between neurons is not typically viewed as continuous. For the intuitions to work, however, we only really need predictions of nearby representations to move closer when the representations do on average. Continuity gives us this, but may not be a necessary condition. To explore this, we add a step-continuity in the output map of the recurrent network. We find qualitatively similar results, albeit with noisier dynamics Appendix D.9.
this section cite: []

Section: Conclusion
Despite the surprising nature of infinite generalization from finite data, there exists a setting in which it can be understood through relatively simple intuitions about inductive bias in gradient descent. In this setting, we found that algorithm development occurs in two phases, an initial tree fitting phase and a secondary merging phase that results in generalization. The merging phase occurs via a phase transition, when the right training conditions are met. Therefore, algorithm learning and infinite generalization can occur in deep networks, but not consistently. We also saw that from a dynamical perspective, redundant states are expected in the final learned algorithm. This is of particular interest to neuroscience, as it suggests that different animals may learn different but equivalent versions of an algorithm, which is something that should be taken into account when comparing representations. Finally, we found that intuitions about automaton formation do not apply as well to transformers, and that at least for some specific tasks, recurrent networks have an advantage in terms of infinite generalization.
this section cite: []

Section: Limitations
The theoretical approach used here is relatively simple and not necessarily a realistic model of the complete learning dynamics. Higher-order local interactions, global interactions, inductive biases from architectural choices, regularization, and noise were ignored, but may have additional effects on algorithm development worth studying. Additionally, the interpretation of an RNN as an automaton may provide incomplete information in more complex or continuous data settings. Other mathematical objects may be necessary to properly represent the internal structure of an RNN in such settings.
this section cite: []

Section: Impact Statement
This paper presents work whose goal is to increase understanding of deep learning, which may lead to advancements in the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.
Algorithm 1 Hopcroft's Algorithm Input: set of states Q with output 0, set of states F with output 1 Output: minimal state partition P P := {F, Q \ F } W := {F, Q \ F } while W is not empty do choose and remove a set A from W for c in Σ do let X be the set of states for which a transition on c leads to a state in A for set Y in P for which X ∩ Y is nonempty and Y \ X is nonempty do replace Y in P by the two sets X ∩ Y and Y \ X if Y is in W then replace Y in W by the same two sets else if |X ∩ Y | ≤ |Y \ X| then add X ∩ Y to W else add Y \ X to W end if end if end for end for end while return P B. Details Theoretical Analysis B.1. Reduction to a 3-dimensional System To model the two-point interaction we consider two sequences x (1) 1 . . . x (m1) 1 and x (1) 2 . . . x (m2) 2 with nearby representations h 1 = h(x (1) 1 . . . x (m1) 1 ) respectively h 2 = h(x (1) 2 . . . x (m2) 2
). Let D = {(x 1,i , y * 1,i ), (x 2,i , y * 2,i )} N i=1 be the set of all datapoints contained within the training dataset which have
x (1) 1 . . . x (m1) 1 or x (1) 2 . . . x (m2) 2
as a subsequence. Assuming high expressivity, we model h 1 and h 2 as freely optimizable vectors and the output predictions of the network for each subsequent sequence in D as given by smooth, freely optimizable maps y i : H → Y .
In contrast to (van Rossem & Saxe, 2024), we cannot use an optimizable linearized hidden map as here we cannot smoothly vary the inputs for the hidden map. Since our input symbols are discrete, we can instead consider arbitrarily optimizable hidden vectors, as no two differing input symbols can ever get arbitrarily close to each other in the input space.
As h 1 and h 2 are close, we take a linear approximation of each output prediction map around the representational mean:
y(x α,i ) = ȳi + 1 2 D yi (h α -h ¬α ). (13
)
The mean squared loss in this approximation takes the form:
L = 1 2 ⟨||ȳ i + 1 2 D yi (h α -h ¬α ) -y * α,i || 2 ⟩ D .(14)
We apply gradient decent optimization directly with respect to D yi , h α and ȳi , resulting in the dynamics:
d dt ȳi = - 1 τ ȳi ∂L ∂ ȳi = - 1 τ ȳi 1 N ⟨ ȳi + 1 2 D yi (h α -h ¬α ) -y * α,i ⟩ α=1,2 = - 1 τ ȳi 1 N ( ȳi - y * 2,i + y * 1,i2
)
d dt h(α) = - 1 τ hα ∂L ∂h α = - 1 τ hα 1 4 ⟨D ⊤ yi (ȳ i + 1 2 D yi (h α -h ¬α ) -y * α,i ) -D ⊤ yi (ȳ i + 1 2 D yi (h ¬α -h α ) -y * ¬α,i )⟩ i=1,...,N = - 1 τ hα 1 4 ⟨D ⊤ yi (D yi (h α -h ¬α ) -(y * α,i -y * ¬α,i ))⟩ i=1,...,N d dt D yi = - 1 τ yi ∂L ∂D yi = - 1 τ yi 1 N 1 2 ⟨( 1 2 D yi (h α -h ¬α )(h α -h ¬α ) ⊤ + ( ȳi -y * α,i )(h α -h ¬α ) ⊤ )⟩ α=1,2 = - 1 τ yi 1 N 1 4 (D yi (h 2 -h 1 ) -(y * 2,i -y * 1,i ))(h 2 -h 1 ) ⊤ ,(15)
where we used the matrix differentiation identities
∂a ⊤ Xb ∂X = ab ⊤ , ∂a ⊤ X ⊤ CXa ∂X = (C + C ⊤ )Xaa ⊤ and ∂||Ax+b|| 2 ∂x = 2A ⊤ (Ax + b).
The ȳi dynamics are decoupled and can be solved directly:
ȳi (t) = y * 2,i + y * 1,i 2 + (y i (0) - y * 2,i + y * 1,i 2 )e -1 τ ȳi 1 N t ,(16)
the solution of which takes the form of exponential decay towards each pairs target output mean.
Define
dh := h 2 -h 1 , dy i := D yi (h 2 -h 1 ), w i := ||dy i || 2 -dy ⊤ i (y * 2,i -y * 1,i ).
We take as an Anzats representational movement of the two points towards or away from each other, i.e.
d dt dh ∝ dh =⇒ d dt dh || d dt dh|| = dh ||dh|| =⇒ d dt dh = || d dt dh|| ||dh|| dh (17
)
Applying this twice allows us to write
D yi d dt dh = || d dt dh|| ||dh|| D yi dh = || d dt dh|| ||dh|| ||dh|| 2 ||dh|| 2 D yi dh = dh ⊤ ( || d dt dh|| ||dh|| dh) ||dh|| 2 D yi dh = dh ⊤ d dt dh ||dh|| 2 D yi dh = 1 2 d dt ||dh|| 2 ||dh|| 2 D yi dh ,(18)
which we can use to find a self-contained scalar system:
d dt ||dh|| 2 = 2dh ⊤ d dt dh = dh ⊤ (- 1 2 1 τ h ⟨D ⊤ yi (D yi dh -(y * 2,i -y * 1,i ))⟩ i=1,...,N ) = - 1 2 1 τ h ⟨||dy i || 2 -dy ⊤ i (y * 2,i -y * 1,i )⟩ i=1,...,N = - 1 2 1 τ h ⟨w i ⟩ i=1,...,N d dt ||dy i || 2 = 2dy ⊤ i d dt dy i = 2dy ⊤ i ( Ḋyi dh + dh ⊤ d dt dh ||dh|| 2 D yi dh) = 2dy ⊤ i ( Ḋyi dh + 1 2 d dt ||dh|| 2 ||dh|| 2 D yi dh) = 2dy ⊤ i (- 1 τ yi 1 N 1 4 (dy i -(y * 2,i -y * 1,i ))||dh|| 2 - 1 4 1 τ h ⟨w i ⟩ i=1,...,N ||dh|| 2 dy i ) = - 1 τ yi 1 N 1 2 (||dy i || 2 -dy ⊤ i (y * 2,i -y * 1,i ))||dh|| 2 - 1 2 1 τ h ⟨w i ⟩ i=1,...,N ||dy i || 2 ||dh|| 2 d dt w i = (2dy i -(y * 2,i -y * 1,i )) ⊤ d dt dy i = (2dy i -(y * 2,i -y * 1,i )) ⊤ (- 1 τ yi 1 N 1 4 (dy i -(y * 2,i -y * 1,i ))||dh|| 2 - 1 4 1 τ h ⟨w i ⟩ i=1,...,N ||dh|| 2 dy i ) = - 1 τ yi 1 N 1 4 (3w i -||dy i || 2 + ||y * 2,i -y * 1,i || 2 )||dh|| 2 - 1 4 1 τ h ⟨w i ⟩ i=1,...,N ||dh|| 2 (||dy i || 2 + w i ) ,(19)
where
1 τ h = 1 τ h 1 + 1 τ h 2 .
In the case that the output effective learning rates are all equal, i.e. ∀ i τ yi = τ y , this system can be reduced to a 3-dimensional scalar system:
d dt ||dh|| 2 = - 1 2 1 τ h ⟨w i ⟩ i d dt ⟨||dy i || 2 ⟩ i = - 1 2 ( 1 N τ y ||dh|| 2 + 1 τ h ⟨||dy i || 2 ⟩ i ||dh|| 2 )⟨w i ⟩ i d dt ⟨w i ⟩ i = - 1 4 1 N τ y (3⟨w i ⟩ i -⟨||dy i || 2 ⟩ i + ⟨||y * 2,i -y * 1,i || 2 ⟩ i )||dh|| 2 - 1 4 1 τ h ⟨w i ⟩ i ||dh|| 2 (⟨||dy i || 2 ⟩ i + ⟨w i ⟩ i ).(20)
this section cite: ['b53']

Section: B.2. Final Representational Structure
In order to study the final representational structure learned by the network, we solve the final representational distance for the pair. Using the relationship d dt
⟨||dy i || 2 ⟩ i ||dh|| 2 = ||dh|| 2 d dt ⟨||dy i || 2 ⟩ i -⟨||dy i || 2 ⟩ i d dt ||dh|| 2 ||dh|| 4 = - 1 2N τ y ⟨w i ⟩ i = τ h N τ y d dt ||dh|| 2 ,(21)
we can solve ⟨||dy i || 2 ⟩ i (t) as a function of ||dh|| 2 (t):
⟨||dy i || 2 ⟩ i (t) = τ h N τ y ||dh|| 4 (t) + ⟨||dy i (0)|| 2 ⟩ i ||dh(0)|| 2 - τ h N τ y ||dh(0)|| 2 ||dh|| 2 (t),(22)
reducing the dynamics to a 2-dimensional system:
d dt ||dh|| 2 = - 1 2 1 τ h ⟨w i ⟩ i d dt ⟨w i ⟩ i = - 1 4 (- τ h N 2 τ y 2 ||dh|| 6 + 1 N τ y ||y 2 -y 1 || 2 ||dh|| 2 + 4 N τ y ||dh|| 2 ⟨w i ⟩ i + 1 τ h ⟨w i ⟩ 2 i ||dh|| 2 + ||dy(0)|| 2 ||dh(0)|| 2 - τ h N τ y ||dh(0)|| 2 ( 1 τ h ⟨w i ⟩ i - 1 N τ y ||dh|| 4 )).(23)
This system has three fixed points
||dh|| 2 = 1 2 A high - 1 4 A 2 high + A 2 low , ⟨w i ⟩ i = 0 ||dh|| 2 = 1 2 A high + 1 4 A 2 high + A 2 low , ⟨w i ⟩ i = 0 ||dh|| 2 = 0, ⟨w i ⟩ i = 0 ,(24)
where
A high = ||dh(0)|| 2 - N τ y τ h ⟨||dy i (0)|| 2 ⟩ i ||dh(0)|| 2 A low = N τ y τ h • ⟨||y * 2,i -y * 1,i || 2 ⟩ i .(25)
The first fixed point has negative representational distance and is thus not a valid solution.
The second fixed point has Jacobian
J = 0 -1 τ h 1 4 τ h N 2 τ 2 y (2A 2 low + 1 2 (A high + A 2 high + 4A 2 low ) 2 ) -1 τy ( 1 2 A high + A 2 high + 4A 2 low ) ,(26)
with negative trace
Tr(J) = - 1 N τ y ( 1 2 A high + A 2 high + 4A 2 low ),(27)
and positive determinant
det(J) = 1 4 1 N 2 τ 2 y (2A 2 low + 1 2 (A high + A 2 high + 4A 2 low ) 2 ),(28)
and is therefore always stable.
The final fixed point has Jacobian
J = 0 -1 τ h 1 2 ( 1 τ h ⟨wi⟩ 2 i ||dh|| 4 -1 N τy ⟨||y * 2,i -y * 1,i || 2 ⟩ i ) ( 1 2 1 τy A high -1 τ h ⟨wi⟩i ||dh|| 2 ) , (29
)
which cannot be directly evaluated at
⟨w i ⟩ i = 0, ||dh|| 2 = 0 because of the undetermined term ⟨wi⟩i ||dh|| 2 . By replacing ⟨wi⟩i ||dh|| 2
with the direction of approach b a we can solve for eigenvectors
0 -1 τ h 1 2 ( 1 τ h b 2 a 2 -1 N τy ⟨||y * 2,i -y * 1,i || 2 ⟩ i ) ( 1 2 1 N τy A high -1 τ h b a ) a b = λ a b (30) to find v ± = 1 -τ h N τy Ahigh± √ A 2 high +4A 2 low 2(31)
with one positive and one negative eigenvalue
λ ± = 1 N τ y A high ± A 2 high + 4A 2 low 2 . (32
)
There is always one direction along which perturbations increase, so this fixed point is not stable.
Only the second fixed point is valid and stable, hence we expect the final representational distance to reach it.
this section cite: []

Section: B.3. Agreeing Pair Dynamics
When the pair agrees on all possible future outputs, i.e. ∀ i y * 2,i = y * 1,i we have that ⟨w i ⟩ i = ⟨||dy i || 2 ⟩ i , allowing us to reduce the system Equation ( 20) to
d dt ||dh|| 2 = - 1 2 1 τ h ⟨||dy i || 2 ⟩ i d dt ⟨||dy i || 2 ⟩ i = - 1 2 ⟨||dy i || 2 ⟩ i ( 1 N τ y ||dh|| 2 + 1 τ h ⟨||dy i || 2 ⟩ i ||dh|| 2 ).(33)
Using Equation ( 22) we can write a self-contained equation for ||dh||(t):
d dt ||dh|| 2 = - 1 2 1 N τ y ||dh|| 4 + 1 2 1 N τ y A high ||dh|| 2 , (34
) which is Bernoulli and has solution ||dh(t)|| 2 = A high 1 + ( Ahigh ||dh(0)|| 2 -1)e -1 2 1 N τy Ahight , (35) which, as A high ≤ ||dh(0)|| 2 exponentially decays towards the final representational distance ||dh(∞)|| 2 = A high .
this section cite: []

Section: B.4. Fixed Expansion Point Interaction Model
We take the same approach before but instead keep the linear expansion point fixed during training:
y(x α,i ) = b i + D yi h α . (36
)
The mean squared loss in this approximation has the form:
L = 1 2 ⟨||b i + D yi h α -y * α,i || 2 ⟩ D .(37)
Motivated by the assumption of high model expressivity, we apply gradient decent optimization directly with respect to D yi , h α and b i , resulting in the dynamics:
d dt b i = - 1 τ bi ∂L ∂b i = - 1 τ bi 1 N ⟨b i + D yi h α -y * α,i ⟩ α=1,2 = - 1 τ ȳi 1 N (b i + D yi ⟨h α ⟩ α -⟨y * α,i ⟩ α ) d dt h α = - 1 τ hα ∂L ∂h α = - 1 τ hα 1 2 ⟨D ⊤ yi (b i + D yi h α -y * α,i )⟩ i=1,...,N d dt D yi = - 1 τ yi ∂L ∂D yi = - 1 τ yi 1 N ⟨D yi h α h ⊤ α + (b i -y * α,i )h ⊤ α ⟩ α=1,2 = - 1 τ yi 1 N (D yi ⟨h α h ⊤ α ⟩ α=1,2 + b i ⟨h ⊤ α ⟩ α=1,2 -⟨y * α,i h ⊤ α ⟩ α=1,2 ).(38)
We again try the Ansatz where the representations only move towards or away from each other, which, since we take the expansion point to be the representational mean at t = 0, can be written by shifting coordinates without loss of generality as
h α ∝ h α v,(39)
for some vector v with ||v|| = 1. We define
d i := v ⊤ D yi v, b i := v ⊤ D ⊤ yi b i , allowing us write using the derivatives d dt h α = - 1 τ hα 1 2 ⟨(D yi v) ⊤ (b i + h α D yi v -y * α,i )⟩ i d dt b i = - 1 τ bi 1 N (b i + ⟨h α ⟩ α D yi v -⟨y * α,i ⟩ α ) d dt D yi v = - 1 τ yi 1 N (⟨h 2 α ⟩ α D yi v + ⟨h α ⟩ α b i -⟨h α y * α,i ⟩ α ),(40)
a scalar system which takes the form
d dt h α = - 1 τ hα 1 2 ⟨b ⊤ i D yi v + h α ||D yi v|| 2 -y ⊤ α,i D yi v⟩ i d dt b ⊤ i D yi v = - 1 τ bi 1 N (b ⊤ i D yi v + ⟨h α ⟩ α ||D yi v|| 2 -⟨y ⊤ α,i D yi v⟩ α ) - 1 τ yi 1 N (⟨h 2 α ⟩ α b ⊤ i D yi v + ⟨h α ⟩ α ||b i || 2 -⟨h α b ⊤ i y * α,i ⟩ α ) d dt b ⊤ i y * β,i = - 1 τ bi 1 N (b ⊤ i y * β,i + ⟨h α ⟩ α y ⊤ β,i D yi v -y ⊤ β,i ⟨y * α,i ⟩ α ) d dt ||b i || 2 = -2 1 τ bi 1 N (||b i || 2 + ⟨h α ⟩ α b ⊤ i D yi v -⟨b ⊤ i y * α,i ⟩ α ) d dt ||D yi v|| 2 = -2 1 τ yi 1 N (⟨h 2 α ⟩ α ||D yi v|| 2 + ⟨h α ⟩ α b ⊤ i D yi v -⟨h α y ⊤ α,i D yi v⟩ α ) d dt y * ⊤ β,i D yi v = - 1 τ yi 1 N (⟨h 2 α ⟩ α y * ⊤ β,i D yi v + ⟨h α ⟩ α y * ⊤ β,i b i -⟨h α y * ⊤ β,i y * α,i ⟩ α ) .(41)
If the output effective learning rates are again all equal, i.e. ∀ τ bi = τ b , ∀ τ yi = τ y , this system can be reduced to 9 scalars:
d dt h α = - 1 τ hα 1 2 (⟨b ⊤ i D yi v⟩ i + h α ⟨||D yi v|| 2 ⟩ i -⟨y ⊤ α,i D yi v⟩ i ) d dt ⟨b ⊤ i D yi v⟩ i = - 1 τ b 1 N (⟨b ⊤ i D yi v⟩ i + ⟨h α ⟩ α ⟨||D yi v|| 2 ⟩ i -⟨y ⊤ α,i D yi v⟩ α,i ) - 1 τ y 1 N (⟨h 2 α ⟩ α ⟨b ⊤ i D yi v⟩ i + ⟨h α ⟩ α ⟨||b i || 2 ⟩ i -⟨h α ⟨b ⊤ i y * α,i ⟩ i ⟩ α ) d dt ⟨b ⊤ i y * β,i ⟩ i = - 1 τ b 1 N (⟨b ⊤ i y * β,i ⟩ i + ⟨h α ⟩ α ⟨y ⊤ β,i D yi v⟩ i -⟨y ⊤ β,i y * α,i ⟩ α,i ) d dt ⟨||b i || 2 ⟩ i = -2 1 τ b 1 N (⟨||b i || 2 ⟩ i + ⟨h α ⟩ α ⟨b ⊤ i D yi v⟩ i -⟨b ⊤ i y * α,i ⟩ α,i ) d dt ⟨||D yi v|| 2 ⟩ i = -2 1 τ y 1 N (⟨h 2 α ⟩ α ⟨||D yi v|| 2 ⟩ i + ⟨h α ⟩ α ⟨b ⊤ i D yi v⟩ i -⟨h α ⟨y ⊤ α,i D yi v⟩ i ⟩ α ) d dt ⟨y * ⊤ β,i D yi v⟩ i = - 1 τ y 1 N (⟨h 2 α ⟩ α ⟨y * ⊤ β,i D yi v⟩ i + ⟨h α ⟩ α ⟨y * ⊤ β,i b i ⟩ i -⟨h α ⟨y * ⊤ β,i y * α,i ⟩ i ⟩ α ) .(42)
Training loss The loss can be written expressed using the variables from this system
L = 1 2 (⟨||b i || 2 ⟩ i + ⟨h 2 α ⟩ α ⟨||D yi v|| 2 ⟩ i + ⟨||y * α,i || 2 ⟩ α,i + 2⟨h α ⟩ α ⟨b ⊤ i D yi v⟩ i -2⟨h α ⟨y * ⊤ α,i D yi v⟩ i ⟩ α -2⟨⟨y * ⊤ α,i b i ⟩ i ⟩ α ). (43
)
Equal hidden effective learning rates When the pairs agree on all outputs,s we have
⟨y * ⊤ 1,i y * α,i ⟩ α,i = ⟨y * ⊤ 2,i y * α,i ⟩ α,i(44)
Assuming no correlation at initialization due to random weights
⟨b ⊤ i D yi v⟩ i (0) = 0 ⟨b ⊤ i y * β,i ⟩ i (0) = 0 ⟨y * ⊤ β,i D yi v⟩ i (0) = 0 ,(45)
we find the relations
d dt ⟨b ⊤ i y * 1,i ⟩ i = d dt ⟨b ⊤ i y * 2,i ⟩ i d dt ⟨y * ⊤ 1,i D yi v⟩ i = d dt ⟨y * ⊤ 2,i D yi v⟩ i ,(46)
such that
⟨b ⊤ i y * 1,i ⟩ i = ⟨b ⊤ i y * 2,i ⟩ i ⟨y * ⊤ 1,i D yi v⟩ i = ⟨y * ⊤ 2,i D yi v⟩ i .(47)
From this it follows that
d dt (τ 2 h 2 -τ 1 h 1 ) = - 1 2 ((h 2 -h 1 )⟨||D yi v|| 2 ⟩ i ).(48)
In the case of equal effective hidden learning rates τ h1 = τ h2 , the representational distance can only increase
d dt ||h 2 -h 1 || 2 = -⟨||D yi v|| 2 ⟩ i ,(49)
so no initial divergence occurs for agreeing pairs in this case.
this section cite: []

Section: C. Experimental Details
For all experiments, the PyTorch library was used to train the models. The code we used can be found at https:  //github.com/loekvanrossem/rnn_structure.git.
this section cite: []

Section: C.1. Streaming Parity Task
Model The neural network architecture consists of a single fully connected recurrent layer with 100 hidden units, ReLU activation function, along with a fully connected linear output layer, both with bias:
h t = ReLU(x t W ⊤ ih + b ih + h t-1 W ⊤ hh + b hh ) y t = W ⊤ hy h t + b hy(50)
The initial hidden vector h 0 was set to all ones, to reduce time being stuck at the zero fixed point early in training.
Training procedure The model was trained on all sequences up to length 10. We used 100 randomly selected sequences of length 50 to compute the validation loss during training. Both the inputs and outputs were embedded in a 2-dimensional space using a one-hot encoding.
Optimization was performed using stochastic gradient descent on the mean squared training loss. The model was trained over 1000 epochs at a learning rate of 0.02 and a batch size of 128. No momentum or regularization was used. Weights were initialized small using Xavier initialization at a gain of 0.1. Bias was initialized at zero.
Automata Automata were extracted using the complete training set. The merger cutoff was set at a factor of 0.01 of the representational standard deviation.
this section cite: []

Section: C.2. Grokking
For the re-implementation of the grokking experiment, we made use of the following publicly available repository: https://github.com/Sea-Snell/grokking.git
The data used is subtraction modulo 96, with a fraction of 0.6 of possible examples set aside for validation.
Every 250 epochs we evaluated the training loss, validation loss, number of states in the attention, and number of states in the hidden layer.
To compute the number of states in the hidden activations, we considered the activations after the normalization layer, as suggested in (Adriaensen & Maene, 2024). Next, we applied a principal component analysis to reduce the space to 512 dimensions and merged pairs with distance below a threshold to count the number of states. The number of states in the attention matrix was computed in a similar manner.
The threshold was chosen to be 1.25 times the initial maximum distance, as states are expected to be close at initialization. Any significantly larger or smaller choice for the threshold resulted in the counting of a single or all possible states at every epoch.
this section cite: ['b1']

Section: D. Supplemental Figures

this section cite: []

Section: D.1. No Representational Contraction
One may wonder if the reduction in the number of automaton states during the second phase of learning is merely the result of the whole representational space contracting, as the merger threshold is a fixed constant. To make sure this is not the case, we computed the representational mean distance over time and found it always increases (Figure 12).
this section cite: []

Section: D.2. Validity Linear Approximation
The theoretical interaction model assumes close enough representations such that the first order term in the Taylor expansion dominates higher order terms. One may wonder whether or not representations in practice are ever close enough for this to be a reasonable assumption, especially since even pairs that end up merging still initially move apart from each other as can be seen in Figure 10. To test this we compute the average ratio of the norm of the second-order Taylor term with respect to
this section cite: []

Section: Epochs Mean distance
Figure 12. The mean pairwise representational distance of training set during training on the streaming parity task. It never decreases during training, hence state mergers cannot be entirely explained by representational contraction.
the first-order term (Figure 13). We find that although the linear approximation does get worse when representations start to move apart, the first-order term still remains dominant, and in the case of merging pairs by at least a factor of 10.
this section cite: []

Section: Mean converging pairs
Mean diverging pairs Epochs Ratio second to first order term Figure 13. The norms of the second-order term divided by the first-order term in the Taylor expansion of the output map during training, averaged over all pairs that ended up below the merging threshold and all pairs that ended up above the merging threshold out of 100 randomly selected pairs. Note that this ratio increases around the point where pairs start to diverge, however, the first order term remains dominant on average throughout the entire training procedure.
this section cite: []

Section: D.3. Diverging Mergers Occurrence Rates
As gradient descent scales with the number of parameters, the effective learning rates of representations should be proportional to the sequence lengths.
1 τ h ∝ n (51
)
This is because for a hidden state corresponding to a sequence of length n, the map assigning that hidden state consists of the recurrent map applied n times. Therefore, this map has nP parameters, where P is the number of parameters on the recurrent map, and parameters that appear multiple times are double counted. Applying gradient descent will thus result in a sum of nP terms, and this increases linearly with n.
From Equation ( 51) we would expect such mergers that initially diverge to be more prevalent among pairs with a large difference in sequence lengths. Such a pattern was indeed observed in the experiment (Figure 14).
this section cite: []

Section: D.4. State Change Patterns Due to Diverging Merger
According to the theory, the reason the number of states initially increases before decreasing is that pairs of datapoints agreeing on all outputs will first move apart before they end up merging. As we can see Figure 15, the decrease of the number of states after the initial increase can indeed be entirely explained by merging pairs that initially diverged.
this section cite: []

Section: Epochs

this section cite: []

Section: Loss

this section cite: []

Section: Experiment
Total number of splits
Training loss Number of agreeing splits Number of disagreeing splits
Figure 15. The number of pairs of datapoints whose representational distance are apart more than the merging threshold during training on the streaming parity task, divided by whether or not they agree on all possible future outputs. We see that the drop in number of automaton states can be explained by pairs agreeing on all outputs initially diverging before they end up merging.
this section cite: []

Section: D.5. Random Regular Tasks
We ran multiple experiments on randomly generated regular tasks with a number of states between 1 and 7. Tasks were defined using random automata, which were generated by starting with an initial state and adding transitions for each input symbol to either a new state with some probability which we set to 0.75, or a uniformly selected already existing state. This procedure was iterated for each state until all transitions were assigned, or 7 states exist at which point all remaining transitions would be assigned to already existing states. Results were found to be consistently qualitatively similar to the streaming parity task. Initial weight gain was set to 0.025, the learning rate to 0.05. An example is shown in Figure 16.
Other samples of regular tasks provided qualitatively similar results.
this section cite: []

Section: D.6. Hyperbolic Tangent Activation Function
We reran the experiments on the same recurrent neural network architecture except with hyperbolic tangent non-linearity. Initial weight gain was set to 0.01 and learning rate to 0.05. All other settings remained the same. Similar qualitative results
this section cite: []

Section: A. Automaton Extraction

this section cite: []

Section: A.1. Definition Deterministic Finite Automaton
Formally a deterministic finite automaton (Ashby, 1956) is a tuple (Q, Σ, δ, q 0 , F ), consisting of 1. A finite set of states Q 2. A finite set of possible input symbols Σ, called the alphabet 3. A transition function δ : Q × Σ → Q 4. An initial state q 0 5. A subset of accepting states F Given some string of input symbols x = x (1) x (2) . . . x (n) the automaton is said to accept the string x when there exists a sequence of states r (0) , . . . , r (n) ∈ Q such that 1. r (0)
= q 0 2. ∀ i r (i+1) = δ(r (i) , x (i+1) ) 3. r (n) ∈ F
In the context of the streaming parity task we can take the subset of accepting states to be precisely those for which the model predicts an output 1.
this section cite: ['b3']

Section: A.2. Extraction Algorithm
In order to extract one from a recurrent neural network, we define the state corresponding to an input string x as the hidden representation in the network after it received that string, i.e.
q 0 := h 0 q xn...x1 := f h (q xn-1...x1 , x n ) .(11)
When two states are on top of each other, i.e. the representations are within some small distance ϵ, we will consider them as the same state. Given an evaluation dataset X of input sequences, the set of states can be defined as
Q := {q x |x ∈ X}/(q ∼ q ′ ⇐⇒ ||q -q ′ || < ϵ).(12)
The alphabet Σ is the set of all possible input symbols in the data. The transition function δ(q, σ) is given by the state corresponding to f h (q, σ). The set of accepting states F are all states for which f y (q) is closest to the output 1. If we are considering a task with more than two possible output symbols, we can generalize this definition using a Moore machine, which replaces the set of accepting states with an output function.
this section cite: []

Section: A.3. State Reduction Algorithm
To help interpret the final learning automaton within the recurrent neural network, we can reduce it to a minimal state automaton with equivalent outputs. For this we use Hopcroft's Algorithm (Hopcroft, 1971) (see Algorithm 1), which returns the unique smallest DFA, equivalent to the provided DFA. Essentially what this algorithm does is it merges all pairs of states which are indistinguishable for any possible input string.
were found as with the ReLU network (Figure 17).
this section cite: ['b23']

Section: D.7. Modular Addition
The single-layer transformer trained on modular subtraction has a much smoother transition to generalization than the RNN trained on parity. Modular addition tends to have a sharper transition. To see if this changes any of the results we run the experiment again. We still find similar merging patterns (Figure 18).
this section cite: []

Section: D.8. State Merging in the Value Matrix
Although less clear, we found something resembling a state merging effect in the value matrix of the transformer (Figure 19).
The key matrix did not show such an effect.
this section cite: []

Section: D.9. Discontinuity
We can break the assumption of continuity by adding a step discontinuity in the output map of the RNN.
f (x) = 0 if x < 0 x + 1 if x > 0 (52
)
We still find similar results albeit with nosier dynamics (Figure 20).
this section cite: []

Section: References
Ref_id:b0 Title: Generalization on the Unseen, Logic Reasoning and Degree Curriculum Year: (2023-07)
Ref_id:b1 Title: Extracting Finite State Machines from Transformers Year: (2024-10)
Ref_id:b2 Title: Exploring Length Generalization in Large Language Models Year: (2022-11)
Ref_id:b3 Title: Automata Studies: Annals of Mathematics Studies Year: (1956-04)
Ref_id:b4 Title: The Onset of Variance-Limited Behavior for Networks in the Lazy and Rich Regimes Year: (2022-12)
Ref_id:b5 Title: One dimensional approximations of neuronal dynamics reveal computational strategy Year: (2023-01)
Ref_id:b6 Title: Language Models are Few-Shot Learners Year: (2020-07)
Ref_id:b7 Title: Gating creates slow modes and controls phase-space complexity in GRUs and LSTMs Year: (2020-08)
Ref_id:b8 Title: Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss Year: (2020-07)
Ref_id:b9 Title: On Lazy Training in Differentiable Programming Year: (2020-01)
Ref_id:b10 Title: Learning Low Dimensional State Spaces with Overparameterized Recurrent Neural Nets Year: (2023-03)
Ref_id:b11 Title: Approximation with Artificial Neural Networks Year: (2001-06)
Ref_id:b12 Title: Revisiting Transformer-based Models for Long Document Classification Year: (2022-10)
Ref_id:b13 Title: Representation mitosis in wide neural networks Year: (2021-10)
Ref_id:b14 Title: Flexible multitask computation in recurrent networks utilizes shared dynamical motifs Year: (2024-07)
Ref_id:b15 Title:  Year: ()
Ref_id:b16 Title: A cellular basis for mapping behavioural structure Year: (2024)
Ref_id:b17 Title: Rich and lazy learning of task representations in brains and neural networks Year: (2021-04)
Ref_id:b18 Title: The emergence of clusters in self-attention dynamics Year: (2023-12)
Ref_id:b19 Title: Learning and Extracting Finite State Automata with Second-Order Recurrent Neural Networks Year: (1992-05)
Ref_id:b20 Title: Multimodal Neurons in Artificial Neural Networks Year: (2021-03)
Ref_id:b21 Title: Implicit Bias of Gradient Descent on Linear Convolutional Networks Year: (2018)
Ref_id:b22 Title: Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems Year: (2024-06)
Ref_id:b23 Title: An n log n algorithm for minimizing states in a finite automaton Year: (1971-01)
Ref_id:b24 Title: Multilayer feedforward networks are universal approximators Year: (1989-01)
Ref_id:b25 Title: Deep Networks Always Grok and Here is Why Year: (2024-07)
Ref_id:b26 Title: Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks Year: (2018-07)
Ref_id:b27 Title: A recurrent neural network without chaos Year: (2016-12)
Ref_id:b28 Title: The Topology and Geometry of Neural Representations Year: (2023-09)
Ref_id:b29 Title: Visualizing Representational Dynamics with Multidimensional Scaling Alignment Year: (2019-07)
Ref_id:b30 Title: Towards Understanding Grokking: An Effective Theory of Representation Learning, October 2022 Year: ()
Ref_id:b31 Title: Rearranging the Familiar: Testing Compositional Generalization in Recurrent Networks Year: (2018-07)
Ref_id:b32 Title: Universality and individuality in neural dynamics across large populations of recurrent networks Year: (2019-12)
Ref_id:b33 Title: History information emerges in the cortex during learning Year: (2023-11)
Ref_id:b34 Title: Extracting Finite Automata from RNNs Using State Merging Year: (2022-04)
Ref_id:b35 Title: Opening the AI black box: program synthesis via mechanistic interpretability, February 2024 Year: ()
Ref_id:b36 Title: History-dependent variability in population dynamics during evidence accumulation in cortex Year: (2016-12)
Ref_id:b37 Title: Progress measures for grokking via mechanistic interpretability Year: (2023-10)
Ref_id:b38 Title: Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning Year: (2015-04)
Ref_id:b39 Title: An Introduction to Circuits Year: (2020)
Ref_id:b40 Title: What Is the Other 85 Percent of V1 Doing? Year: ()
Ref_id:b41 Title:  Year: (2006-01)
Ref_id:b42 Title: Constructing deterministic finite-state automata in recurrent neural networks Year: (1996)
Ref_id:b43 Title: Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets Year: (2022-01)
Ref_id:b44 Title: A deep learning framework for neuroscience Year: (2019-11)
Ref_id:b45 Title: Aligned and oblique dynamics in recurrent neural networks Year: (2024-08)
Ref_id:b46 Title: Encoding Sequential Structure in Simple Recurrent Networks Year: (1989-07)
Ref_id:b47 Title: Opening the Black Box of Deep Neural Networks via Information Year: (2017-04)
Ref_id:b48 Title: The Implicit Bias of Gradient Descent on Separable Data Year: (2022-07)
Ref_id:b49 Title: Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks Year: (2013-03)
Ref_id:b50 Title: Place cells and silent cells in the hippocampus of freely-behaving rats Year: (1989-07)
Ref_id:b51 Title: Finite State Machines and Recurrent Neural Networks -Automata and Dynamical Systems Approaches Year: (1999-03)
Ref_id:b52 Title: Charting and navigating the space of solutions for recurrent neural networks Year: (2021-11)
Ref_id:b53 Title: When Representations Align: Universality in Representation Learning Dynamics Year: (2024-07)
Ref_id:b54 Title: Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, November 2022 Year: ()
Ref_id:b55 Title: Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples Year: (2020-02)
Ref_id:b56 Title: Generalized Shape Metrics on Neural Representations Year: (2022-01)
Ref_id:b57 Title: The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks Year: (2023-11)
Ref_id:b58 Title: Evolving schema representations in orbitofrontal ensembles during learning Year: (2021-02)
Ref_id:b59 Title:  Year: ()
Ref_id:b60 Title: Toward Understanding the Importance of Noise in Training Neural Networks Year: (2019-05)
Ref_id:b61 Title: Transformers Can Achieve Length Generalization But Not Robustly Year: (2024-02)
Ref_id:b62 Title: Formation of Representations in Neural Networks, February 2025 Year: ()
