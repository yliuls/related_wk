Title: ON THE REASONING ABILITIES OF MASKED DIFFU-SION LANGUAGE MODELS
Abstract: Masked diffusion models (MDMs) for text offer a compelling alternative to traditional autoregressive language models. Parallel generation makes them efficient, but their computational capabilities and the limitations inherent in their parallelism remain largely unexplored. To this end, we characterize what types of reasoning problems MDMs can provably solve and how efficiently. We do this by connecting MDMs to the well-understood reasoning frameworks of chain of thought (CoT) and padded looped transformers (PLTs) in the finite-precision log-width setting: We show that MDMs and polynomially-padded PLTs are, in fact, equivalent in this setting, and that MDMs can solve all problems that CoT-augmented transformers can. Moreover, we showcase classes of problems (including regular languages) for which MDMs are inherently more efficient than CoT transformers, where parallel generation allows for substantially faster reasoning.

Section: INTRODUCTION
Many complex problems can be decomposed into smaller, independent sub-problems, making them naturally suited for parallel computation. For example, we can compute the value of a mathematical expression by evaluating its sub-expressions independently and combining the results (see Fig. 3). However, dominant autoregressive language models (LMs) tackle these problems sequentially. Methods like chain of thought (CoT), for instance, generate solutions one step at a time, failing to capitalize on the underlying parallel structure. Parallel generation by masked diffusion models (MDMs) offers a compelling alternative. Recent advances have positioned MDMs as a viable contender to autoregressive LMs in language modeling, code generation, and even molecule design (Lou et al., 2024;Zhang et al., 2025;Sun et al., 2025). However, the fundamental reasoning capabilities of MDMs remain poorly understood, which limits the extent to which we can leverage their potential and apply them to appropriate tasks. This work bridges that gap by providing the first formal characterization of the expressivity of MDMs, clarifying their fundamental computational strengths and weaknesses.
this section cite: ['b33', 'b61', 'b49']

Section: Masked Diffusion Models
Padded Looped Transformers Chain of Thought Added stochasticity
this section cite: []

Section: Thm. 3.1
Extra output space
this section cite: []

Section: Thm. 3.1

this section cite: []

Section: Extra steps

this section cite: []

Section: Thm. 3.4
Extra output space
this section cite: []

Section: Thm. 3.3
Sequentiality Bottleneck under log N steps.
Cor. 3.7 Our analysis builds upon the study of LM expressivity-the formal characterization of the problems whose solution the neural architecture of an LM (with appropriate parameters) can express. While this field comprehensively describes autoregressive LMs, its findings do not directly apply to MDMs due to their fundamentally different, non-sequential processing of text, leaving the theoretical studies of these two paradigms largely disconnected. Prior theoretical work on MDMs has focused on the limitations of their factorized backward process and its convergence properties. This research has shown that while MDMs can approximate n-gram LMs with constantly many denoising steps, the number of steps must grow linearly with the string length even for simple LMs such as probabilistic regular languages (Feng et al., 2025;Li & Cai, 2025). 1 However, the asymptotic nature of these findings, together with the strict assumptions on the theoretical model, makes it difficult to draw concrete conclusions about the practical reasoning capabilities of MDMs, leaving a critical gap: A theoretical framework for MDMs that is both (1) formally rigorous enough to provide a comprehensive picture of how MDMs can use the combination of parallelism and iterative refinement for formal reasoning, and (2) faithful to how they are implemented in practice. Our work introduces such a framework, providing a tight characterization of their reasoning capabilities.
Concretely, we connect MDMs implemented as finite-precision transformers with logarithmicallygrowing model width to known reasoning paradigms of CoT (Wei et al., 2022), looping (Dehghani et al., 2019), and pause tokens (Lanham et al., 2023). For example, we formalize:
Takeaway 1 (Thms. 3.3 and 3.4). MDMs can perform CoT reasoning with some overhead and the MDM denoising process can be (inefficiently) simulated by CoT by generating one symbol at a time.
We also show how MDMs can solve complex problems by solving easier sub-problems in parallel, which makes them inherently more efficient than CoT on parallelizable problems.
Takeaway 2 (Cor. 3.7). MDMs are provably more efficient at parallelizable problems than CoT.
We refer to the fact that CoT cannot take advantage of this parallelism as the sequentiality bottleneck of CoT. Takeaway 2 highlights the potential efficiency gains of parallel CoT that generates multiple symbols at once. Our analysis, in fact, identifies a tighter and more natural connection between MDMs and this variant, which we refer to as pCoT. The parallelism of MDMs also facilitates a close connection with looped and padded transformers, where looping naturally maps to denoising steps and padding tokens to the generated tokens of an MDM. We find that the class of problems solvable by MDMs is, in fact, precisely equivalent to that solvable by padded looped transformers. Takeaway 3 (Thm. 3.1). MDMs are equivalent to PLTs.
These connections allow us to leverage known characterizations of CoT and PLTs together with classical complexity theory results to understand the fundamental capabilities and limits of MDMs (Li et al., 2024;Saunshi et al., 2025;London & Kanade, 2025;Svete et al., 2025). For example, with N representing the length of the input string and with AC d for d P N being the standard class of Boolean circuits with AND, OR, and NOT gates and depth Oplog d N q, we obtain: Takeaway 4 (Thm. 3.2). MDMs with log N denoising steps can recognize regular languages.
Takeaway 5 (Cors. 3.3 and 3.4). For d P N, MDMs with Oplog d N q denoising steps and polypN q output space are equivalent to AC d . As d Ñ 8, this yields NC, the class of all parallelizable problems. With constantly many denoising steps (d " 0), MDMs are equivalent to the limited class AC 0 , i.e., they are no more powerful than standard transformers and, e.g., cannot recognize all regular languages.
We summarize these takeaways and their relationships in Figs. 1 and 2. Our proofs emphasize the affordances and difficulties in solving reasoning problems with MDMs. We find, for example, that the discrete nature of the generated text, which serves as a communication channel between individual denoising steps, limits the amount of information that can be passed between steps. This necessitates extra output space to store the intermediate computations and is analogous to the discrepancy between fixed-precision and log-precision transformers (Li et al., 2024), and the difference between the classes AC and TC, the class of circuits with threshold gates. We find that positional encodings that carry information not computable by transformers themselves are crucial in locating this information. We also observe that, while unmasked attention makes MDM attention patterns flexible, it complicates left-to-right processing, which is often natural in human language-we find that exact implementation of causal masking requires quadratically more output space.
this section cite: ['b25', 'b56', 'b12', 'b23', 'b27', 'b46', 'b32', 'b52', 'b27']

Section: PRELIMINARIES
This section introduces the preliminaries and notation used throughout the paper. We reserve the main text for the high-level intuitions and defer technical details to App. B.
this section cite: []

Section: TRANSFORMERS
We analyze (un)masked transformers for string generation and classification. We present the full model in App. B.4 and focus here on the most relevant aspects.
Transformer families. We study finite precision transformers where the value of each parameter and activation is represented with a fixed number of bits. We allow the model width, i.e., the size of the contextual representations, to grow logarithmically with the input length N . This is a standard assumption in the literature on transformer expressivity (Li et al., 2024) since it is necessary and sufficient for the model to uniquely identify input positions, and aligns with modern implementations of quantized but wide transformers. The growing width results in a separate transformer T N for each N , yielding a family tT N u N PN of transformers. We enforce L-uniformity in the family by requiring an associated Turing machine that constructs T N in Oplog N q space (cf. App. B.4).
(Parallel) CoT transformers. CoT reasoning enables sequential processing by solving problems in multiple steps (Wei et al., 2022). It is an integral part of today's popular "reasoning" models and substantially increases transformers' expressivity (Li et al., 2024;Merrill & Sabharwal, 2024). We define our idealization of CoT transformers in App. B.4, including parallel CoT transformers, which predict P 1 P N symbols in parallel at each time step, enabling some parallelism if the task allows it. 2We denote the classes of CoT and parallel CoT transformers by CoT and pCoT, respectively.
Padded looped transformers (PLTs). Looped transformers repeatedly apply a fixed block of transformer layers to the input (Dehghani et al., 2019). This dynamically increases the depth of the model, enabling more complex reasoning, and does not increase the model size, as the same blocks are reused, thus reducing the memory footprint and computational cost (Bae et al., 2025). Such reasoning steps include both sequential and parallel processing, resulting in both efficiency as well as depth of the reasoning process. Padded transformers additionally pad the input with blank symbols, which can be used to perform additional computations in parallel. This additional padding space is analogous to increasing the circuit width in circuit complexity. We additionally provide PLTs with external noise applied to the residual stream at each step, which enables stochastic computations (cf. App. B.6). We denote the class of padded looped transformers by PLT.
this section cite: ['b27', 'b56', 'b27', 'b12', 'b2']

Section: Transformer LMs.
An alphabet Σ is a finite, non-empty set of symbols. Its Kleene closure is Σ ˚def " Ť 8 n"0 Σ n , the set of all strings. A language model is a distribution over Σ ˚. Most LMs are autoregressive-they define next-symbol
distributions ñ p p¨| wq over Σ def " Σ Y tEOSu for w P Σ ˚,
where EOS R Σ is the end-of-string symbol. A transformer-based LM computes ñ p p¨| wq by linearly transforming the contextual representation of the final symbol to the logits of a distribution over Σ. Moreover, contextual representations can be used for infilling-predicting symbols at masked positions. Infilling probabilities p Ó p¨| wq at masked positions in w P Σ m ˚are distributions over Σ, where m R Σ is the mask symbol and Σ m def " Σ Y tmu.
Transformers and formal languages. Plenty of work describes transformer capabilities and limitations with formal languages (Strobl et al., 2024). These studies typically frame transformers as language recognizers, i.e., classifiers that decide whether a string w P Σ ˚belongs to some formal language L Ď Σ ˚ (Butoi et al., 2025). String membership is usually deterministic and can be formalized by determinizing the LM defined by a transformer: The next-symbol and infilling probabilities are used to decode the most probable symbol or decision. 3 The final prediction of 1 tw P Lu P t1, 0u can be made (1) in a single pass by classifying based on the contextual representation of a particular symbol in the string, analogous to classifying based on the CLS symbol in BERT (Devlin et al., 2019), or (2) after "reasoning", i.e., solving the problem in multiple time steps. In this case, the transformer's prediction is only made after a sequence of intermediate predictions that augment its computation and help the final decision. This is analogous to using CoT reasoning for string recognition and can be thought of as simulating a Turing machine with each step of the CoT process.
Particularly fruitful has been the study of transformers as Boolean circuits. In particular, our idealization of transformers falls under AC 0 circuits (Li et al., 2024)-Boolean circuits of constant depth, polynomial size, and with AND, OR, and NOT gates of unbounded fan-in-and captures the entire class if padding is allowed (London & Kanade, 2025). Other idealizations of transformers can compute functions outside of AC 0 (Li et al., 2024;Merrill & Sabharwal, 2025a) but remain in TC 0 , the class of threshold circuits which add threshold gates (which determine whether the number of inputs exceeds some threshold) to AC 0 circuits. Apps. B.2 and B.4 provide more details on circuit classes and their relation to transformers.
this section cite: ['b48', 'b6', 'b13', 'b27', 'b32', 'b27']

Section: MASKED DIFFUSION LANGUAGE MODELS
Discrete diffusion LMs define a distribution over Σ ˚by progressively denoising noisy strings sampled from some fixed distribution. Formally, they define a forward (noising) process and a reverse (denoising) process. The forward process defines a Markov chain over strings that iteratively corrupts them. Common examples include replacing symbols uniformly at random (uniform diffusion) or masking them with the mask symbol m (masked diffusion models, MDMs). The latter is the focus of this work. In this setting, the forward process starts from an initial string w p0q P Σ ˚of some pre-determined length P and, at each of the T (discrete) steps, independently masks symbols with probability determined by a masking schedule α `t T ˘P r0, 1s:
q t|0 pw ptq | w p0q q " N ź n"1 q t|0 pw ptq n | w p0q n q, q t|0 pw ptq n | w p0q n q " # 1 ´αp t T q, if w ptq n " m αp t T q, otherwise.
The masking schedule is set such that αp0q " 1 (no masking at the start) and αp1q " 0 (fully masked at the end, meaning that the noise distribution is the Dirac delta on the fully masked string).
Starting from the fully masked input, the reverse process q 0|T inverts the forward process q T |0 by (1) uniformly selecting some positions to unmask, and (2) sampling the chosen unmasked symbols. After T denoising steps, q 0|T produces a string w p0q sampled from the LM defined by the diffusion process. It is this reverse process that is learned from data. Its analytical form is generally intractable, so one usually models a parameterized approximation of a single denoising step p q t´1|t pw pt´1q | w ptq q, typically implemented as a transformer, that factorizes across positions:
p q t´1|t pw pt´1q | w ptq q " N ź n"1 p q t´1|t pw pt´1q n | w ptq q(1)
Eq. ( 1) enables parallel generation but ignores inter-symbol dependencies at each denoising step.
Much of the existing work on MDM expressivity analyzes the convergence of p q t´1|t to q 0|T (Li & Cai, 2025;Chen & Ying, 2024;Feng et al., 2025). Studying convergence properties usually requires assuming uniform unmasking and a good approximation of the ground-truth model (e.g., Li & Cai, 2025;Chen & Ying, 2024;Feng et al., 2025;Liu, 2025). In words, Assumption 2.2 states that the transformer perfectly models all conditional distributions of the diffusion process. 4 While this seems necessary, the following observation, proved in App. E, shows that Assumptions 2.1 and 2.2 severely limit the class of functions that the model can compute.
Theorem 2.1. If Assumptions 2.1 and 2.2 hold for an LM p, p cannot compute non-AC 0 functions. 5,6   By assuming that the MDM is unable to choose which positions to unmask, the model has no choice in which sub-problems to solve first, which ignores the possibility of problem decomposition and requires the model to be equally good at solving any sub-problem-including predicting the final answer based on the input directly (with no reasoning steps). This implies that the problem is solvable in a single prediction step of a transformer. However, the expressivity of a single transformer pass is limited-Thm. 2.1 uses the fact that fixed-depth transformers lie in AC 0 (Li et al., 2024). However, not all conditional probabilities have to be known to be able to solve algorithms in few steps. Intuitively, by choosing to solve simple subproblems with non-random unmasking, an MDM can avoid the difficult parts. For example, given the current arithmetic expression, one only has to predict the next set of simplifications-which are simple functions of the current expression. This motivates us to loosen Assumptions 2.1 and 2.2, which we do in our idealization of an MDM.
this section cite: ['b25', 'b9', 'b25', 'b9', 'b30', 'b27']

Section: OUR IDEALIZATION OF MASKED DIFFUSION MODELS
We aim to understand the expressivity of the reverse process. To this end, we introduce an idealization that captures its key aspects-iterative unmasking and infilling-and provides a principled lens for understanding the expressivity of practical MDMs and comparing them to well-known paradigms such as CoT. Here, we describe the high-level ideas; see App. B.7 for the full formal model.
We formalize the reverse process with two components: A planner that decides which positions to unmask at each step and a predictor that samples the symbols at the unmasked positions. This loosens Assumption 2.1 and generalizes standard MDMs in which the planner is implicitly defined by choosing the positions to unmask uniformly at random. It also mirrors popular MDM implementations that generate text by selecting a subset of masked positions-for example, based on model confidence or according to a learned policy-and predicting the symbols conditioned on the current partially unmasked string (Ghazvininejad et al., 2019; Peng et al., 2025; Zheng et al., 2024; Liu et al., 2025a; Kim et al.,  2025; Ben-Hamu et al., 2025). 7 Discarding Assumption 2.1 also sidesteps limitations of the positionwise independence in Eq. ( 1), a restriction that prevents MDMs with uniform unmasking matching even simple distributions exactly (Feng et al., 2025;Wu et al., 2025). In reasoning problems with a deterministic sequential structure, the ability to decide what to unmask enables problem decomposition into a sequence of deterministic steps that can be solved in parallel. 8 To connect our analysis to practical implementations, we assume that the planner and predictor are implemented as transformers.
We allow the planner to choose to resample already unmasked positions. This overcomes another key limitation-the inability to revert decisions and correct earlier mistakes-a challenge that is the focus of much recent research (von Rütte et al., 2025;Song et al., 2025, inter alia). 9 While existing work focuses on reformulating the diffusion process to allow for resampling and refining the resulting training objectives, our idealized MDMs can be seen as a complementary approach that foregoes the complications of training and focuses on the expressivity of the generation process itself-it analyzes what is theoretically possible in a very targeted way when resampling is allowed.
We denote the class of our idealized MDMs by MDM. For a more detailed discussion, including the theoretical connection of our idealization to existing MDM variants, see App. C.
this section cite: ['b57', 'b55']

Section: THEORETICAL RESULTS
This section describes two complementary characterizations of MDM expressivity: One based on their connection to PLTs and the other based on their ability to perform sequential CoT reasoning. 10Notation. Let T " T pN q and P " P pN q be functions of N . In the following, CoTrT s refers to languages recognized by families tT N u N PN of CoT transformers with at most T steps. For C P tpCoT, MDM, PLTu, CrT, P s refers to languages recognized by families in C with at most T generation, denoising, or looping steps, respectively, and P total output or padding symbols. r
OpN q refers to big-O notation that ignores logarithmic factors, and polypN q to polynomial functions in N .
this section cite: []

Section: MDMS ARE EQUIVALENT TO PADDED LOOPED TRANSFORMERS
Intuitively, PLTs closely resemble MDMs: Both iteratively refine information in parallel-MDMs by unmasking and predicting discrete symbols, and PLTs by updating the residual stream. 11 Thm. 3.1 formalizes this, assuming PLTs are supplied with external sampling noise (cf. §2, §B.6) like MDMs. Theorem 3.1 (PLTs and MDMs). MDMrT, P s Ď PLTrT, P s (2a) and PLTrT, P s Ď MDMrT, pN `P qDs. (2b)
The simulation of a PLT by an MDM incurs a factor D increase in the required padding (cf. Eq. ( 2b)), where D is the model width of the PLT. In our setting where D " Oplog N q, this implies that the classes of finite-precision MDMs and PLTs coincide up to a logarithmic factor in the padding length: Corollary 3.1. For any K ě 1,
MDMrT, r OpN K qs " PLTrT, OpN K qs.(3)
The close connection between MDMs and PLTs allows us to leverage existing results about PLT expressivity to understand MDMs. Saunshi et al. (2025, Thm. 5.1), for example, show that log-depth unpadded transformers can recognize regular languages. Combined with Cor. 3.1, this implies:
Corollary 3.2. Regular languages are in MDMrlog N, N log N s.
In fact, we obtain a tighter bound with a more specialized construction. 12Theorem 3.2. Regular languages are in MDMrlog N, N s. London & Kanade (2025) show that polynomially padded finite-precision PLTs with constantly many steps are equivalent to L-uniform AC 0 , the class of AC 0 circuits that can be constructed by a logspace Turing machine (cf. App. B.2). We leverage this result together with Thm. 3.1 to characterize the expressivity of MDMs with constantly many denoising steps.
Corollary 3.3 (MDMs with constantly many denoising steps).
MDMrOp1q, polypN qs " PLTrOp1q, polypN qs " L-uniform AC 0 .
Allowing for a constant number of decoding steps therefore does not increase the expressivity of MDMs beyond the limited class AC 0 . This further corroborates the empirical observation that the number of denoising steps must scale with the input complexity and complements existing results on MDM expressivity as a function of the number of denoising steps (Li & Cai, 2025;Feng et al., 2025).
We can, however, increase expressivity with more denoising steps. Svete et al. (2025) show that finite-precision PLTs with Oplog d N q steps and polynomial padding are equivalent to L-uniform AC d , similar to the case of log-precision PLTs and L-uniform TC d (Merrill & Sabharwal, 2025a). Thus:
Corollary 3.4 (MDMs with polylogarithmically many denoising steps).
MDMrOplog d N q, polypN qs " PLTrOplog d N q, polypN qs " L-uniform AC d(5)
In particular, since NC d Ď AC d for d P N (where NC d denotes AC d circuits with bounded fan-in) (Vollmer, 1999), we get that with polylogarithmic looping and polynomial padding, MDMs converge to NC, the class of all parallelizable problems. Cor. 3.4 also implies that regular languages are in MDMrlog N, polypN qs; Thm. 3.2 provides a more efficient construction with a linear output space. Moreover, Cor. 3.4 implies that L-uniform NC 1 Ď MDMrlog N, polypN qs.
this section cite: ['b32', 'b25', 'b52', 'b54']

Section: MDMS AND COT CAN (INEFFICIENTLY) SIMULATE EACH OTHER
While the close connection between MDMs and PLTs provides a useful lens to analyze MDMs in terms of known complexity classes, the lack of practical PLT implementations makes it difficult to draw intuitive conclusions. We therefore complement §3.1 by connecting MDMs and the more popular CoT paradigm, and consider how MDM can behave "autoregressively" like CoT. The intuition is simple: An MDM can simulate CoT by unmasking one symbol at a time, effectively mimicking the sequential generation. More precisely, we connect MDMs to pCoT since the latter's ability to generate multiple symbols at once naturally maps to MDMs' parallel generation. In particular:
Theorem 3.3 (MDMs can simulate pCoT transformers). pCoTrT, P s Ď MDMrT, P `pN `P q 2 s (6
)
The simulation incurs a quadratic blow-up in the padding length. This is not due to an inherent feature of the diffusion process but rather the challenge of simulating masked attention with unmasked one (cf. Lem. D.14). 13 In particular, if the MDM transformer is causally masked, the blow-up disappears. Moreover, if the unmasked transformer can simulate masking more efficiently (with, for example, more expressive scoring functions), the blow-up can be alleviated. 14 While Lem. D.14 could possibly be improved, it is interesting to note that the seemingly more general, unmasked nature of MDMs might negatively impact their ability to align with human-oriented sequential processing captured by causal masking, which could provide a useful inductive bias for the masked models. 15   The other direction of Thm. 3.3 shows that pCoT transformers can simulate MDMs. Theorem 3.4 (pCoT transformers can simulate MDMs).
MDMrT, P s Ď pCoTrT, LT pP `N qs,(7)
where L is the number of layers in the transformer implementing the MDM.
Again, the factor L comes from the need to simulate unmasked attention in MDMs with causally masked transformers implementing pCoT (cf. Lem. D.13). However, here, the blow-up is only linear. The additional factor of T comes from the pCoT having to write out every padding token after each denoising step, as the MDM can unmask tokens in an arbitrary order.
The results above can be summarized by the following sequence of inclusions. Corollary 3.5. We have the following set of inclusions:
CoTrT s " pCoTrT, T s (8a, Prop. B.1) Ď MDMrT, T `pN `T q 2 s (8b, Thm. 3.3) Ď pCoTrT, LT pN `T `pN `T q 2 qs (8c, Thm. 3.4) Ď CoTrLT pN `T `pN `T q 2 qs (8d, Prop. B.1) Ď CoTrLT pN `T `1q 2 s.(8e)
In particular, when T ě N , we have CoTrT s Ď MDMrT, OpT 2 qs Ď pCoTrT, OpT 3 qs Ď CoTrOpT 3 qs.
this section cite: []

Section: Cor. 3.5 lower-and upper-bounds MDM expressivity based on the expressivity of CoT transformers.
For example, MDM with polynomially many denoising steps remain within the class P, the problems solvable in polynomial time by a non-random-access multitape Turing machine. This follows from the equivalence of CoT transformers with polynomially many steps to P (Li et al., 2024). Corollary 3.6 (MDMs with polynomially many denoising steps). For any K ě 1, we have that
MDMrT, N K s Ď CoTrOpT N K qs,(9)
meaning that MDMs with polynomially many denoising steps remain in P:
MDMrpolypN q, polypN qs Ď CoTrpolypN qs Ď P.(10)
this section cite: ['b27']

Section: A SEPARATION BETWEEN MDMS AND COT TRANSFORMERS
Merrill & Sabharwal (2024); Li et al. (2024) show that CoT transformers with logarithmically many decoding steps remain in TC 0 . Combining this with Thm. 3.2, the widely accepted assumption that TC 0 ‰ NC 1 , and known NC 1 -completeness of specific regular languages, we obtain the following separation in expressivity under a small (logarithmic) number of decoding steps. We term the inability of CoT to leverage parallelism the sequentiality bottleneck of CoT.
Corollary 3.7 (A strict separation in efficient reasoning abilities of MDMs and CoT transformers).
CoTrlog N s Ĺ MDMrlog N, N s.(11)
Concretely, MDMrlog N, N s z CoTrlog N s, for example, contains all NC 1 -complete regular languages.
this section cite: ['b27']

Section: DISCUSSION
Strengths and weaknesses of MDMs. §3 provides insights into the suitability of using MDMs for different classes of problems. On the one hand, Cor. 3.7 reveals the sequentiality bottleneck of CoT 15 Interestingly, some existing work finds that MDMs that decode based on the most confident symbols tend to decode autoregressively (Gong et al., 2025). In this sense, the unmasked nature of MDMs could be seen as a hurdle that the model has to overcome to eventually rely on more autoregressive generation. This further motivates the development of hybrid models that combine autoregressive generation of entire blocks with non-autoregressive infilling within the blocks (Nie et al., 2025;Arriola et al., 2025;Song et al., 2025, inter alia).
and an expressivity gap between CoT transformers and MDMs with logarithmically many model evaluations: While CoT transformers remain in TC 0 , MDMs can solve NC 1 -complete problems. This formalizes the intuition that MDMs are more suitable for highly-parallelizable problems and has implications for the practical applications of these two paradigms with a limited number of model evaluations. For example, the common state-tracking benchmark used to evaluate the reasoning abilities (Liu et al., 2023;Merrill et al., 2024) can be solved with MDMs with logarithmically many steps, while CoT transformers require linearly many steps. On the other hand, the equivalence of MDMs with polylogarithmically many denoising steps to the class NC (cf. Cor. 3.4) reveals problems where efficiency gains from parallelism are limited. For example, assuming the widely-believed hypothesis that NC ‰ P, none of the following (P-complete) problems benefit from MDM parallelism:
• Circuit value problem: Given a circuit, its inputs, and a gate, calculate the gate's value.
• Linear programming: Maximize a linear function subject to linear inequality constraints.
• Context free grammar (CFG) membership: Given a CFG G and a string w, is w P LpGq?
• Horn-satisfiability (P version of SAT): Is there a satisfying assignment to a set of Horn clauses?
In other words, these problems, in general, require a "CoT-style" step-by-step sequential solution. Due to the overhead introduced by unmasked attention of MDMs (such as the inability to maintain KV-cache), such problems are more efficiently solved by standard autoregressive CoT transformers.
Equivalence to padded looped transformers. Cor. 3.1 reveals a tight connection between MDMs and PLTs. While this suggests these two frameworks are largely interchangeable, important distinctions exist. On the one hand, unlike MDMs, standard PLTs perform sequential computations deterministically. This makes MDMs more suitable for ambiguous generation tasks, where decisions early in the generation make subsequent decisions easier. PLTs would, in that case, have to keep track of all possible generations in the residual stream until the final-decoding-step. Moreover, MDMs are easier to train and steer-since their intermediate computation steps are based on partially masked inputs, the model explicitly learns to solve complex tasks from random sub-tasks, which benefits their reasoning abilities (Kim et al., 2025). PLTs, in contrast, only receive training signal from the final decision and have to construct the sub-steps of the computation on their own. This could lead to suboptimal utilization of the sequential computation or even to failure to use it at all. Similarly, the human-understandable sub-tasks that MDMs have to solve make their training more interpretable and easier to control. On the other hand, the more information-rich intermediate states of PLTs make them more efficient at storing and processing information, and the lack of sampling steps makes them more efficient at inference time.
Generalizations. By focusing on transformer-based MDMs, we can draw from the rich theory developed on the expressivity of transformers. However, MDMs do not have to be implemented by a transformer-they could, for example, be implemented by a state-space model. Nevertheless, the parallelizable nature of MDMs suggests that any reasonable real-world implementation will include parallelizable components-for example, a model implementable by a constant-depth circuit, such as a TC 0 circuit. 16 The close connection between transformers (with logarithmically-growing precision and padding) and the class TC 0 (Merrill & Sabharwal, 2023;Li et al., 2024) suggests that the results of §3 will largely carry over to such implementations. We conjecture that, regardless of whether MDMs are implemented by finite-precision transformers (AC 0 circuits) or a more expressive TC 0 circuit, MDMrlog d N, polypN qs would remain in AC d (cf. Cor. 3.3) or a similar class like TC d . Moreover, we state the sequentiality bottleneck only for log N decoding steps, since the expressivity of CoTrlog N s is known to be limited. However, we believe that a similar separation exists for polylogarithmically many decoding steps: While the expressivity of CoTrlog d N s has not been formalized yet, it likely does not capture all of AC d , unlike MDMrlog d N, polypN qs (cf. Cor. 3.3).
this section cite: ['b18', 'b1', 'b28', 'b2', 'b34', 'b27']

Section: References
Ref_id:b0 Title: Generating text from language models Year: (2023)
Ref_id:b1 Title: Block Diffusion: Interpolating between autoregressive and diffusion language models. arXiv Year: (2025)
Ref_id:b2 Title: Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation. arXiv Year: (2025)
Ref_id:b3 Title: Accelerated sampling from masked diffusion models via entropy bounded unmasking. arXiv Year: (2025)
Ref_id:b4 Title: Evaluation of large language models via coupled token generation Year: (2025)
Ref_id:b5 Title: What languages are easy to language-model? a perspective from learning probabilistic regular languages Year: (2024)
Ref_id:b6 Title: Training neural networks as recognizers of formal languages Year: (2025)
Ref_id:b7 Title: On affine homotopy between language encoders Year: (2025)
Ref_id:b8 Title: Counterfactual token generation in large language models Year: (2024)
Ref_id:b9 Title: Convergence analysis of discrete diffusion model: Exact implementation through uniformization. arXiv Year: (2024)
Ref_id:b10 Title: Overcoming a theoretical limitation of self-attention Year: (2022)
Ref_id:b11 Title: Formal aspects of language modeling. arXiv, 2024 Year: ()
Ref_id:b12 Title: Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv Year: (2019)
Ref_id:b13 Title: BERT: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b14 Title: Towards revealing the mystery behind chain of thought: A theoretical perspective Year: (2023)
Ref_id:b15 Title: Theoretical benefit and limitation of diffusion language model. arXiv, 2025 Year: ()
Ref_id:b16 Title: Parity, circuits, and the polynomial-time hierarchy Year: (1984-12)
Ref_id:b17 Title: Mask-predict: Parallel decoding of conditional masked language models Year: (2019)
Ref_id:b18 Title: DiffuCoder: Understanding and improving masked diffusion models for code generation Year: (2025)
Ref_id:b19 Title: Why are sensitive functions hard for transformers Year: (2024-08)
Ref_id:b20 Title: DiffusionBERT: Improving generative masked language models with diffusion models Year: (2023)
Ref_id:b21 Title: Unique hard attention: A tale of two sides. arXiv Year: (2025)
Ref_id:b22 Title: Train for the worst, plan for the best: Understanding token ordering in masked diffusions. arXiv, 2025 Year: ()
Ref_id:b23 Title: Measuring faithfulness in chain-of-thought reasoning. arXiv Year: (2023)
Ref_id:b24 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b25 Title: A convergence theory for diffusion language models: An informationtheoretic perspective. arXiv Year: (2025)
Ref_id:b26 Title: Characterizing the expressivity of transformer language models. arXiv Year: (2025)
Ref_id:b27 Title: Chain of thought empowers transformers to solve inherently serial problems Year: (2024)
Ref_id:b28 Title: Transformers learn shortcuts to automata Year: (2023)
Ref_id:b29 Title: Think while you generate: Discrete diffusion with planned denoising. arXiv, 2025a Year: ()
Ref_id:b30 Title: Perfect diffusion is TC 0 -Bad diffusion is Turing-complete. arXiv Year: (2025)
Ref_id:b31 Title: The serial scaling hypothesis. arXiv, 2025b Year: ()
Ref_id:b32 Title: Pause tokens strictly increase the expressivity of constant-depth transformers. arXiv Year: (2025)
Ref_id:b33 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2024)
Ref_id:b34 Title: The parallelism tradeoff: Limitations of log-precision transformers Year: (2023)
Ref_id:b35 Title: The expressive power of transformers with chain of thought Year: (2024)
Ref_id:b36 Title: Exact expressive power of transformers with padding Year: ()
Ref_id:b37 Title: A little depth goes a long way: The expressive power of log-depth transformers. arXiv, 2025b Year: ()
Ref_id:b38 Title: The illusion of state in state-space models Year: (2024)
Ref_id:b39 Title: Large language diffusion models. arXiv, 2025 Year: ()
Ref_id:b40 Title: On the representational capacity of neural language models with chain-of-thought reasoning Year: (2024)
Ref_id:b41 Title: Counterfactual off-policy evaluation with Gumbel-max structural causal models Year: (2019)
Ref_id:b42 Title: Path planning for masked diffusion model sampling. arXiv, 2025 Year: ()
Ref_id:b43 Title: Attention is turing-complete Year: (2021)
Ref_id:b44 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b45 Title: Gumbel counterfactual generation from language models Year: (2025)
Ref_id:b46 Title: Reasoning with latent thoughts: On the power of looped transformers. arXiv Year: (2025)
Ref_id:b47 Title: Seed Diffusion: A large-scale diffusion language model with high-speed inference. arXiv Year: (2025)
Ref_id:b48 Title: What formal languages can transformers express? a survey Year: (2024)
Ref_id:b49 Title: Speed always wins: A survey on efficient architectures for large language models. arXiv Year: (2025-08)
Ref_id:b50 Title: Transformers can represent n-gram language models Year: (2024)
Ref_id:b51 Title: Can transformers learn n-gram language models? Year: (2024)
Ref_id:b52 Title: The exact expressive power of fixed-precision looped padded transformers Year: (2025)
Ref_id:b53 Title: Attention is all you need Year: (2017)
Ref_id:b54 Title: Introduction to Circuit Complexity: A Uniform Approach Year: (1999)
Ref_id:b55 Title: Generalized interpolating discrete diffusion Year: (2025)
Ref_id:b56 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b57 Title: Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv Year: (2025)
Ref_id:b58 Title: To CoT or to loop? A formal comparison between chain-of-thought and looped transformers. arXiv Year: (2025)
Ref_id:b59 Title: Masked hard-attention transformers recognize exactly the star-free languages Year: (2024)
Ref_id:b60 Title: Beyond autoregression: Discrete diffusion for complex reasoning and planning Year: (2025)
Ref_id:b61 Title: A survey on parallel text generation: From parallel decoding to diffusion language models. arXiv Year: (2025)
Ref_id:b62 Title: A reparameterized discrete diffusion model for text generation. arXiv, 2024 Year: ()
