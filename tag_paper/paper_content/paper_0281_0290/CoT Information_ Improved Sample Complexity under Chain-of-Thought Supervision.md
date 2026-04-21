Title: CoT Information: Improved Sample Complexity under Chain-of-Thought Supervision
Abstract: Learning complex functions that involve multi-step reasoning poses a significant challenge for standard supervised learning from input-output examples. Chainof-thought (CoT) supervision, which augments training data with intermediate reasoning steps to provide a richer learning signal, has driven recent advances in large language model reasoning. This paper develops a statistical theory of learning under CoT supervision. Central to the theory is the CoT information, which measures the additional discriminative power offered by the chain-of-thought for distinguishing hypotheses with different end-to-end behaviors. The main theoretical results demonstrate how CoT supervision can yield significantly faster learning rates compared to standard end-to-end supervision, with both upper bounds and information-theoretic lower bounds characterized by the CoT information.

Section: Introduction
"Chain-of-thought" (CoT) reasoning has been a driving force behind recent advances in the capabilities of large language models. While chain-of-thought began as a prompting technique [1][2][3], CoT-supervised training is now an important component of the post-training pipeline for large language models, and has been found to be highly effective in recent empirical research [4][5][6].
This paper proposes new concepts in statistical learning theory that are aimed at gaining insight into chain-of-thought learning. Consider the following concrete example of chain-of-thought, to ground the theoretical framework to be developed. The input x is the sequence "Which is larger, 8.9 or 8.10?" and the intended answer y is "8.9". When asked to answer directly, earlier systems (e.g., GPT-4) might respond incorrectly [e.g., 7]. However, newer models trained with chain-of-thought supervision will typically first output a CoT z, represented as a sequence of tokens, enabling the model to arrive at the correct answer. For example, the CoT might be, "Compare the numbers as decimals, with 8.9 written as 8.90 and where 8.10 already has two decimal places. Then compare them digit by digit. The numbers agree in the first digit. But 9 is larger than 1 in the tenths place, so 8.9 is larger than 8.10.". At test time, the CoT z serves as an explanation of the answer. During training, however, the CoT z plays the role of a natural language description of the execution trace of an algorithm-a step-by-step procedure that is to be learned. In this way, CoT is used as a rich, additional supervised learning signal that goes beyond standard input-output ("end-to-end") supervision.
The focus of our theory is to describe how this additional information impacts the statistical complexity of CoT-supervised learning. A key contribution of the paper is to identify a quantity that we call the chain-of-thought information, denoted by I CoT D,h⋆ (ε; H). As we show, the CoT information characterizes the statistical complexity of CoT-supervised learning and captures the additional discrimination power granted to the learning algorithm by observing the chain-of-thought. In particular, the CoT information governs how the end-to-end error of the learned algorithm scales with the number of CoT training examples. Specifically, in the standard setting of PAC learning for bi-nary classification, the sample complexity scales as m = O (d/ε) in the realizable setting, where d describes the size or complexity of the hypothesis space (e.g., the VC dimension), and ε is the target classification error. In contrast, under CoT supervision, we show that the sample complexity scales according to m = O(d/I CoT
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']

Section: D,h⋆ (ε; H)).
A case where the chain-of-thought is highly informative will have I CoT D,h⋆ (ε; H) ε, which translates into favorable sample complexity. By establishing information-theoretic lower bounds, it is shown that the CoT information thus provides a fundamental measure of the value of this type of non-classical supervision. Because the construction of CoT training sets can be a time-consuming and expensive process, the theoretical framework developed in this paper may ultimately be of practical relevance, contributing to a formal understanding and quantification of the value of chain-of-thought supervision.
The remainder of the paper is organized as follows. Section 2 introduces an abstract model of chainof-thought supervised learning, together with formal definitions of the learning objective and the key notions of risk, which we will use in our investigation. In Section 3, we motivate and introduce the CoT information measure, establish its fundamental properties, and, as an initial pedagogical result, show that it can be used to capture the improved sample complexity of CoT-supervised learning in the setting of finite-cardinality hypothesis classes. Section 4 extends this analysis to infinite hypothesis classes, as well as to the agnostic setting where the data are not assumed to be generated by a member of the class. In Section 5, two types of information-theoretic lower bounds are established that, together with the upper bounds, lends further support to considering the CoT information as a fundamental characterization of the value of chain-of-thought supervision. Section 6 concludes with a summary of further extensions that are presented in the appendix, a discussion of related work, and directions for future research that are suggested by the results of this paper.
this section cite: []

Section: Preliminaries: A Model of Chain-of-Thought Supervised Learning
The standard statistical learning problem is formulated as the problem of selecting a distinguished member of a function class F : X → Y, mapping from an input space X to an output space Y. The learner observes a dataset of input-output examples {(x i , y i )} i∈[m] and seeks to identify the ground truth function f ⋆ ∈ F (in the realizable setting) or compete with its closest approximation in F (in the agnostic setting). A learning algorithm in the standard ("end-to-end") setting is a mapping A : (X × Y) * → Y X from input-output datasets to predictors.
When the target function class F is highly complex-such as functions representing multi-step reasoning processes-learning from input-output examples alone can be statistically intractable. To overcome this difficulty, a natural approach is to provide the learner with increased supervision through the step-by-step execution of the target function on the input. To formulate this, we assume that each example observed by the learner includes not only the input x and output y, but also an auxiliary observation z that represents information about the function's execution on x.
A chain-of-thought (CoT) hypothesis class H is a family of functions h : X → Y × Z. For each h ∈ H, an input x ∈ X yields h(x) = (y, z), where y ∈ Y is the output and z ∈ Z is the corresponding CoT. We denote the components of h returning only the output as its end-to-end restriction, h e2e : X → Y, and the component returning only the CoT as its CoT restriction, h CoT : X → Z. In the chain-of-thought learning setting, the learner observes a dataset {(x i , y i , z i )} i∈ [m] and seeks to learn the underlying end-to-end function. A chain-of-thought learning algorithm is a mapping A : (X × Y × Z) * → Y X from CoT datasets to predictors.
A key example captured by this framework is autoregressive sequence models, generating the CoT sequentially as z t = f (x 1 , ..., x n , z 1 , ..., z t-1 ) until a final output y = f (x 1 , ..., x n , z 1 , ..., z t(x) ) is generated. In this case, the spaces X , Y, Z would correspond to spaces of variable-length sequences over some vocabulary. Sequence models like transformers [8] are an important way to implement such hypothesis classes in a way that allows for CoT supervision. However, the details of any such implementation are not important for our theoretical treatment.
this section cite: ['b7']

Section: Types of risk.
It will be crucial to distinguish between two notions of risk: End-to-end risk and the chain-of-thought risk. Let D be a distribution over X . For a reference hypothesis h ⋆ ∈ H and a predictor h ∈ (Y × Z) X , we define these risks as follows: That is, R e2e D (h) is the probability that the predictor's end-to-end output is incorrect, whereas R CoT D (h) is the probability that either the output h e2e (x) or the CoT h CoT (x) disagrees with h ⋆ . A key characteristic of the chain-of-thought supervised learning setting is that the training objective is the CoT loss, whereas the testing evaluation metric is the end-to-end risk. This asymmetry has important information-theoretic implications, which are a main focus of this work.
R e2e D (h) = P x∼D h e2e (x) = h e2e ⋆ (x) , R CoT D (h) = P x∼D [h(x) = h ⋆ (x)] .
this section cite: []

Section: Interlude: The problem of linking the end-to-end and chain-of-thought risks
Before introducing the CoT information measure and our main theoretical result, we first motivate a key aspect of the analysis, which is specific to the CoT setting. In CoT learning, the learner observes training examples S = {(x i , y i , z i )} i∈[m] and seeks to identify the input-output relationship using information from both the output y i and CoT labels z i . That is, although the CoT error is used as a signal during training, only errors in the final output y are penalized at test time. Consequently, to derive sharp statistical rates, it is necessary to link the two risk functions precisely.
To explain, recall that standard statistical learning theory characterizes the statistical complexity of learning from input-output examples without chain-of-thought supervision. For example, focusing on the realizable case for clarity, standard results in PAC learning [e.g., 9] show that the sample complexity to obtain end-to-end error ε scales as d/ε, where d is a complexity measure such as logcardinality or the VC dimension of the end-to-end loss class L e2e (H). Intuitively, the ε-dependence can be understood in terms of the amount of information per sample, as O(1/ε) samples are required to distinguish between two hypotheses whose outputs disagree on a subset of measure ε in the input space. Matching information-theoretic lower bounds validate that these are the optimal learning rates for the standard E2E-supervised setting.
In the CoT-supervised setting, the learning algorithm potentially has access to more information by observing the CoT, and thus faster rates of convergence are expected. The theoretical challenge lies in capturing this added information as improved rates in the analysis. Standard learning theory results cannot be directly applied to the CoT setting due to the mismatch between the training objective and the evaluation metric. One approach to address this challenge, which is taken by Joshi et al. [10], is to side-step this asymmetry by noting that the end-to-end error is always upper bounded by the CoT error, with R e2e D (h) ≤ R CoT D (h), and to instead establish a guarantee on the CoT risk, allowing the use of standard results in learning theory. In particular, one can define the CoT loss class for the hypothesis class H as a function class over X × Y × Z according to
L CoT (H) = ℓ CoT h : (x, y, z) → 1{h(x) = (y, z)} h ∈ H .
Then, appealing to standard results in PAC learning [e.g., Vapnik's "General Learning" framework 9], one can learn H to obtain a CoT risk of ε with a sample complexity m(ε) = O(VC(L CoT (H))/ε), which in turn guarantees that the end-to-end risk is also bounded by ε.
This method of analysis leads to a sample complexity with the same 1/ε rate that we see in the endto-end supervision setting, despite the increased amount of information per sample. In particular, this does not imply improved sample complexity over standard end-to-end supervision in the case of finite-cardinality classes (c.f. Table 1). In the general case, improved sample complexity hinges on whether or not the inequality VC(L CoT (H)) VC(L e2e (H)) holds, which is a priori unclear, even if it is possible to construct artificial classes for which this holds [10]. This suboptimality stems from the fact that this approach does not distinguish between the two types of risk and does not explicitly measure the amount of information encoded in the chain-of-thought. As a consequence, this approach cannot achieve matching information-theoretic lower bounds. Moreover, it is unclear whether it is meaningful to apply this type of analysis to the agnostic setting, where the distribution over input-output-CoT examples is not realizable by the CoT hypothesis class.
this section cite: ['b9', 'b9']

Section: Key Idea: The CoT Information Measure
We now describe a new approach that explicitly accounts for the additional information provided in the CoT supervision for distinguishing between hypotheses with different end-to-end behaviors.
this section cite: []

Section: Definition 1 (CoT information).
For a CoT hypothesis class H ⊂ (Y × Z) X and distribution D over X , we define the CoT information measures as follows:
I CoT D (h 1 , h 2 ) = -log P x∼D h CoT 1 (x) = h CoT 2 (x), h e2e 1 (x) = h e2e 2 (x) I CoT D,h⋆ (ε; H) = inf h∈∆ e2e D (ε;H,h⋆) I CoT D (h ⋆ , h).
where the infimum is over ∆ e2e D (ε; H, h ⋆ ), the set of hypotheses that disagree with the end-to-end behavior (i.e., output) of h ⋆ with probability at least ε,
∆ e2e D (ε; H, h ⋆ ) := h ∈ H : P x∼D h e2e ⋆ (x) = h e2e (x) > ε .
The relative CoT information between two hypotheses I CoT D (h 1 , h 2 ) quantifies how effectively the observed CoT behavior distinguishes the two hypotheses. In particular, the probability
P[h CoT 1 (x) = h CoT 2 (x), h e2e
1 (x) = h e2e 2 (x)] ∈ (0, 1) represents the proportion of inputs on which a pair of hypotheses have matching behavior on both the CoT and the end-to-end output, rendering them indistinguishable from these observations. The relative CoT information between a pair of hypotheses is the negative logarithm of this probability; thus, I CoT D (h 1 , h 2 ) takes values in [0, ∞). The CoT information of a hypothesis class H, relative to the reference hypothesis h ⋆ , is a function of the error level ε, denoted I CoT D,h⋆ (ε; H). It is defined as the minimal relative CoT information between h ⋆ and every alternative hypothesis h ∈ ∆ e2e D (ε; H, h ⋆ ) which disagrees with h ⋆ 's end-to-end output on more than an ε fraction of the inputs. A large I CoT D,h⋆ (ε; H) thus ensures high distinguishability (via CoT) between h ⋆ and any such "bad" alternative.
A primary message of this work is that the CoT information characterizes the ε-dependence of sample complexity in Chain-of-Thought supervised learning by quantifying the informativeness of CoT supervision. The CoT information can be much larger than ε, yielding rapid learning under CoT supervision. The intuition is that when two hypotheses differ in terms of their end-to-end behavior, even with small probability, they will typically differ in terms of their computational traces (i.e., CoT) with high probability. Consequently, CoT supervision allows these differing hypotheses to be distinguished far more rapidly than by observing input-output samples alone.
this section cite: []

Section: Properties of the CoT information
The following result outlines key properties of the CoT information measure. Among these, the property I CoT D,h⋆ (ε; H) ≥ ε is particularly important. As will be demonstrated, this implies that, in the realizable setting, CoT supervision is never detrimental, information-theoretically. The proof of these properties is given in Appendix A.
this section cite: []

Section: Lemma 1.
Let H ⊂ (Y × Z) X be a CoT hypothesis class. Then the CoT information I CoT D,h⋆ (ε; H) satisfies the following properties:
1. I CoT D,h⋆ (ε; H) ≥ ε. 2. I CoT D,h⋆ (ε; H) is monotonically increasing in ε.
this section cite: []

Section: I CoT
D,h⋆ (ε; H) is monotonically decreasing in H (under the subset relation). Before proceeding with bounding sample complexity in terms of CoT information, we note how the measure behaves in extreme boundary conditions. First, let us consider an example where the CoT annotations are entirely independent of the end-to-end behavior. In particular, consider a CoT hypothesis class with a product structure H = F CoT × F e2e , where F CoT ⊂ Z X , F e2e ⊂ Y X . In this case, we would expect no statistical advantage from observing the CoT-this is captured by the CoT information measure, which coincides with the "end-to-end information" in this case. At the other extreme, consider the case where the CoT from any single example reveals the entire target function. For example, let F ⊂ Y X and consider the CoT hypothesis class H = {h f : x → (f, f (x)) : f ∈ F}. In this case, I CoT D,h⋆ (ε; H) = ∞, which corresponds to the fact that a single example is sufficient to attain zero error. Finally, consider the problem of learning a regular language with CoT supervision. Here, we take the output y to indicate whether or not the string x is in the language, but we let z be the sequence of states visited in a DFA representing the language as it processes x; see Figure 2. Appendix B provides a more detailed discussion of these examples.
this section cite: []

Section: Improved sample complexity via CoT information
To illustrate the main ideas and intuitions underpinning this paper's results, we next prove a sample complexity bound for CoT-supervised learning with finite hypothesis classes in the realizable setting. While other proofs are deferred to the appendix for clarity and brevity, this particular result is proven here due to its simplicity and pedagogical value.
The learning rule we consider is chain-of-thought consistency, CoT-Cons(S; H): given a sample S, the learner returns any hypothesis in H which is consistent with the sample S = {(x i , y i , z i )} i∈[m] in terms of both outputs and the chain-of-thought.
this section cite: []

Section: Result 1 (Learning with Chain-of-Thought Supervision).
Let H ⊂ (Y × Z) X be a finite CoT hypothesis class. For any distribution D over X × Y × Z realized by some h ⋆ ∈ H, the CoT consistency learning rule has a sample complexity of
m(ε, δ) = log |H| + log(1/δ) I CoT D,h⋆ (ε; H) .
That is, for any m ≥ m(ε, δ), with probability at least 1 -δ over S ∼ D m , any hypothesis h that is CoT consistent on S will have end-to-end risk satisfying R e2e D (h) ≤ ε.
Proof. We aim to bound the probability of the "bad event"
{∃h ∈ H : R e2e D (h) > ε, R CoT S (h) = 0}
over the draw of (x 1 , . . . , x m ) i.i.d.
this section cite: []

Section: ∼ D.
We highlight that the training loss is the empirical CoT risk, R CoT S (h), whereas the test metric is the end-to-end risk R e2e D (h). Fix any h ∈ H with end-to-end error larger than ε, R e2e D (h
) = P x∼D [h e2e (x) = e2e(h ⋆ )(x)] > ε (i.e., h ∈ ∆ e2e D (ε; H, h ⋆ )).
We bound the probability that h is CoT consistent on S, h ∈ CoT-Cons(S; H) = {h ∈ H : R CoT S (h) = 0}, as follows Choosing m = log|H|+log(1/δ)
P S∼D ⊗m [h ∈ CoT-Cons(S; H)] = P S∼D ⊗m ∀i, h CoT (x i ) = h CoT ⋆ (x i ), h e2e (x i ) = ( e2e h ⋆ )(x i ) = P x∼D h CoT (x) = h CoT ⋆ (x), h e2e (x) = h e2e ⋆ (x)
I CoT D,h⋆ (ε;H)
implies that for each hypothesis h ∈ ∆ e2e D (ε; H, h ⋆ ) with end-to-end error larger than ε, the probability that it is in the CoT consistency set is bounded by
P S∼D ⊗m [h ∈ CoT-Cons(S; H)] ≤ δ |H| .
Applying a union bound over H yields
P S∼D ⊗m ∃h ∈ H : R e2e D (h) > ε, R CoT S (h) = 0 ≤ δ
to complete the proof. This result demonstrates that, for CoT learning, the ε-dependence of the sample complexity is O(1/I CoT D,h⋆ (ε; H)), contrasting with the typical rate of O(1/ε). Intuitively, the ratio I CoT D,h⋆ (ε; H)/ε ≥ 1 quantifies the relative value of a CoT training example compared with an endto-end training example. Figure 2 previews simulation results exploring CoT information in the context of learning a regular language, where the CoT is the sequence of states from the underlying deterministic finite automaton (DFA). Its left panel depicts the ratio I CoT D,h⋆ (ε; H)/ε as a function of ε. The plot can be interpreted as follows: For this hypothesis class H and distribution D (uniform), each CoT example is roughly 600 times more valuable than a single end-to-end example. The right panel presents empirical learning curves for the CoT-Cons and E2E-Cons rules, illustrating the statistical advantage of CoT supervisionan advantage theoretically captured by the CoT information measure. Further simulation results are presented in Appendix C.
this section cite: []

Section: Guarantees for CoT-Supervised Learning: Upper Bounds
This section extends our exploration of statistical upper bounds to infinite hypothesis classes and the agnostic learning setting, thereby further elucidating the statistical advantage of CoT supervision.
this section cite: []

Section: The realizable setting: Extension to infinite classes
Result 1 established a sample complexity bound determined by two key factors: the term 1/I CoT D,h⋆ (ε; H), which captures the information per CoT-supervised sample, and the log-cardinality of the class, log |H|, which reflects its size or dimension. We now extend this result to infinite classes, replacing the log-cardinality term with the VC dimension of the CoT loss class. As before, the upper bound is achieved by the CoT consistency learning rule.
this section cite: []

Section: Result 2 (Learning infinite classes under CoT supervision).
Let H ⊂ (Y ×Z) X be a CoT hypothesis class. For any distribution D over X ×Y ×Z realized by some h ⋆ ∈ H, the CoT consistency learning rule has a sample complexity of
m(ε, δ) = O 1 I CoT D,h⋆ (ε; H, h ⋆ ) + 1 VC(L CoT (H)) • log 1 I CoT D,h⋆ (ε; H) + 1 + log(1/δ) .
That is, for any m ≥ m(ε, δ), with probability at least 1 -δ over S ∼ D m , any hypothesis h that is CoT consistent on S will have end-to-end risk satisfying R e2e D (h) ≤ ε.
The proof is provided in Appendix D.1. The result follows from a lemma that relates the CoT risk of any proper CoT learning rule (i.e., one that returns a predictor in the hypothesis class) to its performance with respect to the end-to-end error.
this section cite: []

Section: The agnostic setting
The previous results assume that the data distribution D over X × Y × Z is realizable by the CoT hypothesis class H. Such an assumption can be stringent, particularly in the presence of noise. This section, therefore, addresses the agnostic setting, where no restriction is made on the distribution; the goal, instead, is to compete with the best hypothesis in the class H in terms of end-to-end risk.
In the agnostic setting, a natural learning rule is CoT empirical risk minimization, which selects a hypothesis that minimizes the empirical CoT risk: CoT-ERM(S; H) = arg min h∈H R CoT S (h).
Recall that CoT supervision never hurts in the realizable setting since I CoT D,h⋆ (ε; H) ≥ ε for any hypothesis class. The picture is more complicated in the agnostic setting. In particular, CoT supervision can be harmful or distracting, and discarding the CoT annotation and learning from only the input-output examples can be preferable, as the following example shows. The issue arises when the CoT hypothesis class H is not aligned with the data distribution, especially when H can fit the end-to-end behavior but not the CoT behavior.
Example. Consider a CoT hypothesis class H : X → Y × Z and suppose D is a distribution over X × Y × Z for which the output component is realizable by H but the CoT component is not realizable. In particular, it is easy to construct examples for which
inf h∈H R e2e D (h) = 0 while inf h∈H R CoT D (h) = 1.
Clearly, in such cases, the CoT-ERM learning rule provides no guarantees whatsoever since CoT-ERM(S; H) = H for any S supported by D. In contrast, E2E-ERM enjoys the standard PAC learning guarantees, with a sample complexity O 1 /ε • VC(L e2e (H)) .
Thus, CoT supervised learning in the agnostic setting requires a different notion of CoT information, which captures how well-aligned the data distribution is to the hypothesis class, and whether fitting the CoT aligns with fitting the end-to-end behavior. This uses an excess risk variant of the CoT information, defined in the following result, which extends our results to the agnostic setting.
this section cite: []

Section: Result 3 (Agnostic learning under CoT supervision).
Let H ⊂ (Y ×Z) X be a CoT hypothesis class. For any distribution D over X × Y × Z, the CoT-ERM learning rule has a sample complexity of m(ε, δ) = O VC(L CoT (H)) + log(1/δ)
I CoT D (ε; H) 2
, where I CoT D (ε; H), the agnostic CoT information, is defined via excess risks as
I CoT D (ε; H) := inf E CoT D (h) : h ∈ H, E e2e D (h) ≥ ε , where E e2e D (h) := R e2e D (h)-inf h R e2e D (h) and E CoT D (h) := R CoT D (h)-inf h R CoT D (h)
are the excess CoT and end-to-end risks, respectively. That is, for any m ≥ m(ε, δ), with probability at least 1 -δ over the draw of S ∼ D m , the excess end-to-end risk is bounded as R e2e D ( ĥ) ≤ inf h R e2e D (h) + ε, where ĥ ∈ CoT-ERM(S; H).
The proof is presented in Appendix D.2. Note that, unlike in the realizable case, we do not necessarily have the lower bound I CoT D (ε; H) ≥ ε. For instance, in the motivating example above, I CoT D (ε; H) = 0. However, CoT supervision yields an advantage when the excess CoT risk dominates the excess end-to-end risk (i.e., R CoT D (h
) -R CoT ⋆ > R e2e D (h) -R e2e ⋆ ).
this section cite: []

Section: Information Theoretic Lower Bounds for CoT Supervised Learning
This section establishes information-theoretic lower bounds on sample complexity, further validating the CoT information I CoT D,h⋆ (ε; H) as a fundamental measure of statistical complexity for learning with CoT supervision. In general, the statistical complexity of a learning problem depends on several parameters, including the size or complexity of the hypothesis class (e.g., VC(H) in binary classification) and the error parameter (e.g., 1/ε or 1/ε 2 for the realizable and agnostic settings, respectively). Different types of lower bounds scale accordingly with one or both of these factors. Our main focus in this work is on the dependence of the sample complexity on the error parameter, which corresponds to the amount of information encoded in the CoT supervision for discriminating between hypotheses with different end-to-end behavior.
We begin with a lower bound demonstrating that CoT information I CoT D,h⋆ (ε; H) characterizes the ε dependence of sample complexity. The essence of the result is to lower bound the minimum number of samples required to reliably distinguish a pair of hypotheses with a given end-to-end disagreement, reducing the learning problem to a binary hypothesis testing problem [11], and relating the total variation distance between distributions over X × Y × Z induced by a pair of hypotheses to the relative CoT information between them.
this section cite: ['b10']

Section: Result 4 (Lower bound via CoT information).
Let H ⊂ (Y × Z) X be a CoT hypothesis class and let D be a distribution on X . Let x 1 , . . . , x m ∼ D be an i.i.d sample from D. For any h ⋆ ∈ H and ε > 0, if the sample size satisfies m < log(1/δ) I CoT D,h⋆ (ε; H) then with probability at least δ, there exists h ∈ H with end-to-end error at least ε which is indistinguishable from h ⋆ on the sample. Moreover, the expected error of any algorithm A satisfies
sup h⋆∈H E S∼P ⊗m h⋆ R e2e D,h⋆ (A(S)) ≥ 1 2 sup h⋆∈H ε>0 ε • exp(-m • I CoT D,h⋆ (ε; H)).
This result validates the CoT information as characterizing the ε-dependence of the rate. However, a weakness of two-point methods is that they do not scale with the size of the hypothesis space.
The following result addresses this by reducing the learning problem to that of testing multiple hypotheses, using a packing of the hypothesis space with respect to the end-to-end error. We then use Fano's inequality to lower bound the probability of error in terms of a mutual information, and relate this mutual information to the CoT information. To apply Fano's method in this way, we extend the framework to allow the observed z to be a stochastic function of the hypothesis CoT.
Result 5 (Lower bound via Fano's method). Let H ⊂ (Y × Z) X be a CoT hypothesis class and let D be a distribution over X . Suppose that x 1 , . . . , x m ∼ D. Let Q ∈ P(Y × Z | Y × Z) be a noisy channel from h(x) = (y, z) to observations ȳ, z.
Let C Q = max a,b D KL (Q(• | a) Q(• | b))
be the capacity factor of the channel. The learner observes the noisy sample S = {(
x i , ȳi , zi )} m i=1 . Define the pseudo-metric d e2e D (h 1 , h 2 ) = P x [h e2e 1 (x) = h e2e 2 (x)]
, and let M (ε; H, d e2e D ) be the ε-packing number of H with respect to this pseudo-metric. Then, for any algorithm A, we have that
m ≤ log M (H, d e2e D , ε) 2 • C Q • sup π E h1,h2∼π I CoT D (h 1 , h 2 ) + log 2 ,
implies large error for some h ⋆ ∈ H with high probability, i.e.,
sup h⋆∈H P S∼P ⊗m h⋆ R e2e D,h⋆ (A(S)) ≥ ε 2 ≥ 1/2.
Here, C Q is a bound on the capacity of the channel that adds noise to the chain-of-thought. This lower bound relates the probability of large error to the CoT information measure, like the previous result, but also scales with the size of the hypothesis space, as measured by its packing number. Additionally, the result also models noise in the learning process by observing the CoT through a noisy channel. The proofs of both results, along with further discussion, are presented in Appendix E.
6 Discussion and Related Work
this section cite: []

Section: Further explorations
We describe additional results not included in the main paper, and defer to the appendix for details.
this section cite: []

Section: Learning with mixed CoT and E2E supervision.
In practice, obtaining CoT-annotated examples can be a costly and labor-intensive process, limiting their quantity, whereas input-output examples without CoT annotation can be relatively cheap and plentiful. This motivates a need for learning algorithms that can make use of both types of examples. Appendix F.1 studies learning from datasets with a mix of E2E and CoT supervision.
this section cite: []

Section: CoT learning with inductive priors.
Encoding prior knowledge about solution structure is critical for learning complex functions, such as those representing multi-step reasoning processes, particularly from limited data. Appendix F.2 explores chain-of-thought learning with inductive priors.
this section cite: []

Section: Transfer learning and out-of-distribution generalization.
Chain-of-thought supervision has significant implications for out-of-distribution generalization, as it guides the learning algorithm toward solutions that exhibit the correct step-by-step reasoning, potentially enabling robust generalization to novel input instances. In Appendix F.3, we define a variant of the CoT Information measure, I CoT Dtr→Dtest (ε; H), that captures transfer learning under CoT supervision. We also present a result on learning under CoT supervision with distribution shift, supported by experimental simulations.
this section cite: []

Section: Related work
Early usage of the term "chain-of-thought" referred to empirical prompting techniques that conditioned large language models to generate a series of intermediate reasoning steps before returning the final answer [1][2][3]12]. Such prompting often employs in-context learning, where CoT examples are provided within the model's context before it processes the input [3]. Today, the term chainof-thought takes a broader meaning, as it now comprises a core component of the training of large language models [4][5][6].
Several works have sought to theoretically understand the advantages of the chain-of-thought paradigm by analyzing the representational capacity of neural sequence models with and without chain-of-thought [13][14][15][16]. For example, Pérez et al. [13] show that Transformers can simulate Turing machines by generating CoT tokens, and Merrill and Sabharwal [14] extend this analysis by providing a more refined characterization of function classes in terms of the number of CoT steps.
While these studies demonstrate the existence of neural network models capable of computing a given function via a specific chain-of-thought, they do not address the statistical question of whether such models can be efficiently learned from data. Our work focuses on these statistical learning aspects, a direction also pursued by a few recent studies. For example, Malach [17] studies the problem of learning autoregressive next-token prediction on CoT datasets. The core idea of this work is to express a CoT function as a composition of T (a fixed number) different sequence-domain functions, H = H 1 × • • • × H T , where each H t : X × Σ t-1 → Σ maps the input and CoT generated so far (x, z 1 , ..., z t-1 ) to the next CoT symbol z t , with the T -th CoT symbol serving as the final output. With this formulation, Malach [17] proposes learning each H t independently (with independent parameters), enabling the direct application of standard PAC results. While this approach simplifies the analysis, its assumption of independently learned functions at each iteration is a notable limitation, which does not accurately reflect real-world settings.
Building on this work, Joshi et al. [10] considers a time-invariant composition of sequence-domain functions, where the function at each iteration remains the same. Their analysis relies on bounding the CoT error using standard PAC learning tools based on the VC dimension of the CoT loss class, noting that the CoT error provides an upper bound on the end-to-end error (cf. Section 2.1 and the second row of Table 1). Moreover, Joshi et al. [10] construct a synthetic autoregressive class exhibiting a gap between the VC dimension of the end-to-end loss class and CoT loss class, implying a statistical advantage for CoT supervision. While our results also involve the CoT loss class, thus inheriting the advantages of such class complexity differences, the focus of our analysis is on the content of information per CoT-supervised sample. This is represented in the dependence of the sample complexity on the target error ε, captured by the CoT information measure. This provides a more complete description of the statistical advantage of CoT supervision in statistical learning. We contend that this information-theoretic analysis, centered on CoT information rather than solely on loss class complexity, identifies a more fundamental source of statistical advantage in CoT-supervised learning. This view is supported by our lower bound results and the close agreement between our theory and simulations.
We close the discussion of related work by noting that learning with chain-of-thought supervision can be framed as a transfer learning problem [e.g., [18][19][20], where the "source task" involves learning the mapping x → (z, y) that jointly predicts the output y and the auxiliary chain-of-thought signal z, while the "target task" is the end-to-end prediction task x → y. When the CoT information is large (i.e., I CoT D,h⋆ (ε) > ε), the source distribution allows learning the target task more rapidly than the target distribution itself. This is sometimes referred to as super transfer [21].
this section cite: ['b0', 'b1', 'b2', 'b11', 'b2', 'b3', 'b4', 'b5', 'b12', 'b13', 'b14', 'b15', 'b12', 'b13', 'b16', 'b16', 'b9', 'b9', 'b17', 'b18', 'b19', 'b20']

Section: Conclusion and future work
This work provides a theoretical analysis of learning with chain-of-thought (CoT) supervision, introducing the CoT information measure to characterize its statistical advantages via both upper and lower bounds. This opens several promising directions for future theoretical study of CoT learning.
The upper bounds obtained in this work are based on analyzing natural but relatively simple learning rules: CoT-consistency in the realizable setting, and CoT-ERM in the agnostic setting. Investigating the optimality of these algorithms and exploring the design of optimal learning strategies remain key open questions. This may be especially relevant in the agnostic setting, where the alignment between the data distribution and the CoT hypothesis class is critical. For instance, future work could explore adaptive learning rules that balance the optimization of the CoT error and end-to-end error to avoid over-optimizing the CoT when the hypothesis class is poorly aligned with the data. While we use the VC dimension of CoT loss class as the measure of complexity or size of the hypothesis class, it will also be important to consider other measures of model complexity, including covering numbers, local Radamacher complexities, and one-inclusion graphs [22][23][24][25][26][27].
Furthermore, while our current lower bound results address the realizable setting, establishing corresponding lower bounds for the agnostic setting remains an important open problem. Additionally, future research could investigate the formulation of different structural conditions, such as low-noise assumptions [28,29] in the CoT setting, to achieve faster learning rates. Developing more sophisticated probabilistic analysis, beyond the standard formulation of agnostic learning, holds promise for more faithfully capturing the complexities of training language models with chain-of-thought reasoning traces, which are often inherently probabilistic.
this section cite: ['b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28']

Section: References
Ref_id:b0 Title: Show Your Work: Scratchpads for Intermediate Computation with Language Models Year: (2021)
Ref_id:b1 Title: Training Verifiers to Solve Math Word Problems Year: (2021)
Ref_id:b2 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b3 Title: Large Language Models Are Reasoning Teachers Year: (2023)
Ref_id:b4 Title: Scaling Instruction-Finetuned Language Models Year: (2022)
Ref_id:b5 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b6 Title: Evaluating ChatGPTs decimal skills and feedback generation in a digital learning game Year: (2023)
Ref_id:b7 Title: Attention is all you need Year: (2017)
Ref_id:b8 Title: Estimation of Dependencies Based on Empirical Data Year: (1982)
Ref_id:b9 Title: A Theory of Learning with Autoregressive Chain of Thought Year: (2025)
Ref_id:b10 Title: Convergence of Estimates Under Dimensionality Restrictions Year: (1973-01-01)
Ref_id:b11 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b12 Title: Attention is turing-complete Year: (2021)
Ref_id:b13 Title: The expresssive power of transformers with chain of thought Year: (2023)
Ref_id:b14 Title: Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective Year: (2023)
Ref_id:b15 Title: Chain of Thought Empowers Transformers to Solve Inherently Serial Problems Year: (2024)
Ref_id:b16 Title: Auto-Regressive Next-Token Predictors are Universal Learners Year: (2024)
Ref_id:b17 Title: Advances in Domain Adaptation Theory Year: (0210)
Ref_id:b18 Title: A theory of learning from different domains Year: (2010)
Ref_id:b19 Title: Domain Adaptation: Learning Bounds and Algorithms Year: (2009)
Ref_id:b20 Title: On the Value of Target Data in Transfer Learning Year: (2019)
Ref_id:b21 Title: Localized rademacher complexities Year: (2002)
Ref_id:b22 Title: Local rademacher complexities Year: (2005)
Ref_id:b23 Title: Some Local Measures of Complexity of Convex Hulls and Generalization Bounds Year: (2004-05-18)
Ref_id:b24 Title: Complexity Regularization via Localized Random Penalties Year: (2004-08-01)
Ref_id:b25 Title: Improving the Sample Complexity Using Global Data Year: (2002-07)
Ref_id:b26 Title: Predicting {0,1}-Functions on Randomly Drawn Points Year: (1994)
Ref_id:b27 Title: Smooth discrimination analysis Year: (1999)
Ref_id:b28 Title: Risk Bounds for Statistical Learning Year: (2006-10)
Ref_id:b29 Title: Complexity of Automaton Identification from given Data Year: (1978-06)
Ref_id:b30 Title: A Note on the Number of Queries Needed to Identify Regular Languages Year: (1981-10)
Ref_id:b31 Title: Learning Regular Sets from Queries and Counterexamples Year: (1987-11-01)
Ref_id:b32 Title: Diversity-Based Inference of Finite Automata Year: (1987-10)
Ref_id:b33 Title: Inference of Finite Automata Using Homing Sequences Year: (1989)
Ref_id:b34 Title: Efficient Learning of Typical Finite Automata from Random Walks Year: (1993)
Ref_id:b35 Title: Piecewise testable events Year: (1975)
Ref_id:b36 Title: On the learnability of shuffle ideals Year: (2013)
Ref_id:b37 Title: Introduction to automata theory, languages, and computation Year: (2001)
Ref_id:b38 Title: Festschrift for Lucien Le Cam Year: (1997)
Ref_id:b39 Title: Information Theory: From Coding to Learning Year: (2025)
Ref_id:b40 Title: Elements of Information Theory Year: (2006)
Ref_id:b41 Title: A formal theory of inductive inference. Part I Year: (1964)
Ref_id:b42 Title: Three approaches to the quantitative definition of information Year: (1965)
Ref_id:b43 Title: ) for what should or should not be described Year: ()
