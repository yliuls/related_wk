Title: On the Power of Context-Enhanced Learning in LLMs
Abstract: We formalize a new concept for LLMs, contextenhanced learning. It involves standard gradientbased learning on text except that the context is enhanced with additional data on which no autoregressive gradients are computed. This setting is a gradient-based analog of usual in-context learning (ICL) and appears in some recent works.Using a multi-step reasoning task, we prove in a simplified setting that context-enhanced learning can be exponentially more sample-efficient than standard learning when the model is capable of ICL. At a mechanistic level, we find that the benefit of context-enhancement arises from a more accurate gradient learning signal. We also experimentally demonstrate that it appears hard to detect or recover learning materials that were used in the context during training. This may have implications for data security as well as copyright.* Equal contribution with theoretical framework and setting up the problem, XZ undertook most of the experiments.

Section: Introduction
Pre-trained LLMs (Brown et al., 2020;Touvron et al., 2023;Team et al., 2023) show strong capability to learn new material at inference time, for instance via in-context-learning (ICL). There is also emerging evidence that gradient-based learning on a piece of text (say, math Q&A) can be enhanced if additional helpful text is placed in the context, even though no auto-regressive loss is computed on this helpful text (Liao et al., 2024;Zou et al., 2024;Choi et al., 2025). Such strategies have also been shown to benefit pre-training as prepending source URLs to documents can enhance the model's training efficiency and memorization capacity (Allen-Zhu & Li, 2024;Gao et al., 2025).
In this paper we seek to formally study this phenomenon, whereby LLMs' gradient-based learning is enhanced via placement of additional helpful material in the context, but without actual auto-regressive gradient updates on this material. We will call this form of learning contextenhanced learning. Because the material used for contextenhancement can evolve over the course of training, this approach naturally aligns with the idea of a curriculum.
Context-enhanced learning intuitively mirrors how humans learn: when solving problems, they refer to textbooks or demonstrations for guidance, yet they do not seek to memorize these resources per se. An analogous concept Learning using Privileged Information (or LUPI), has been wellstudied in the context of kernel SVMs (Vapnik & Vashist, 2009) and classification models. Our work adapts this concept for LLMs, and surfaces the following questions: Q1: Even though autoregressive loss is computed on the same set of tokens, can context-enhanced learning be significantly more powerful than usual auto-regressive learning that has no additional in-context materials? If so, can we theoretically characterize and understand the mechanism behind such improvement?
Q2: Do models need a certain capability level to benefit from context-enhanced learning? This is a natural question, since leveraging in-context information (e.g., ICL) likely requires a minimum capability level or model size (Brown et al., 2020;Wei et al., 2022).
Q3: Is context-enhanced learning a viable way to use privileged/private information during learning? Providing such privileged information in the context could conceivably enhance the model's learning, but since no auto-regressive gradient updates happen on the privileged/private information, there might be a lower risk of leakage of such information via API calls.
Paper Overview: Section 2.1 formally defines contextenhanced learning. To allow rigorous understanding of the power of context-enhanced learning, Section 2.2 introduces a multi-step reasoning task called Multi-layer translation. This is a synthetic setting involving d + 1 languages L 1 , L 2 , . . . , L d+1 over finite alphabets. For each i, there is a simple phrasebook that describes how to translate from L i to L i+1 , and the mapping from L 1 to L d+1 is a sequential application of the set of phrasebooks.
The goal is to learn how to translate text from L 1 into L d+1 without explicitly writing down intermediate steps. The learner is provided excerpts from these phrasebooks as helpful information in the context during training, but allowed no auto-regressive gradient updates on these tokens.
If we train with auto-regressive loss on translation output conditioning on the phrasebooks' excerpts and the input, a model with certain ICL capacity level may quickly learn the translation task by leveraging the in-context phrasebooks. However, this learning could be brittle, that the model becomes reliant on having the phrasebooks' excerpts in context. This reliance can be weaned off by use of probabilistic dropout on phrasebooks tokens in context. Intuitively, this curriculum forces the model to not only read phrasebooks' excerpts, but also gradually internalize the phrasebooks' contents. Over time, the model's ability to translate from L 1 to L d+1 will become robust to the dropout of phrasebooks' excerpts, and eventually, their complete removal.
Experiments show that this training strategy indeed works when the learner is a pre-trained LLM that is capable of ICL (but fails when LLM is incapable of ICL). Even when training with 20% dropout rate, the model can perfectly translate strings from L 1 to L d+1 without any phrasebooks' excerpts at test time. The rest of the paper is structured as follows:
• Section 3 details our experiments and the findings sketched above. Experiments show that an ICL-capable model follows an intuitive sequential processing of the phrasebooks provided in-context, whereby transformer layers approach stages of translation in an intuitive way; e.g., L 3 → L 4 is done after L 2 → L 3 (Section 3.3).
• Section 4, shows that after context-enhanced learning, the output probabilities of the model reveal little about the phrasebooks rules that were seen during training.
• In Section 5, we propose a theoretical framework using a surrogate/simplified model that represents an ideal LLM for the translation task (Section 5.1). This framework shows an exponential gap in sample complexity depending on whether the model is trained with or without in-context information on phrasebooks (Sections 5.2 and 5.3). Experiments reveal that the mechanism behind the increased sample efficiency of context-enhanced learning is an improved gradient signal, measured by gradient prediction accuracy (Section 5.4).
this section cite: ['b41', 'b40', 'b14', 'b59', 'b0', 'b42', 'b46']

Section: Setup

this section cite: []

Section: Context-Enhanced Learning
Let X be the space of all possible text strings and let Y be the space of all possible distributions over texts. Let g be a language task mapping inputs x ∈ X g ⊂ X to a distribution Y ∈ Y. Let f θ : X → Y be a general autoregressive language model. We characterize f θ 's capability on task g as follows:
Definition 2.1 (g-capable model, informal). A language model f θ is g-capable for a language task g if f θ is close to g, as measured by a suitable metric on X g .
Vanilla supervised fine-tuning (SFT) aims to create a g-capable model by minimizing auto-regressive loss ℓ auto on a supervised dataset D g = {(x i , y i )} N i=1 , where the label y i for each x i ∈ X g is sampled from g(x i ).
Context-enhanced learning involves augmenting the supervision with additional curriculum-text that depends on the task g, input x, and training step t. We denote curriculum-text as CURR g (x, t), which could be anything (helpful explanations, excerpts from textbooks, worked-out examples, etc.).
this section cite: []

Section: Algorithm 1 Context-Enhanced Learning
In contrast to standard SFT, it relies on curriculum-text in context on which no auto-regressive loss is computed.
Input: Supervised dataset D g , curriculum-text CURR g , initialization θ, total steps T for t = 1 to T do Sample (x, y) ∼ D g Compute loss l ← ℓ auto (f θ ([CURR g (x, t), x, y]), y) . Update parameters θ with gradient ∇ θ l end for Return θ On sample (x, y) drawn from the supervised dataset, we use auto-regressive loss for model's prediction on y conditioned on [CURR g (x, t), x] to train our models. Note that no loss is computed for curriculum-text tokens. We denote this loss as ℓ auto (f θ ([CURR g (x, t), x, y]), y).
this section cite: []

Section: Multi-level Translation (MLT)
To study the power of context-enhanced learning, we introduce a multi-step translation task that is easy to learn with a straightforward curriculum in the context, but very difficult to learn with just input-output examples.
The task is inspired by encryption methods 1 such as the Feistel cipher (Knudsen, 1993). The multi-level translation (MLT) task involves a bijective mapping from strings to strings that is a composition of 2d simpler bijections, each involving simple shift by 1, or transforming bigrams (2tuples of characters) via a bijection. The depth-d translation can be described by O(d) bits. But we will show that learning the task only from input-output pairs would require e Ω(d)  sample complexity in the SQ-learning framework (Kearns, 1998) (see Theorem 5.4). Translation task with a set of phrasebooks Π MLT(d, n) Family of translation tasks of depth d and n characters Concretely, let A 1 , . . . , A d+1 be d + 1 alphabets all of the same size with n characters. For every consecutive pair of alphabets A i and A i+1 , we fix a phrasebook π i :
A 2 i → A 2
i+1 as a bijective mapping from 2-tuples in A i to 2-tuples in A i+1 . Each phrasebook π i can be represented by a binary stochastic matrix Matrix(π i ) with rules represented as one-hot columns (see Definition G.4).
π 2 π 2 π 2 π 2 π 1 π 1 π 1 π 1 B C D E F G H A e b h d a f g c b h d a f g c e A B C D E F G H Phrasebook π 1 BC → eb FG → af DE → hd HA → gc … Phrasebook π 2 bh → γβ fg → θδ da → απ ce → ζµ … γ β α π θ δ ζ µ Circular Shift Circular Shift s 1 s1 s 2 s2 s 3 Figure 1.
Illustration of an MLT(2, 8) instance with input sequence of 8 tokens. Input sequence s1 went through 2 translation steps to s3. Each output character depends on 4 input characters, for example, character µ is derived from c,e in s2, which, in turn, are computed from A,B,C,H in s1.
The input of the translation process is an even-length sequence, referred to as s 1 ∈ A L 1 where L is the sequence length. The translation process modifies s 1 recursively. For every i ∈ [d], s i ∈ A L i will be transformed to s i+1 ∈ A L i+1 using phrasebook π i through the following 2 sub-processes:
• Circular shift: The characters in s i ∈ A L i are shifted by 1 character leftward (and wrapped around to the end if necessary) to give sequence si ∈ A L i . Formally, for each j ∈ [1, L] we have si,j = s i,(j+1)%L .
• Translate: Using the phrasebook π i : A 2 i → A 2 i+1 , we translate 2-tuples (bigrams) of consecutive characters in sequence si to create s i+1 . That is, for every odd j ∈ [1, L], (s i+1,j , s i+1,j+1 ) = π i (s i,j , si,j+1 ).
We denote the mapping from s i to s i+1 as s i+1 = T πi (s i ). The d-step translation is defined as the composition s d+1 = T π d •T π d-1 •• • ••T π1 (s 1 ) which converts the input sequence s 1 to s d+1 through d translation steps. Please see Figure 1 for a visual illustration with d = 2, n = 8. We denote Π = {π i } d i=1 the set of phrasebooks of all levels, and denote MLT Π : s 1 → MLT Π (s 1 ) := T π d • • • • • T π1 (s 1 ) the mapping from input s 1 to output s d+1 using Π. We use MLT(d, n) to refer to the family of translation tasks involving d step and n characters in each alphabet.
We note that MLT(d, n) has two key properties: 1. Once the phrasebooks are fixed, the translation task defines a bijection between input and output strings since Circular shift and Translate are invertible (see Lemma E.1). 2. Each character in the output string depends on 2d characters from the input text string (see caption of Figure 1), making learning from input-output pairs very difficult (Theorem 5.4).
For each phrasebook π i , we compose its textual representation STR (π i ) to be of the form ... a b -> C D; e d -> B A; ... which lists (insensitive of ordering) phrasebook rules between 2-tuples in the previous alphabet and the next alphabet. Moreover, we denote the concatenation [STR (π 1 ) , . . . , STR (π d )] as STR (Π), and will be used to define curriculum-text.
this section cite: ['b10', 'b8']

Section: Needed: Curriculum without Explicit CoT
To teach the model a particular translation task MLT Π from input-output pairs of the form (s 1 , MLT Π (s 1 )), we can train it with relevant sections of the phrasebooks STR (Π) in context as curriculum, but at test time it would not have access to the phrasebook so it is important not to teach it explicit chain-of-thought (CoT) containing in-context information. (Another consideration is data privacy, with the phrasebook being considered privileged information.) However, a dual use of CoT is to provide the model extra compute at inference time (Goyal et al., 2023), which is needed here since the translation task has d stages. To facilitate such silent computation we teach the model to output a fixed number of <THINK> tokens, sometimes refered to as silent CoT or internalized CoT.
this section cite: ['b2']

Section: ICL-capablity for MLT(d, n)
To learn from books, one needs to know how to read. The analogous notion under study here is whether contextenhanced learning requires capability to sort-of "understand" the in-context material (Q2). In the context of MLT, we formalize such capability as being able to achieve low loss on the translation task when provided with the relevant phrasebook sections in context while allowing silent CoT.
this section cite: []

Section: Definition 2.2 (MLT(d, n)-ICL-capability, informal).
A language model f θ is MLT(d, n)-ICL-capable if for any set of phrasebooks Π in MLT(d, n), f θ ([STR (Π) , s 1 ]) is close to s d+1 = MLT Π (s 1 ) disregarding the <THINK> tokens, when measured by a discrepancy metric over all valid input strings s 1 .
this section cite: []

Section: Experiments and Observations
In this section, we fix a set of phrasebooks Π * and study context-enhanced learning on MLT Π * .
We first introduce the preparation of an MLT(d, n)-ICLcapable model. We then introduce a context-enhanced learning curriculum involving random dropping of phrasebook rules in context. We then present empirical evidence for significant sample efficiency of context-enhanced learning. We conclude the section with mechanistic insights into contextenhanced learning concerning internal representations and evolution of parameters.
We use the Llama 3.2-3B instruction-tuned model (Dubey et al., 2024) as the base model and fix d = 5 with n = 8 or 10. Detailed configurations are available in Appendix B.5.
this section cite: []

Section: Experimental Setup (i) Preparing an MLT(d, n)-ICL-Capable Model:
The Llama 3.2B model is MLT(d, n)-ICL-capable as it has not seen the task during training. To make it ICL-capable for our purpose, we use SFT on other random translation tasks with random phrasebooks Π 1 , . . . , Π M , following common CoT internalization pipeline (Deng et al., 2024;Pfau et al., 2024;Hao et al., 2024). We use one training example per set of phrasebooks to prevent memorization of specific phrasebooks. At the end of training, given input [STR(Π), s 1 ] for any s 1 and Π the model can generate <THINK>,..., MLT Π (s 1 ) correctly. Details on the first stage of training are described in Appendix B.4.
(ii) Setting up context-enhanced learning for MLT Π * : We use the MLT(d, n)-ICL-capable model above as initialization and train for MLT Π * . Supervised dataset D Π * is curated with input-label pairs of the form (s 1 , [<THINK>,..., MLT Π * (s 1 )]), where s 1 is a random string sampled from A 1 , with length between 20 and 40.
We define curriculum-text CURR Π * (s 1 , t) using excerpts from phrasebooks STR(Π * ) (selected based on s 1 ) with random dropout of rules (parameterized by training step t). We explore the following curriculum and study their impact:
• No Context (vanilla SFT): Empty curriculum-text.
• Fixed Dropout: A simple strategy independent of step t; given s 1 , only curate rules in Π * used in the translation of s 1 , then randomly drop 20% of the curated rules.
• Annealing Dropout: A better strategy: for s 1 , select the necessary rules from Π * plus 25% unused rules. Apply random dropout on these rules, increasing linearly from 0% to 100% over the first 60% of training, then maintain 100%.
• No Dropout (ablation): Given s 1 , always provide all rules in Π * used in the translation of s 1 in curriculum-text.
• Wrong Context (ablation): Equivalent to Annealing Dropout but the rules in the curriculum are incorrect.
this section cite: ['b27', 'b3']

Section: Experiment Results
To check the sample efficiency benefit of context-enhanced learning (Q1), we construct supervised datasets D Π * with 10 4 to 10 6 unique samples and train the models for one epoch on each. 2 We report the next-token prediction accuracy on the final answer tokens (ignoring thought tokens) for held-out samples when conditioning on no curriculum-text (100% dropped-out) and compare against the supervised dataset size. To check the necessity of proper ICL capability (Q2), we ablate with Annealing Dropout but starting from non-MLT(d, n)-ICL-Capable 3B base model.
10 4 10 5 10 6 Number of Samples |D * | 0.00 0.25 0.50 0.75 1.00 Test Accuracy (without context) Random Guess d=5, n=8 10 4 10 5 10 6 Number of Samples |D * | Random Guess d=5, n=10 No Context (baseline) Fixed Dropout Annealing Dropout No Dropout (ablation) Wrong Context (ablation) No ICL (ablation) Figure 2 demonstrates the significant sample efficiency of context-enhanced learning. Moreover, models trained with subsets of phrasebooks and just 20% dropout give perfect heldout test-time accuracy with 100% dropout rate. Thus they are able to effectively use phrasebook rules from subsets of the phrasebook that did not co-occur in the same training sample. Clearly, the model has learned the phrasebook atomically, and can combine the rules as needed at test time.
In an ablation (Appendix C), we show that context-enhanced learning only internalizes the rules whose dropout from curriculum-text leads to an increase in loss on training data.
The experiment results can be summarized as follows:
(i) Context-enhanced learning from an ICL-capable model greatly improves training sample efficiency.
(ii) The phrasebook rules are internalized atomically, and only when missing them incurs an increased loss. 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Layer4 -Layer10 Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Layer14 -Layer17 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Layer18 -Layer19 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Layer20 -Layer22 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Layer25 -Layer25 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy Bright colors close to the diagonal reflects that knowledge required to compensate the dropped phrasebook in context can be localized in a small subset of layers in f θ * . It is particularly worth noting that for any level i, the end of the group of layers responsible for storing π * i matches the start of the layers responsible for reading πi (see Figure 3) before context-enhanced learning.
this section cite: []

Section: Mechanistic Insights: Layer by Layer Mapping of the Translation Task
To understand the behavior of context-enhanced learning, we probe into the hidden representations and weights of models before and after context-enhanced learning.
this section cite: []

Section: Sequential Processing in ICL-capable model
First, we look into how phrasebooks in curriculum-text are used by an MLT(d, n)-ICL-capable model during the translation task via the following controlled experiment. Fix a random set of phrasebooks Π = {π 1 , . . . , π d } and an input sequence s 1 , we pass [STR(Π), s 1 , <THINK>,..., s d+1 ] into the ICL-capable model, and record the model's hidden representations (output of each transformer block) for the tokens <THINK>,..., s d+1 .
Next, we apply independent perturbations to each phrasebook π i in the curriculum-text and measure the change in the model's representations. That is, for each level 1 ≤ i ≤ d, we randomly sample 10 rules in π i used in the translation of s 1 and replace their image by other randomly chosen tuples. We denote the perturbed phrasebook STR(π i ), and compute the norm difference between the model's representations at tokens <THINK>,..., s d+1 before and after replacing STR(π i ) with STR(π i ). The first layer showing a significant difference in its output representation is identified as the layer where the model begins processing the phrasebook.
In Figure 3, we show that an ICL-capable model processes phrasebooks in curriculum-text sequentially, with earlier phrasebooks read by earlier layers. Formal details are in Appendix B.2 and additional experiments are in Appendix C.2.
this section cite: []

Section: Localized Storage after Context-Enhanced Learning
Here, we verify whether a similar sequential pattern for internalizing phrasebooks is used in a MLT Π * -capable model.
We denote the ICL-capable model as f θ and its post contextenhanced learning counterpart as f θ * . As f θ * 's capability no longer depends on textual representation of the phrasebooks, our analysis focuses on the model's parameters.
We assess the importance of each layer in model f θ * for each phrasebook by constructing "stitched" models and measuring their output behavior. For every 1 ≤ L start ≤ L end ≤ 28 (total layers in Llama3.2 3B), we replace layers L start to L end of the ICL model f θ with corresponding layers from f θ * to create a stitched model. This strategy has been used in prior works on localizing information after SFT (Gong et al., 2022;Panigrahi et al., 2023;Wei et al., 2024).
For each level 1 ≤ i ≤ d, we evaluate the stitched models on MLT Π * using the following in-context information:
[STR(π * 1 ), • • • , STR(π * i-1 ), STR(π * i+1 ), • • • , STR(π * d )].
this section cite: ['b1', 'b24', 'b45']

Section: Note here the textual representation of π *
i has been dropped. If a stitched model shows high accuracy on inputs that use π * i for translation but contain only the above in-context information, then the layers selected from f θ * to create the stitched model can be deemed responsible for storing information on π * i in their parameter space. Figure 4 demonstrates that information of all phrasebooks can be localized to a few mutually disjoint layers in f θ * . Moreover, the end of the group of layers where π * i is localized in f θ * marks the start of the layers that begin processing level i phrasebook in the ICL-capable model f θ (e.g. compare the role of layer 17 in Figures 3 and 4). Formal details and additional experiments are in Appendices B.3 and C.2. This suggests that instead of storing phrasebooks as a single chunk in its parameters, the model re-learns each translation step locally to compensate for missing information when rules are dropped. Thus, we conjecture that contextenhanced learning leverages curriculum-text to improve training by localizing learning in the parameter space. We build a surrogate model in Section 5 to show how such localized learning can prove beneficial for faster training.
this section cite: []

Section: Curriculum-text: Detectable from Queries?
Context-enhanced learning for MLT uses the phrasebooks in its curriculum-text. Can rules of the phrasebooks be recovered post-hoc querying the model (Q3) using likelihoodbased methods for detecting training data?
We take an MLT Π * -capable model (with d = 5, n = 10) trained with context-enhanced learning from Section 3.2. Let STR (Π * ) denote the concatenation of all phrasebooks. For each rule in STR (Π * ), which are of the form "a b -> C D" 3 , we measure the probability of the model generating the ground truth tokens "C D" after observing "a b ->". We compute (1) the fraction of cases where the model's top-1 prediction matches the ground truth (greedy decoding) and ( 2) the probability of generating ground truth tokens via random sampling (with temperature set as 1). These are standard tests on textual description, and we would expect high probabilities for the correct tokens.
We also test two adversarial strategies with additional token filtering 4 in generation: (i) setting probability of <THINK> tokens to zero (ii) when querying for a rule in π i with output alphabet A i+1 , setting probability of all tokens outside A i+1 to zero 5 . Note that this corresponds 3 a,b,C,D are generic tokens for presentation 4 See detailed characterization in Appendix C. 3.  5 We did not explore the full spectrum of attacks (e.g. adversarial prompt engineering) and leave that as interesting future work.
this section cite: []

Section: Mathematical Analysis
Having empirically demonstrated the sample efficiency benefits of context-enhanced learning, we now formalize them through a mathematical lens. We first define the sample complexity of an algorithm for learning the task.
Definition 5.1 (informal). Sample complexity for an algorithm to learn MLT Π is defined as the minimum total length of all (possibly repeated) input sequences required by the algorithm to return an MLT Π -capable model f θ .
Analysis of gradient-based learning on a multilayer transformer (let alone a 28-layer model like Llama 3.2-3B) is an open mathematical question. We use our mechanistic findings (i.e., translation layers map onto transformer layers) to propose a surrogate model to think about how transformers learn MLT(d, n). In this surrogate model we demonstrate that learning a task MLT Π via vanilla SFT will require n Ω(d) samples. Then, we prove that, with context-enhanced learning, the surrogate model can learn MLT Π * with a sample complexity of O(poly(n)d log d).
this section cite: []

Section: Surrogate Model (SURR-MLT)
Formalization of the surrogate model, in short SURR-MLT, relies on observations from Section 3.3, which reveal that an ICL-capable model (Definition 2.2) performs the translation task step-by-step, with earlier phrasebook processed by lower layers. This aligns with how the model stores internalized knowledge in layers after context-enhanced learning. Our SURR-MLT represents an idealized and simplified transformer that has already been "pre-conditioned" to solve MLT in this sequential fashion. Using SURR-MLT, we will show the benefits of context-enhanced learning.
Without loss of generality we assume the alphabet sets as A 1 , . . . , A d+1 := A = {1, 2, . . . , n}. SURR-MLT will represent a length-L sequence s i = (s i,1 , . . . , s i,L ) as an embedding matrix V i ∈ R n 2 ×L/2 that uses {v (s i,1 , s i,2 ) , • • • , v (s i,L-1 , s i,L )} as columns. Here, for any 2-tuple (a, b), v(a, b) ∈ R n 2 represents a one-hot vector with 1 at dimension an + b. SURR-MLT operates on embedding matrices, transforming
V 1 → V 2 → • • • → V d+1 .
Each layer i will be primarily defined by two matrices, C i , W i ∈ R n 2 ×n 2 . C i presents a (possibly partial or completely dropped) phrasebook that is provided in-context, and W i is a trainable parameter storing phrasebook information during context-enhanced learningfoot_2 .
Definition 5.2. SURR-MLT with trainable parameters
{W i } d
i=1 and in-context representation {C i } d i=1 , is represented by its operation on an input s 1 as
V d+1 = SURR-MLT {Wi} d i=1 {C i } d i=1 , V 1 , where V i+1 = HardMax(C i + W i ) Shift(V i ), for i ≥ 1,
and V 1 is the embedding matrix for the input string s 1 .
Here Shift represents Circular shift operation and is defined as a Hadamard product on the embedding matrices (details in Definition G.2). HardMax represents hard-max function converting C i + W i to a binary column stochastic matrix. In the following discussion, we show 2 examples where the surrogate model can perfectly represent an ICLcapable model and MLT Π * -capable model.
Case 1 (Representing MLT(d, n)-ICL-capable): The trainable matrices are all 0's as the model hasn't undergone context-enhanced learning. To produce the output for a task MLT Π , SURR-MLT takes phrasebooks into in-context representations by setting each C i as a stochastic matrix Matrix(π i ), which represents rules of π i as one-hot columns (please see Definition G.4 and Lemma G.5).
Case 2 (Representing MLT Π * -capable): For a model that has performed context-enhanced learning on a translation task MLT Π * , no in-context information will be provided to the surrogate model and so {C i } d
i=1 will be all 0s. A MLT Π * -capable-model should contain the phrasebooks as {Matrix(π * i )} d i=1 in its trainable parameters {W i } d i=1 . SURR-MLT as an Ideal Transformer for MLT: The following theorem constructs a transformer that can simulate SURR-MLT. Thus, while our discussions and proofs focus on the surrogate model for simplicity, they remain fully applicable to the transformer architecture.
this section cite: []

Section: Theorem 5.3 (cf Lemma H.4).
There exists a transformer that can simulate SURR-MLT with 2d self-attention and 2d MLP layers with embedding dimension 2n 2 + 2d + 4.
this section cite: []

Section: Sample Complexity for Vanilla SFT
For the surrogate model, vanilla SFT corresponds to always setting in-context representations {C i } d i=1 to 0s when training for MLT Π * . While we use the surrogate model for consistency in our discussion, the argument generalizes to any model learning MLT Π * with vanilla SFT.
Our analysis is built on the Statistical Query (SQ) framework (Kearns, 1998), which measures the difficulty of learning tasks using algorithms that rely on expectation estimates of specific functions to approximate the true solution. Gradientbased methods, such as Stochastic Gradient Descent (SGD), fall under this framework as they compute gradients by estimating expectations of loss functions and their derivatives.
The complexity of learning a task is quantified by the SQ dimension, which measures the number of candidate functions that are pairwise uncorrelated under the input distribution and difficult to distinguish with limited samples. A higher SQ dimension implies a richer hypothesis class, that requires more samples to identify the correct function. We show in the following theorem that the SQ dimension of MLT(d, n) grows exponentially with task parameters.
Theorem 5.4. SQ dimension of MLT(d, n) under uniform input distribution is at least n Ω(d) .
Informally this implies that any algorithm that tries to learn a MLT Π * -capable SURR-MLT with trainable parameters {W i } d i=1 , and in-context information {C i } d i=1 always fixed at 0s, by minimizing loss across samples will require at least n Ω(d) sample complexity.
A corollary is that for vanilla SFT with SGD, sample complexity to learn MLT Π * can be at least n Ω(d) . This is adapted from Edelman et al. (2023), who analyse for sparse parity that has similar SQ dimension (Corollary F.6).
Informal proof for SQ dimension: Our proof extensively analyses the case where number of characters is 2 (lem. F.10), on which we will build proof for general n (lem. F.20). The proof for n = 2 leverages uncorrelations between two randomly selected translation tasks; i.e. we show that for two random set of phrasebooks Π α , Π β , the translation tasks MLT Π α , MLT Π β will have 0 output correlation with probability at least 1 -2 -Ω(d) w.r.t. random choice of Π α , Π β (Lemma F.13). We then show that we can pick exponentially many such random set of phrasebooks for which the translation tasks will be pairwise uncorrelated. This translates to a high SQ dimension.
this section cite: ['b8']

Section: Sample Complexity of Context-Enhanced Learning
Here, we show that context-enhanced learning substantially improves the sample complexity of learning in SURR-MLT. An MLT(d, n)-ICL-capable model gets a set of phrasebooks Π * by setting {Matrix(π * i )} d i=1 as in-context representations {C i } d i=1 . When a curriculum is followed such that a translation rule is dropped from a phrasebook, say π * i , SURR-MLT will set the corresponding column in its in-context representation C i as 0's. Denote the zero-ed out column by C (j) i for some generic column index j, we note that the loss will be low if and only if the corresponding column in the learnable parameters W (j) i exactly matches Matrix(π * i ) (j) . A heuristic search algorithm, that searches among n 2 possibilities for the dropped rule and stores its one-hot representation in the corresponding column in W i , can be used to minimize the loss. By sequentially dropping rules, followed by a search and store process, the algorithm achieves polynomial sample complexity. Theorem 5.5 (Informal; cf Corollary G.19). For any task MLT Π * , there is a heuristic search algorithm paired with a curriculum of iteratively dropping rules from phrasebooks, that can learn a MLT Π * -capable SURR-MLT with sample complexity O(n 6 d log d) with high probability.
The enumerative step in the heuristic search algorithm requires Θ(n 2 ) steps on average, as the algorithm needs to search over Θ(n 2 ) possibilities when a rule is dropped from a phrasebook. Instead, we show that gradient descent requires only a few steps per dropped rule. Dropping a rule sets the respective column in the in-context representation to 0, causing the gradient for the corresponding column in the trainable parameters to strongly align with the missing column. We formally present results for d = 2; due to exponentially growing number of terms to analyse with higher d, we keep the result for general d as a conjecture. Theorem 5.6 (Informal; cf Corollary G.25). When d = 2, there is a gradient descent based algorithm, paired with a curriculum of iteratively dropping a random rule from phrasebooks, that can return MLT Π * -capable SURR-MLT with sample complexity O(n 4 ) with high probability.
While theoretical analysis of GD dynamics beyond d = 2 is challenging, empirically we show that when training with SGD, the trainable parameters for much deeper SURR-MLT can quickly learn the set of phrasebooks in Π * . In Figure 5, we show context-enhanced learning results for a SURR-MLT on a randomly selected set of phrasebooks Π * in MLT(10, 10). Regardless of whether we learn each layer separately with layer-wise SGD or all layers simultaneously with SGD in the surrogate model, the trainable parameters learn the phrasebooks in Π * very quickly. More discussions are deferred to Appendix G.5.
this section cite: []

Section: Insight from SURR-MLT: Gradient Quality
In this section, we show the major difference between context-enhanced learning and vanilla SFT to be the amount of predictive information in gradients for the trainable parameters. Our theoretical analysis in Theorem 5.6 on SURR-MLT primarily shows that when a single translation rule in a phrasebook π * i is dropped, equivalently a column in the in-context representation, say C (j) i , is zero-ed out, the gradient of the corresponding column in trainable parameter W (j) i aligns strongly with the one-hot vector representation of the dropped rule, Matrix(π * i ) (j) . However, this strong alignment heavily relies on the presence of other rules in-context. When more rules are dropped, the gradient signal gets increasingly "inaccurate". We quantitatively characterize such degradation as follows:
Consider an ICL-capable SURR-MLT, we focus on a column (with index j, denoted as superscript) W (j) 1 in the first layer with ground truth Matrix(π * 1 ) (j) . We define gradient prediction accuracy (see formal definition in Definition B.1) for W (j) 1 as the probability that the argmax entry of the negative stochastic batch gradient on W (j) 1 matches the argmax entry of Matrix(π * 1 ) (j) . Intuitively, a higher gradient prediction accuracy will make learning Matrix(π * 1 ) easier. We track the accuracy metric when only C (j) i is zero-ed out as well as when taking more aggressive context dropout schemes.
0.00 0.25 0.50 0.75 1.00 Rule Dropping Rate 0.4 0.6 0.8 1.0 Gradient Prediction Accuracy 10 0 10 1 10 2 10 3 Batch size 2 4 6 8 10 Max. Phrasebook Index 0.1 0.2 0.3 0.4 0.5 Rule Dropping Rate In Figure 6, we report the gradient prediction accuracy with dropout schemes involving (1) multiple rules dropped out from the first phrasebook and (2) rules dropped from multiple phrasebooks. We can see that higher dropping rates significantly degrade the gradient prediction accuracy, highlighting the necessity of proper in-context information in order to obtain the optimization benefit. More details on the metric and experiments are deferred to Appendix B.1. We note that finding above is based on observations on the simplified surrogate model. A quantitative characterization of the optimization benefit for context-enhanced learning in the LLM regime is left as an interesting future work.
this section cite: []

Section: Related Works
In this section, we discuss related works to this paper. More related works regarding (1) compositional and OOD generalization and (2) mechanistic understanding of transformers are discussed more extensively in Appendix D.
this section cite: []

Section: Learning using Privileged Information (LUPI)
LUPI was formally introduced by Vapnik & Vashist (2009) in kernel SVMs, and it is heavily related to concepts of Learning with Side Information (Kuusela & Ocone, 2002;Jonschkowski et al., 2015;Zhang et al., 2018). Primarily, both concepts refer to training a model with additional information that could help training but may not be available at test time. This framework has been extended theoretically for classification tasks (e.g. Pechyony & Vapnik (2010); Momeni et al. (2018)) and has been used to explain benefits of knowledge distillation (Vapnik et al., 2015;Lopez-Paz et al., 2015). To name a few applications, this concept has been heavily studied for improving boosting for classification tasks (Chen et al., 2012), visual and video encoders (Hoffman et al., 2016;Cheng et al., 2020;Xu et al., 2017), human preference predictions (Farias & Li, 2019), multiagent games (Sessa et al., 2020), speech recognition (Synnaeve et al., 2014), and medical recognition (Ceccarelli & Maratea, 2008;Sabeti et al., 2020).
While LUPI has rich application in classification, extending LUPI to LLMs introduces unique challenges due to their auto-regressive training. Such a concept raises questions on whether applying auto-regressive loss on the additional information is necessary to get the benefits of additional supervision information in context, and how the additional information changes the training behavior of LLMs. Hence, our work is a nontrivial generalization of this framework to LLMs that connects to their in-context learning strengths.
this section cite: ['b42', 'b11', 'b6', 'b55', 'b26', 'b20', 'b43', 'b16', 'b5', 'b48', 'b33', 'b39', 'b32']

Section: In-context learning and memorization
Recent works have studied emergence of ICL and competition with in-weights learning (IWL) (equivalent to knowl-edge memorization) during pre-training of language models (Chan et al., 2022;Reddy, 2024;Singh et al., 2023;2024;Nguyen & Reddy, 2024). These works show that data distribution properties affect the behavior of the model during training. Our work can be thought of as contemporary to the works above, where we show that strong ICL capabilities can be utilized for improving knowledge on a task by context-enhanced learning.
Benefits of in-context learning: In-context learning has been primarily studied in the context of few-shot prompting of large language models. Better supervision with in-context supervision can help in improved performance (e.g. some representative works (Arora et al., 2022;Si et al., 2022;Wu et al., 2022;Lu et al., 2021;Su et al., 2022)), OOD generalization and factuality (reduced hallucination) (Yang et al., 2023;Dhuliawala et al., 2023;Chen et al., 2023;Didolkar et al., 2024), and more structured latent representations (Park et al., 2024) for large language models. On the other hand, we show that improved supervision with in-context supervision can also help a model learn faster in SFT, while seemingly not leaking the in-context information in its output probabilities.
this section cite: ['b31', 'b35', 'b31', 'b23', 'b34', 'b47', 'b19', 'b39', 'b49', 'b49', 'b25']

Section: Discussion, Limitations, and Future work
Some experimental works have implicitly used the notion of context-enhanced learning but the current paper formalized this notion for auto-regressive models and showed, using MLT, that this form of learning can be exponentially more sample-efficient than standard SFT. At the end of training it is hard to recover the in-context information seen during training from the model's output probabilities. We note that this finding appears to have implications about copyright law (e.g., whether or not LLM training amounts to "transformative use" of text (Carlini et al., 2021;2022;Karamolegkou et al., 2023)) whose further study is left for future work.
Our experiments focus on a synthetic MLT task for a few reasons: (1) to ensure that the task is absent from LLM pre-training, which allows precise quantification of benefits of context-enhanced learning, including not revealing the curriculum text at inference time. (2) the task is too difficult (at least for Llama 3.2 3B model) to learn via vanilla SFT, but is learnable via context-enhanced learning. Extending these findings to real-world complicated tasks (e.g., in math and coding) is left for future work.
Our convergence analysis for context-enhanced learning relies on a surrogate model, and extending it to an actual transformer remains an open challenge for theory of deep learning. Extending formalization of context-enhanced learning to explore LLM training in multi-agent settings-where models collaborate and learn from each other to discover novel concepts-would be an exciting avenue for future research.
this section cite: ['b7']

Section: Impact statement
We formulate a basic notion "context-enhanced learning", and study how it is different from usual learning. One consequence of our study is that enhancing the context with good quality data can enhance the quality of the learner. This enhancement could be used with privileged information and it appears that the training can be done in such a way that the model does not leak this privileged information. This can be seen as enhancing privacy, since there is a lower chance of leaking privileged training data.
The flip side is that it suggests -albeit in very toy and synthetic setting, with full study left for future work -that model training could use off-limits data (albeit with no gradient updates on it) and this use might not be detectable from querying the trained model. But this is hypothetical at this point since the paper concerns a very toy setting.
this section cite: []

Section: Scope of Notation Symbol Description

this section cite: []

Section: General Notations
A A set A A matrix A (j)
The j-th column of matrix A e k One-hot embedding vector with 1 at dimension k
Language Modeling X All possible text strings (input space for a causal LM) Y All possible distribution over texts (output space for a causal LM) f θ Causal LM parameterized by θ g Language task mapping inputs x ∈ X g ⊂ X to a distribution Y ∈ Y ℓ auto Auto-regressive cross entropy loss CURR g (x, t) In-context curriculum for learning task g with input x at step t MLT Translation Task d Depth of translation task n Number of characters in each alphabet A An alphabet set s A sequence in alphabet A π A phrasebook between 2-tuples in two alphabets, e.g. (π 1 : A 2 1 → A 2 2 ). B π Space of all possible phrasebook on A 2 1 → A 2 2 Π A set of phrasebooks {π i } defining a translation task MLT Π Translation task with a set of phrasebooks Π MLT(d, n) Family of translation tasks of depth d and n characters STR (π) Descriptive text for a phrasebook π (see Section 2.2) Surrogate Model V i Embedding matrix representing sequence s i (Definition G.1) C i In-context information matrix for level i W i Trainable parameter matrix for level i P i Effective translation matrix for level i (Definition G.6) HardMax Column-wise hard-max function Matrix(π) Matrix representation of a phrasebook π (Definition G.4) SURR-MLT {Wi} d i=1 Surrogate model parameterized by W i 's (Definition G.7)
this section cite: []

Section: B. Deferred definitions, and experimental details from the main paper B.1. Gradient Prediction Accuracy
Our theoretical analysis in Theorem 5.6 was built on the fact that when a translation rule in a phrasebook is dropped by zeroing out a column in an in-context representation C i , the gradient for the corresponding column in W i points to the direction of the dropped rule.
On the other hand, we show that when multiple rules are simultaneously dropped from the phrasebooks, the gradients for the trainable parameters become increasingly noisy. To quantify this degradation, we compute gradient for each column of the trainable parameters for an ICL-capable model, when the corresponding rule is dropped from phrasebooks by zeroing out the relevant column in the in-context representations. We then compute whether the computed gradient points to the right rule. By progressively increasing the number of simultaneously dropped rules, we measure the resulting degradation in the accuracy of the gradient's predictions.
More formally, denote RANDOM-DROP as an operation that takes in column dropping rates per layer p 1 , • • • , p d , set of phrasebooks Π * , and returns in-context representations
{C i } d i=1 , such that C (j) i = Matrix (π * i ) (j) (jth
E {Ci} d i=1 =RANDOM-DROP(p1,••• ,p d ,Π * ) E j∈[1,n 2 ]|C (j) 1 =0 I HardMax -∇ W (j) 1 L = (Matrix (π * i )) (j) L = E s1 ℓ SURR-MLT {Wi} d i=1 {C i } d i=1 , V 1 , MLT Π * (s 1 ) ,
where ℓ, adapted from Section 2.1, computes cross-entropy loss on the predicted output embeddings of surrogate model using true output string and I denotes the indicator function.
The above definition computes gradients on the expected loss of the model. We primarily focus on predicting the trainable parameters of the first layer, i.e. W 1 , as that is the deepest layer in the surrogate model and intuitively should suffer the most with noise accumulation from dropped rules. On the other hand, we can further adapt the definition to compute the accuracy for batched gradients, where the gradients are computed using average loss on a randomly sampled batch of input sequences.
In Figure 6, we report the predictive accuracy of gradient for an MLT(d, n)-ICL-capable model, and its behavior with varying batch size and the column dropping rates. We report for two cases, one where column dropping rates is non-zero only for the first phrasebook, and one where we increase the number of phrasebooks for which rules are independently and uniformly dropped. In both cases,
• Increased column dropping rates leads to noisier gradients and reduced prediction accuracy.
• Larger batch sizes improve gradient accuracy but cannot fully compensate for high dropout rates.
• Dropping rules from multiple phrasebooks significantly degrades gradient prediction accuracy.
this section cite: []

Section: B.2. Hidden Representations of a Model
A transformer f θ with embedding dimension p and K layers takes any input sequence x, say of length L, converts to an embedding matrix H 1 ∈ R L×p , and modifies the embeddings using a succession of K transformer layers; which we will denote by f
θ , f
θ , • • • , f (K) θ(2)
. We refer to the hidden representations for the input x, with embedding matrix H 1 , as the output of the model after every layer. We will denote them as H i+1 ∈ R L×p for the output of layer f (i) θ . That is,
H i+1 = f (i) θ • • • • • f (2) θ • f (1) θ (H 1 ) , for all i ≥ 1. ℓ 2 -norm in change in hidden representation with perturbation in in-context information For an MLT(d, n)- ICL-capable model, we supply in-context information for the textual description of a set of phrasebooks Π as STR(Π) = [STR(π 1 ), • • • , STR(π d )].
Our inputs to the transformer for an input string s 1 will be of the form [STR (Π) , s 1 , <THINK>,..., s d+1 ], where s d+1 = MLT Π (s 1 ). By the definition of hidden representations, H 2 , • • • , H K+1 will denote the output of the transformer layers for this input string. However, we will be only interested in the hidden representations for the tokens involved in the tokens for <THINK>,..., s d+1 ; and we will refer to the corresponding subsets of
H 2 , • • • , H K+1 that represent these specific tokens as V 2 , • • • , V K+1 .
Now, suppose we randomly take a phrasebook π i in Π and change to a random phrasebook πi . The corresponding textual description that will augment the context for an input string will then be
[STR(π 1 ), • • • , STR(π i-1 ), STR(π i ), STR(π i+1 ), • • • , STR(π d )]. If Ṽ2 , • • • , ṼK+1
now denote the hidden representations that represent the tokens for <THINK>,..., s d+1 , then the ℓ 2 -norm in the change of the hidden representation after layer j (for any 1 ≤ j ≤ K) with the perturbation in Π will be given by Ṽj -V j 2 .
this section cite: []

Section: B.3. Definition of a "stitched" model
We reuse notations from Appendix B.2. Suppose we have an MLT(d, n)-ICL-capable model f θ and an MLT Π * -capable model f θ * . Their corresponding transformer layers are denoted by f
(1) θ , f (2) θ , • • • , f (K) θ and f (1) θ * , f (2) θ * , • • • , f (K) θ * .
Each model takes in an input sequence and processes them with their K transformer layers.
Formally, we will write for the ICL-capable model. It takes in input sequence x, and converts to an embedding matrix, say H 1 , and the output after the Klayers are given by:
H K+1 = f (d) θ • • • • • f (2) θ • f (1) θ (H 1 ) .
Process of "stitching": The process of stitching takes in two parameters L start and L end and replaces all layers from L start to L end in f θ with the corresponding layers in f θ * to give a "stitched" model, say f θ,θ * ,Lstart,Lend . The output of the "stitched" model f θ,θ * ,Lstart,Lend on an input sequence x will be given by
H K+1 = f (d) θ • • • • • f (Lend+1) θ • f (Lend) θ * • • • • f (Lstart) θ * Layers are replaced by layers from f θ * •f (Lstart-1) θ • f (2
) θ • f(1)
θ (H 1 ) .
this section cite: []

Section: B.4. Pipeline on CoT Internalization
We randomly sample M sets of phrasebooks π 1 , . . . , π M not equal to Π * . For each set of phrasebooks π i , we randomly sample a single input sequence s 1 and compute all the intermediate translation steps s 2 , . . . , s d+1 . We first train the model to do robust explicit CoT by auto-regressive training on sequences [STR(π i ), s 1 , s 2 , . . . , s d , s d+1 ] with loss computed over s 2 , . . . , s d+1 . Then we follow common CoT internalization strategies (Deng et al., 2024;Hao et al., 2024;Yu et al., 2024;Su et al., 2024) and gradually replace the intermediate sequences by <THINK> tokens in training. After all intermediate sequences have been replaced, the model has low loss on s d+1 with input [STR(Π 1 ), s 1 , <THINK>, . . . , <THINK>, s d+1 ], satisfying Definition 2.2. Since we only sample on sequence per set of phrasebooks, there is little memorization on particular phrasebooks.
Details on training hyperparameters: We use M = 3 × 10 5 random sets of phrasebooks with length between 20 and 40, for getting MLT(5, 8)-ICL-capable model, and M = 10 6 random sets of phrasebooks with length between 20 and 40, for getting MLT(5, 10)-ICL-capable model. We use cosine learning rate schedule (Loshchilov & Hutter, 2016), with peak
On the Power of Context-Enhanced Learning in LLMs learning rate 10 -4 and a 6% warmup phase, where learning rate is linearly increased from 0 to the peak. We use AdamW optimizer (Loshchilov & Hutter, 2019) with weight decay fixed at 10 -4 . We use a batch size of 64 for training.
CoT internalization curriculum: For the first 10% fraction of training, we train the model with explicit CoT tokens that contain the intermediate steps in translation. Then between 10% to 60% fractions of training, CoT tokens are gradually replaced by <THINK> tokens, with the rate of replacement increasing linearly from 0% to 100%. We follow a deterministic first-to-last order for replacing CoT tokens; earlier CoT tokens are replaced first with <THINK> tokens. After that, the model is trained with the <THINK> CoT tokens till the end of training.
this section cite: ['b3', 'b53', 'b38', 'b17', 'b18']

Section: B.5. Experiment configuration for context-enhanced learning
For context-enhanced learning, we create supervised datasets D Π * of different sizes; each containing between 10 4 to 10 6 samples. When performing Annealing Dropout or Fixed Dropout, at each step of training, we randomly perform dropout on the rules of all phrasebooks or apply dropout to the rules of a randomly sampled phrasebook to define curriculum-text.
That is, if π * 1 , . . . , π * 5 represent the phrasebooks, then at each step of training, we either randomly drop rules uniformly from all of π * 1 , . . . , π * 5 , or just drop from one of the phrasebooks randomly selected from π * 1 , . . . , π * 5 , while keeping the rules of all other phrasebooks intact, to create curriculum-text.
Hyperparameters: Training hyperparameters are set equal to the optimization hyperparameters used in preparation of ICL-capable training phase (Appendix B.4), except we set weight decay to 0 in all experiments. We report the performance of the trained model after single epoch of training on each D Π * and plot against the size of the dataset in Figure 2.
this section cite: []

Section: C. Additional Experiments Results

this section cite: []

Section: C.1. Selective Internalization of Context
In this experiment, we test whether the model can internalize rules that don't incur an increase in loss when dropped during training. To do so, we ablate on Annealing Dropout. We select a phrasebook and create 2 splits of rules in the phrasebook; one set of rules will be utilized by training samples for performing the translation task (which we call the training split), while other set of rules will appear in curriculum-text during training but never utilized for the translation task (which we call the heldout split).
At test time, we measure the performance of the trained model on 2 sets of evaluation examples, one that only use rules from the training split for their translation (equal to training distribution), and other that uses rules only from the heldout split for their translation (different from training distribution). The model is being measured without any phrasebooks information at evaluation. We conduct the above experiment for each phrasebook, i.e. we create 5 sets of experiments where we only create heldout split for one specific level of phrasebook. In Figure 7, we show that the model fails to perform any translation that use the rules from the held-out split in all settings. This shows that the model only internalizes those rules that are important for the translation task for the training samples, and which incurs an increase in training loss when dropped.
this section cite: []

Section: C.2. Mechanistic Insights
Here we provide additional figures corresponding to Figure 3 and Figure 4, but in more settings (n = 10 vs n = 8, Annealing Dropout vs Fixed Dropout).  1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (a) D Π * with 10000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (b) D Π * with 25000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (c) D Π * with 50000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (d) D Π * with 100000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (e) D Π * with 250000 samples  4). We observe that phrasebooks are progressively internalized with the number of training samples available during context-enhanced learning; later phrasebooks are internalized with fewer samples than the earlier ones. We observe similar localization patterns across layers from different phrasebooks across the models, however, we also observe that the localization patterns get increasingly sparser as training continues.
1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (a) D Π * with 25000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (b) D Π * with 50000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (c) D Π * with 100000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (d) D Π * with 250000 samples 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 L start Dropping STR( * 1 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 2 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 3 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 4 ) 1 4 7 10 13 16 19 22 25 28 L end 1 4 7 10 13 16 19 22 25 28 Dropping STR( * 5 ) 0.0 0.2 0.4 0.6 0.8 1.0 Evaluation Accuracy (e) D Π * with 500000 samples
this section cite: []

Section: C.3. (Non)Verbatim Memorization of Phrasebook Rules
In this subsection, we provide a more detailed explanation for the evaluation on the feasibility of recovering phrasebook rules from models trained with context-enhanced learning with phrasebook excerpts in context.
Given a MLT Π * -capable model f θ * trained with context-enhanced learning where phrasebook rules from STR(Π * ) are provided within the context during training, we test if the model retains explicit memory of the textual format of rules. Namely, for each of the phrasebook rule of the form "a b -> C D" in STR(Π * ), whether it can complete the ground truth output tokens "C D" providing "a b ->" and additional context of the phrasebook STR(Π * ) before "a b -> C D". We conduct the test by computing the forward pass through f θ * with STR(Π * ) as the input (see exact input format in Figure 17).
Now we formally define the query strategies and metrics we used for creating Table 2. Suppose there are h unique tokens and STR(Π * ) contains L tokens, let S ∈ R h×L denote the corresponding output logit score matrix when we pass STR(Π * ) into f θ * . For a translation rule with ground truth output tokens of index a 1 , a 2 ∈ A i+1 ⊂ [h], let S (k) and S (k+1) ∈ R h denote the logit vector corresponding to predicting these two entries.
If we are doing greedy decoding, then we will recover the correct phrasebook rule if and only if we greedily select both a 1 from S (k) and a 2 from S (k+1) , so the probability of correctly recovering (a 1 , a 2 ) conditioned on S is
P [Greedy-Recover(a 1 , a 2 )] = 1 arg max i∈[h] S (k) = a 1 • 1 arg max i∈[h] S (k+1) = a 2 .
Meanwhile, if we apply random sampling with softmax temperature of 1, the probability of correctly recovering (a 1 , a 2 ) conditioned on S is just then
P [Sampling-Recover(a 1 , a 2 )] =   exp S (k) a1 i∈[h] exp S (k) i   •   exp S (k+1) a1 i∈[h] exp S (k+1) i   .
Recall from Section 4 that we have also introduced two stronger adversaries with additional token filtering when doing decoding: (1) setting probability of <THINK> tokens to zero (2) when querying for a rule in π i with output alphabet A i+1 , setting probability of all tokens outside A i+1 to zero.
Denoting the token index of <THINK> as a <THINK> , then filter 1 corresponds to
P [Greedy-Filter1-Recover(a 1 , a 2 )] = 1 arg max i∈[h]\{a<THINK>} S (k) = a 1 • 1 arg max i∈[h]\{a<THINK>} S (k+1) = a 2 , P [Sampling-Filter1-Recover(a 1 , a 2 )] =   exp S (k) a1 i∈[h]\{a<THINK>} exp S (k+1) i   •   exp S (k) a1 i∈[h]\{a<THINK>} exp S (k+1) i   .
Similarly, for filter 2, the probabilities are
P [Greedy-Filter2-Recover(a 1 , a 2 )] = 1 arg max i∈Ai+1 S (k) = a 1 • 1 arg max i∈Ai+1 S (k+1) = a 2 , P [Sampling-Filter2-Recover(a 1 , a 2 )] =   exp S (k) a1 i∈Ai+1 exp S (k+1) i   •   exp S (k) a1 i∈Ai+1 exp S (k+1) i   .
Note that the second filter is a very strong adversarial assumption which assumes the user already has side information on the set of tokens contained in alphabet A i and A i+1 . To compute the final statistics, we sample 20 permutations of STR(Π * ) for forward passes and compute the mean of the above statistics over all atomic phrasebook rules appearing in the context. That is 1280 entries for each phrasebook in the case of n = 8 and 2000 entries for each phrasebook in the case of n = 10.  Here we report the query success rate for a variaty of models trained with context-enhanced learning using different curriculum on different datasets sizes. All models reaches nearly perfect test accuracy when no context is provided (see Figure 2), i.e., the in context phrasebook rules played significant role in the learning of the models.
For all runs, we can see that it is nearly impossible (with recovery probability < 0.1% in most cases) to recover the correct phrasebook rules for hidden steps (π 1 , . . . , π 4 ) even when we provide the correct partial phrasebooks in context (see columns corresponding to Query Method "Base"). Ruling out <THINK> token when sampling also did not significantly increase the recovery rate. We note that the phrasebook knowledge are not memorized in an completely undetectable manner, as the strongest token filtering gives non-random probability of outputting the correct target 2-tuple. However the probability is still very low (< 3%), which can be considered as negligible to recover the full correct phrasebooks STR(Π * ).
this section cite: []

Section: D. Additional Related Works
Differences with Masked Language Modeling (MLM) and Language Infilling MLM models like BERT, RoBERTa, and T5 (Kenton & Toutanova, 2019;Liu et al., 2019;Raffel et al., 2020) train models by either masking or removing tokens from a sequence and compute loss on the model's prediction on the missing tokens. This concept has been adapted for training auto-regressive models via language infilling task (Bavarian et al., 2022;Li et al., 2022;Donahue et al., 2020;Li et al., 2022). The primary difference from these works is that context-enhanced learning does not take loss on the context tokens when they are removed from our curriculum-text.
this section cite: ['b9', 'b15', 'b29', 'b12', 'b12']

Section: Compositional and OOD generalization
Measuring generalization for a transformer beyond training distribution has been a study of interest in many prior works. OOD generalization is measured by training a transformer on simpler examples and measuring its performance on harder ones. Prominent studies include length generalization, informally defined as the ability of the model to reason longer than what it has been trained on (Zhou et al., 2023;Anil et al., 2022), and compositional generalization on concepts, defined as the ability of the model to reason on composition of the concepts that it has seen during training (Press et al., 2022;Allen-Zhu & Li, 2023b;Ramesh et al., 2023;Yu et al., 2023;Zhao et al., 2024;Wang et al., 2024;Yang et al., 2024). Our experiments on Fixed Dropout in Section 3, where we train with 20% dropout on the phrasebooks information but measure performance with 100% dropout at test time, measures compositional OOD generalization behavior of the language model. The results show that the model internalizes the rules from the phrasebooks in an atomic way, and re-compose them together as necessary at test time.
Mechanistic behavior of transformers with synthetic datasets:
Our work builds on a growing body of research exploring the behavior of transformers trained on synthetic datasets. Prior studies have examined tasks such as modular addition (Nanda et al., 2023;Zhong et al., 2023), context-free grammars (Zhao et al., 2023;Allen-Zhu & Li, 2023a), regular and n-gram languages (Bhattamishra et al., 2020;Yao et al., 2021;Akyürek et al., 2024;Li et al., 2023), and synthetic article-style datasets (Allen-Zhu & Li, 2023b;2024;Eldan & Li, 2023). While our work is structurally similar to these studies, it investigates mechanistic study on context-enhanced learning that has not been explored in previous works.
this section cite: ['b58', 'b28', 'b30', 'b52', 'b56', 'b44', 'b50', 'b22', 'b57', 'b56', 'b51', 'b13', 'b31']

Section: E. Properties of MLT
MLT(d, n) is defined by the phrasebooks π 1 , • • • , π d at each of its translation layers. We use B π as all possible set of bijective maps that can be used to define the phrasebooks. We will use variable π to refer to an arbitrary bijective map from the set B π . For simplicity of proof, we will refer to any alphabet set A of size n as {0, 1, • • • , n -1}.
Here, we formally mention some of the properties of MLT(d, n).
this section cite: []

Section: Lemma E.1 (Invertibility of sequence translation).
For any level i ∈ [d], fixing {π i , π i+1 , . . . , π d } gives a bijection between s i and s d+1 .
Proof. All of the four operations involved in the mapping from s i to s i+1 are invertible.
Structure of the mappings: The mappings π i are selected as random bijective maps between 2-tuples of characters in A 2 i and 2-tuples of characters in A 2 i+1 . A combinatorial argument can then give the number of such possible phrasebooks to be n 2 !. Importance of Circular shift: The composition of the d random bijective maps can be demonstrated to result in another random bijective map. Consequently, without the Circular shift, each character in the output sequence s d+1 depends on only two characters from the input sequence s 1 via a shared random map across all the 2-tuples.
Incorporating Circular shift on the other hand enables each character in the output sequence to depend on 2d characters from the input sequence. This is because the character positions are shifted to the right at each step, causing the input 2-tuples to the bijective map to also shift to the right at each stage. As we will elaborate later, incorporating Circular shift increases the required number of training samples to learn the set of phrasebooks from input and label pairs to n Ω(d) , whereas without Circular shift, this requirement is only O(n 2 ).
Representing the mappings on 2-tuples in-context: Each map can be defined using O(n 2 ) characters, as it can be simply defined by the n 2 relations each connecting 2 unique random 2-tuples from their corresponding alphabet sets. Thus, defining d maps in-context will require O(n 2 d) characters in curriculum-text. In contrast, describing a completely random bijective mapping that maps d-tuples of characters in input sequence to a character in output sequence will require Ω(d n d ) bits. Thus, MLT with d translation steps involving random mappings on 2-tuples and Circular shift helps define a mapping where each character in the output sequence can depend on d character in the input sequence, and the set of phrasebooks can be described using O(n 2 d) characters in curriculum-text. F. Lower Bound: Hardness of Learning MLT(d, n) without Context F.1. Brief introduction to SQ framework Statistical query (SQ) bounds : The statistical query (SQ) framework measures the computational hardness of learning a task in the presence of noise. It measures the hardness of learning a task by the number of statistical queries needed by a learning algorithm to learn the true function. Statistical queries are defined by some polynomially-computable property Q of labeled instances and a tolerance parameter τ ∈ [0, 1] over (x, y) ∼ D where D is the data distribution. For a query, the algorithm receives a response from the oracle within τ error of the true value. The statistical dimension, or SQ-dim, is measured in terms of the number of functions in the hypothesis class that the learning algorithm needs to distinguish and the number of queries necessary to do the same. Correlation between two functions is used to define the statistical query dimension. Definition F.1. Correlation of two functions f 1 , f 2 on a domain X with respect to a distribution D is given by
Correlation(f 1 , f 2 , D) := Pr x∈D [f 1 (x) = f 2 (x)] -Pr x∈D [f 1 (x) ̸ = f 2 (x)] . For functions f 1 , f 2 : X → {0, 1}, the above definition is also equivalent to Correlation(f 1 , f 2 , D) := |1 -2E x∈D [f 1 (x) ⊕ f 2 (x)]| . Remark F.2. Two functions f 1 , f 2 : X → {0, 1} are said to be uncorrelated w.r.t. D if Pr x∈D [f 1 (x) = f 2 (x)] = Pr x∈D [f 1 (x) ̸ = f 2 (x)] (alternately) E x∈D [f 1 (x) ⊕ f 2 (x)] = 1 2 .
On the other hand, if
P r x∈D [f 1 (x) = f 2 (x)] = 1 (or 0) (alternately) E x∈D [f 1 (x) ⊕ f 2 (x)] = 0 (or 1), then Correlation(f 1 , f 2 , D) = 1.
We take the following formal definitions of SQ-dim and its relation to computational hardness in the SQ framework from (Blum et al., 1994).
Definition F.3 (Definition 2 in Blum et al. ( 1994)). For a function class F of boolean functions over {0, 1} n and D a distribution over {0, 1} n , SQ-dim(F, D), the statistical query dimension of F with respect to D, is defined to be the largest natural number µ such that F contains µ functions f 1 , • • • , f µ with the property that for all i ̸ = j we have:
Correlation(f i , f j , D) := Pr x∈D [f i = f j ] -Pr x∈D [f i ̸ = f j ] ≤ 1 µ 3 .
Theorem F.4 (Theorem 12 in Blum et al. (1994)). Let F be a class of functions {0, 1} n and D a distribution such that SQ-dim(F, D) ≥ µ ≥ 16. Then if all queries are made with a tolerance of atleast 1 µ 1/3 , at least µ 1/3 /2 queries are required to learn F with error less than 1/2 -1/µ 3 in the statistical query model.
this section cite: []

Section: F.2. Lower bound lemma
Here, we mention the 2 main theorems that study the SQ dimension of the MLT task at hand. The first theorem shows the SQ dimension bound when number of characters n = 2, which is then adapted to get the SQ dimension bound for general n. Proofs of both the theorems are given in Appendix F.4 and Appendix F.5 respectively. Theorem F.10 (SQ dimension for n = 2). The family of translation task MLT(d, 2) on input distribution U({0, 1} 2d ) has statistical query dimension SQ-dim(MLT(d, 2)) atleast 2 Ω(d) .
Theorem F.20 (SQ dimension for general n). For the translation task MLT(d, n) that has depth d and n characters per level, the statistical query dimension SQ-dim(MLT(d, n)) is atleast n Ω(d) .
Notations: We will require the following notations for proving the above theorems. First, we will denote a 2-tuple that contains arbitrary characters a and b as (a, b). For a bijective map π, π((a, b)) i will represent ith character in the output of π on any 2-tuple input (a, b) and any i ∈ {1, 2}. Similarly, for any set of phrasebooks Π, we will use MLT Π (s 1 ) i to denote the ith character in the output of MLT Π on any L length input sequence s 1 and any i ∈ [1, L].
Recall that for any input sequence s 1 , the output of the ith level of a MLT task will be denoted as s i . Furthermore, to denote the jth character (arbitrary) in the sequence s i , we will use s i,j .
this section cite: []

Section: F.2.1. BOUNDS FOR SGD
The following corollary measures the sample complexity needed to learn MLT Π * by SGD. It has been adapted from proposition 3 in Edelman et al. (2023), who study the sample complexity for learning d-sparse parity task on n length sequences, whose SQ dimension is n d = Θ(n d ). We simply state the corollary without specifying the proof. Loss function and SGD updates: Consider training of a model f θ with r parameters that is trained with mean squared error, i.e. ℓ auto (f θ ([CURR g (x, t), x, y]), y) = ∥f θ ([CURR g (x, t), x, y]) -y∥ 2 7 . The empirical loss on a batch S of batch size B from the supervised dataset D Π * will be denoted by L
S (f θ , MLT Π * ) = E (x,y)∼S ℓ auto (f θ ([CURR g (x, t), x, y]), y) = ∥f θ ([CURR g (x, t), x, y]) -y∥ 2 ;
the population loss will be denoted by L U ({0,1} L ) (f θ , MLT Π * ). SGD updates are of the form:
θ t+1 = θ t -η t (∇ θ L St (f θt , MLT Π * ) + R(θ t ) + ζ t ).
for some sample S t , step size η t , regularizer R(•), and adversarial noise ζ t ∈ [-τ, τ ] r . For simplicity, we assume the gradient ∇ θ L St (θ t ) is bounded on all parameters θ in parameter space of the model.
Fake trajectory: Suppose 0 denote the constant function that maps all inputs to 0. Then, with the contemporary losses L S (f θ , 0) and L U ({0,1} L ) (f θ , 0) that computes difference from this constant function, consider the following trajectory θ1 , • • • , θt , • • • starting from the same initiation θ 0 :
θt+1 = θt -η t (∇ θ L St (f θt , 0) + R( θt )). Assumption F.5. For all t, suppose ∇ θ L U ({0,1} L ) (f θt , 0) -∇ θ L St (f θt , 0) 2 ≤ τ /2.
Corollary F.6 (Sample complexity bounds for SGD). Fix an initialization θ 0 such that f θ0 is statistically independent (correlation 0) from all possible MLT Π * . Under this assumption, if LT B τ 2 ≤ n Ω(d) /r, then there exists at least one task or
Correlation(π α (•) 1 , π β (•) 2 , U({0, 1} 2 )) = 1. Correlation(π α (•) 2 , π β (•) 1 , U({0, 1} 2 )) = 1.
Other cases are not possible, i.e. for any i ∈ {1, 2}, both Correlation(π α (•) i , π β (•) 1 , U({0, 1} 2 )) = 1 and Correlation(π α (•) i , π β (•) 2 , U({0, 1} 2 )) = 1 can't hold true.
Remark F.9. Implication of Lemma F.22 is as follows: With probability at most 1 3 , output of two random maps can stay correlated at either one output character pair or both pairs of output characters. This would suggest that the probability of the output of MLT under two random set of phrasebooks staying correlated should decay exponentially with the depth of the task.
this section cite: []

Section: F.4. Proof for Statistical dimension lower bound for n = 2
We repeat the lemma of interest for presentation.
Theorem F.10 (SQ dimension for n = 2). The family of translation task MLT(d, 2) on input distribution U({0, 1} 2d ) has statistical query dimension SQ-dim(MLT(d, 2)) atleast 2 Ω(d) .
Proof. We narrow our argument to the first character output of the task. The proof goes through 2 major steps.
1. First, we show that two random instances of MLT(d, 2), defined by 2 set of phrasebooks {π α 1 , • • • , π α d } and {π β 1 , • • • , π β d }, will be uncorrelated with probability at least 1 -(1/3)(7/9) d-1 w.r.t. the selection of the random set of phrasebooks. The lemma is formally given in Lemma F.13.
2. Then, we can apply Lovász local lemma (Theorem F.11) to show that we can create 2 Ω(d) instances of MLT(d, 2) that will have zero pairwise correlation of their output (Corollary F.12).
By the definition of SQ-dim from Definition F.3, the above observations suggest that SQ-dim(MLT(d, 2)) ≥ 2 Ω(d) .
Theorem F.11 (Lovász local lemma, theorem 1.5 in Spencer (1977)). Let A 1 , • • • , A k be events in some probability space with Pr(
A i ) ≤ p, 1 ≤ i ≤ k, such that each event is dependent on atmost k 0 other events. If ep(k 0 + 1) < 1, then Pr(¬A 1 ∧ ¬A 2 ∧ ¬A 3 • • • ∧ ¬A k ) > 0.
Corollary F.12 (Number of uncorrelated instances of MLT(d, 2)). There exists a set of 2 Ω(d) instances in MLT(d, 2) that are pairwise uncorrelated to each other on input distribution U({0, 1} 2d ).
Proof. Suppose Π 1 , • • • , Π k represent k randomly sampled set of phrasebooks. For each 1 ≤ i < j ≤ k, we will denote event A ij as the event that the outputs of MLT Πi and MLT Πj are uncorrelated on input distribution U({0, 1} 2d ). This happens with probability p = 1 -(1/3)(7/9) d-1 from Lemma F.13. Because each event can atmost depend on atmost k(k + 1)/2 < k 2 events (total number of events), by Theorem F.11, if
ep(k 2 + 1) < 1,
then there exists a list of such set of phrasebooks which are pairwise uncorrelated w.r.t. the outputs of their corresponding translation tasks on input distribution U({0, 1} 2d ). Solving the above for k, we can set k = ((ep) -1 -1) 1/2 = 2 Ω(d) .
this section cite: ['b37']

Section: F.4.1. AUXILIARY LEMMAS
Here, we will prove the following primary lemma, that is used to prove Theorem F.10.
Lemma F.13. With probability at least 1 -1 3 7 9
d-1 w.r.t. random map selection, the following holds true for 2
sets of random phrasebooks Π α = {π α 1 , • • • , π α d } and Π β = {π β 1 , • • • , π β d }: Correlation(MLT Π α ={π α ℓ } d ℓ=1 (•) 1 , MLT Π β ={π β ℓ } d ℓ=1 (•) 1 , U({0, 1} 2d )) = 0.
Proof. We first dive into the dependencies between characters in input and output sequences in the translation process. In Lemma F.15, we show that at any level 1 ≤ ℓ ≤ d of MLT Π with phrasebooks Π = {π ℓ } d ℓ=1 , the following relation holds true on an input s 1 ∈ {0, 1} 2d :
(s ℓ+1,1 , s ℓ+1,2 ) = π ℓ ((s ℓ+1,2 , s ℓ+1,3 )).
We will use superscripts α and β to differentiate the intermediate outputs of MLT when the phrasebooks are set as
Π α = {π α 1 , • • • , π α d } and Π β = {π β 1 , • • • , π β d } respectively.
We will use an induction strategy to find the correlation of s α d+1,1 and s β d+1,1 . To do so, we will require the following variables:
1. p ℓ,i;j : For one pair of i, j ∈ {2, 3}, this represents the probability at a level 2 ≤ ℓ ≤ 1 + d w.r.
t. the randomness of {π α 1 , • • • , π α ℓ-1 } and {π β 1 , • • • , π β ℓ-1 }, that s α ℓ,i and s β ℓ,j are perfectly correlated. Additionally, we define p ℓ,1;1 that represents the probability at a level 1 ≤ ℓ ≤ 1 + d w.r.t. the randomness of {π α 1 , • • • , π α ℓ-1 } and {π β 1 , • • • , π β ℓ-1 }, that s α ℓ,1 and s β ℓ,1 are perfectly correlated. 2. p ℓ,both : This represents the probability at a level 2 ≤ ℓ ≤ 1 + d w.r.t. the randomness of {π α 1 , • • • , π α ℓ-1 } and {π β 1 , • • • , π β ℓ-1 }, that either s α ℓ,2 , s β ℓ,2 are correlated and s α ℓ,3 , s β ℓ,3 are correlated, or s α ℓ,2 , s β ℓ,3 are correlated and s α ℓ,3 , s β ℓ,2 are perfectly correlated.
There are three relations that we need to keep in mind, before we proceed with the induction proof.
1. First, the correlations are circular in nature, i.e., for j ∈ {2, 3} and any integer k, p ℓ,j;j = p ℓ,j;(j+2k)%2d (Lemma F.17). We will use this relation to connect p ℓ,1;1 with p ℓ,3;3 , i.e. p ℓ,1;1 = p ℓ,3;3 .
2. Second, p ℓ,both ≤ max ij p ℓ,i;j . Intuitively, this is because getting correlations at both positions must be at most as probable as getting correlations at one of the positions.
p ℓ,both ≤ max i,j p ℓ,i;j .(2)
3. Third, cross-correlation probabilities given by p ℓ,2;3 and p ℓ,3;2 must be ≤ 1 2 (p ℓ,2;2 + p ℓ,3;3 ) (Lemma F.19). p ℓ,i;j ≤ 1 2 (p ℓ,i;i + p ℓ,j;j ), for any i, j ∈ {2, 3}, i ̸ = j.
Base condition: We will use the values of the variables at ℓ = 2 as our base condition. The input for first layer of translation is same to both MLT Π α , MLT Π β . Then, for an input s 1 ,
(s α 2,1 , s α 2,2 ) = π α 1 ((s 1,2 , s 1,3 )) (s β 2,1 , s β 2,2 ) = π β 1 ((s 1,2 , s 1,3 )).
We can then use Lemma F.22 to show that p 2,i;j = 1 3 , for all i, j ∈ {1, 2}, p 2,both = 1 3 .
Connecting the variables at ℓ and ℓ + 1: We connect p ℓ+1,2;2 to p ℓ,2;3 , p ℓ,3;2 , p ℓ,3;3 and p ℓ,both . This will undergo a case by case analysis.
1. First, if under phrasebooks π α 1 , • • • , π α ℓ-1 and π β 1 , • • • , π β ℓ-1 , if (either) s α ℓ,2 and s β ℓ,2 are perfectly correlated , and s α ℓ,3 and s β ℓ,3 are perfectly correlated (or) s α ℓ,2 and s β ℓ,3 are perfectly correlated , and s α ℓ,3 and s β ℓ,2 are perfectly correlated , (4) we can use Lemma F.14 to show that with probability 1 3 w.r.t. the selection of π α ℓ and π β ℓ , s α ℓ+1,2 and s β ℓ+1,2 will be correlated. Conditions necessary for Lemma F.14, i.e. uniform distribution of the characters in the input sequence, are shown to hold true in Lemma F.16. The probability w.r.t. the selection of phrasebooks π α 1 , • • • , π α ℓ-1 and π β 1 , • • • , π β ℓ-1
such that Equation ( 4) holds true is given by the variable p ℓ,both .
2. On the other hand, if there exists a pair i, j ∈ {2, 3}, such that under phrasebooks π α 1 , • • • , π α ℓ-1 and π β 1 , • • • , π β ℓ-1 , s α ℓ,i and s β ℓ,j are perfectly correlated, then with probability 1 9 w.r.t. the selection of π α ℓ and π β ℓ , s α ℓ+1,2 and s β ℓ+1,2 will be correlated. We again refer to Lemma F.14 for this statement. Probability that this condition happens under phrasebooks π α 1 , • • • , π α ℓ-1 and π β 1 , • • • , π β ℓ-1 is given by the variable p ℓ,i;j .
3. If none of the above situation occurs, then for all pairs i, j ∈ {2, 3}, s α ℓ,i and s β ℓ,j are uncorrelated, and so, by Lemma F.22, correlation between s α ℓ+1,2 and s β ℓ+1,2 will stay 0 for any choice of π α ℓ and π β ℓ .
Thus, combining the 3 cases, we must have
p ℓ+1,2;2 ≤ 1 3 p ℓ,both + 1 9 i,j∈{2,3} p ℓ,i;j(5)
≤ 1 3 p ℓ,both + 4 9 max i,j∈{2,3} p ℓ,i;j ≤ 1 3 max i,j∈{2,3} p ℓ,i;j + 4 9 max i,j∈{2,3} p ℓ,i;j = 7 9 max i,j∈{2,3} p ℓ,i;j
Reasoning for each step is as follows:
1. Equation ( 5) follows from conditional probability computations using the 3 cases that we discussed before.
2. Equation ( 6) uses Equation ( 2) to connect p ℓ,both to p ℓ,i;j .
We can give the same inequality for p ℓ+1,1;1 . As p ℓ+1,1;1 = p ℓ+1,3;3 from Equation ( 1) and p ℓ,2;3 and p ℓ,3;2 must be atmost the average of p ℓ,2;2 and p ℓ,3;3 (Equation ( 3)), we can write
max i,j∈{2,3} p ℓ+1,i;j ≤ 7 9 max i,j∈{2,3} p ℓ,i;j .
This implies that the probability of correlation at any output character under the two random set of phrasebooks decays at the rate of 7 9 . Solving the recurrence will give:
max i,j∈{2,3} p d+1,i;j ≤ 7 9 d-1 p 2,i;j = 7 9 d-1 1 3 .
As p d+1,3;3 should be equal to p d+1,1;1 , this implies that with probability at least 1 -p d+1,3;3 = 1 -7 9 d-1 1 3 , the following must hold true:
Correlation(MLT Π α (•) 1 , MLT Π β (•) 1 , U({0, 1} 2d )) = 0.
Because f (•) and g(•) are inputs to the phrasebooks π α and π β respectively and f (•) and g(•) are related by selective not operations in their outputs in each of the possible 4 subcases, the output of π β (g(•)) can be replaced by πβ (f (•)) for a mirror map πβ of π β . We showcase this formally for one subcase, say subcase 2; argument for other subcases are similar.
Suppose Subcase 2 is true: Then, for any two phrasebooks π α and π β , suppose πβ denotes a mirror map of π β such that
πβ (x) 1 = not(π α (x) 1 ), πβ (x) 2 = π α (x) 2 , for all x ∈ {0, 1} 2 .
Such a map exists by Lemma F.21. Then, for a pair i, j ∈ {1, 2},
Correlation(π α (f (•)) i , π β (g(•)) j , U({0, 1} k )) = Correlation(π α ((f (•) 1 , f (•) 2 )) i , π β ((g(•) 1 , g(•) 2 )) j , U({0, 1} k )) (7) = Correlation(π α ((f (•) 1 , f (•) 2 )) i , π β ((not(f (•) 1 ), f (•) 2 )) j , U({0, 1} k )) (8) = Correlation(π α ((f (•) 1 , f (•) 2 )) i , πβ ((f (•) 1 , f (•) 2 )) j , U({0, 1} k )) (9
) = Correlation(π α (•) i , πβ (•) j , U({0, 1} 2 )).(10)
Reasoning for each step is as follows.
• Because the outputs of f and g are a 2-tuple on any input, Equation ( 7) simply replaces f (x) (similarly g(x)) as (f (x) 1 , f (x) 2 ) for any input x.
• Equation ( 8) uses the relation between the outputs of f and g when subcase 2 is true.
• Equation ( 9) simply replaces π β with πβ due to their relation as mirror phrasebooks.
• Finally, because of the assumption on the output of
f , i.e. E x∼U ({0,1} k ) f (x) j = 1/2, E x∼U ({0,1} k ) f (x) 1 ⊕ f (x) 2 = 1/2, the distribution of outputs of f is identical to U({0, 1} 2 ). Hence, Correlation(π α (f (•)) i , π β (g(•)) j , U({0, 1} k )) boils down to Correlation(π α (•) i , πβ (•) j , U({0, 1}2
)). We can then use Lemma F.22 to show the probabilities of each of the desired conditions.
2. Only a pair of characters in output have correlations between f and g: For typographical simplicity, consider the case when
Correlation(f (•) 1 , g(•) 1 , U({0, 1} k )) = 1,
and for all pairs i ′ , j ′ with atleast one of them being not equal to 1, Correlation(f (•) i ′ , g(•) j ′ , U({0, 1} k )) = 0. The argument when the condition holds for other i, j pairs can be similarly handled. Then, there are 2 cases possible.
Subcase 1: f (x) 1 = g(x) 1 , for all x ∈ {0, 1} k Subcase 2: f (x) 1 = not(g(x) 1 ), for all x ∈ {0, 1} k , while in each of these pairs, (f (•) 2 , g(•) 2 ), (f (•) 2 , g(•) 1 )
, and (f (•) 1 , g(•) 2 ), the output bits of f and g are completely independent of each other. We only consider subcase 1; subcase 2 can be handled using mirror phrasebooks similar to the proof for case 1 (in particular with Equation ( 9)).
For simplicity, we will argue for i ′ , j ′ = 1, 1; the argument about other i ′ , j ′ pairs are similar. The proof will follow by two steps,
this section cite: []

Section: • Step (a):
We first argue about the probability with which Correlation( b): We then argue that if the condition in (a) holds true, then Correlation(π
π α (f (•)) 1 , π β (g(•)) 1 , U({0, 1} k )) is equal to 1, • Step (
α (f (•)) i ′ , π β (g(•)) j ′ , U({0, 1} k ))
must be 0 for any other pair i ′ , j ′ where atleast one of them is not equal to 1.
Step (a): For simplicity, we will assume that π α , π β ∈ {∆ 1 , ∆ 2 , • • • , ∆ 6 } defined in Table 7; other cases can be handled similarly as Lemma F.23.
Correlation(π α (f (•)) 1 , π β (g(•)) 1 , U({0, 1} k )) = Correlation(π α ((f (•) 1 , f (•) 2 )) 1 , π β ((g(•) 1 , g(•) 2 )) 1 , U({0, 1} k )) (11
) = Correlation(π α ((f (•) 1 , f (•) 2 )) 1 , π β ((f (•) 1 , g(•) 2 )) 1 , U({0, 1} k )) (12
) = Pr x∼U ({0,1} k ) π α ((f (x) 1 , f (x) 2 )) 1 = π β ((f (x) 1 , g(x) 2 )) 1 - Pr x∼U ({0,1} k π α ((f (x) 1 , f (x) 2 )) 1 ̸ = π β ((f (x) 1 , g(x) 2 )) 1 (13
) = Pr x∼U ({0,1}) Pr y,y ′ ∼U ({0,1} 2 ) π α ((x, y)) 1 = π β ((x, y ′ )) 1 - Pr y,y ′ ∼U ({0,1} 2 ) π α ((x, y)) 1 = π β ((x, y ′ )) 1 (14
)
The reasoning for each step is as follows:
• Because the outputs of f and g are a 2-tuple on any input, Equation ( 11) simply replaces f (x) (similarly g(x)) as (f (x) 1 , f (x) 2 ) for any input x.
• Equation ( 12) uses the relation between the outputs of f and g when subcase 1 is true.
• Equation ( 9) simply writes the definition of correlation.
• Finally, because of the assumption on the output of
f , i.e. E x∼U ({0,1} k ) f (x) j = 1/2, E x∼U ({0,1} k ) f (x) 1 ⊕
f (x) 2 = 1/2, the distribution of outputs of f can be shown to be identical to U({0, 1} 2 ).
From the definitions of π α , π β ∈ {∆ 1 , ∆ 2 , • • • , ∆ 6 } in Table 7, the first character in the outputs of π α and π β can be expressed using 3 operators on the input characters, which are copy 1 , copy 2 , and xor. These operators are independently selected for π α and π β , as they are randomly picked from these 6 possibilities. Formally, for any tuple of variables (x, y) and (x, y ′ ),
copy 1 (x, y) = x, copy 2 (x, y) = y, xor(x, y) = x ⊕ y, copy 1 (x, y ′ ) = x, copy 2 (x, y ′ ) = y ′ , xor(x, y ′ ) = x ⊕ y ′ .
However, because x, y, y ′ are independent variables, only operations copy 1 (x, y) and copy 1 (x, y ′ ) for (x, y, y ′ ) ∼ U({0, 1} 3 ) will be correlated. This can be verified by writing the definition of correlation and using copy 1 for the first character output of π α and π β :
Pr x∼U ({0,1}) Pr y,y ′ ∼U ({0,1} 2 ) π α ((x, y)) 1 = π β ((x, y ′ )) 1 - Pr y,y ′ ∼U ({0,1} 2 ) π α ((x, y)) 1 = π β ((x, y ′ )) 1 = Pr x∼U ({0,1}) Pr y,y ′ ∼U ({0,1} 2 ) [copy 1 (x, y) = copy 1 (x, y ′ )] - Pr y,y ′ ∼U ({0,1} 2 )
[copy 1 (x, y) ̸ = copy 1 (x, y ′ )] = 1, as the first term is 1 and the second term is 0. However, if you pick any other pair of operations, the correlation will be 0. We demonstrate by using copy 1 and xor for π α and π β respectively.
Pr x∼U ({0,1}) Pr y,y ′ ∼U ({0,1} 2 ) π α ((x, y)) 1 = π β ((x, y ′ )) 1 - Pr y,y ′ ∼U ({0,1} 2 ) π α ((x, y)) 1 = π β ((x, y ′ )) 1 = Pr x∼U ({0,1}) Pr y,y ′ ∼U ({0,1} 2 ) [copy 1 (x, y) = xor(x, y ′ )] - Pr y,y ′ ∼U ({0,1} 2 ) [copy 1 (x, y) ̸ = xor(x, y ′ )] = Pr x∼U ({0,1}) Pr y,y ′ ∼U ({0,1} 2 ) [x = x ⊕ y ′ ] - Pr y,y ′ ∼U ({0,1} 2 ) [x ̸ = x ⊕ y ′ ] = 0,
as both terms are equal to 1 2 in the final step. By a counting argument, one can reason that the probability of corr(π α (f (•)) 1 , π β (g(•)) 1 , U({0, 1} k )) being 1 is equal to the probability of copy 1 being selected to define the first characters of both π α and π β , which will be equal to
α (f (•)) 1 , π β (g(•)) 1 , U({0, 1} k )) = 1.
From the proof of step (a), this is only possible when copy 1 was selected to define the first characters in the outputs of π α and π β . Any other operation pairs for defining the first characters in the outputs of π α and π β would have meant correlation to be 0.
We can use this same argument to show that for any other pair i ′ , j ′ where atleast one of them is not equal to 1. Correlation(π α (f (•)) i ′ , π β (g(•)) j ′ , U({0, 1} k )) must be 0. This is because once copy 1 has been used to define the operation for the first character, it can't be used to define the operation for the second character (please refer at the truth tables for {∆ 1 , • • • , ∆ 6 } in Table 7). Following a similar argument, we can then show that correlation between output characters π α (f (•)) i ′ , π β (g(•)) j ′ will be 0, as copy 1 can't be used to define at least one of these characters.
3. The third case is when none of the above conditions hold true. That is, for any pair i, j ∈ {1, 2},
Correlation(f (•) i , g(•) j , U({0, 1} k )) = 0.
In such case, following similar arguments as case 1 and 2, one can show that for any i, j ∈ {1, 2}, for any choice of π α and π β :
Correlation(π α (f (•)) i , π β (g(•)) j , U({0, 1} k )) = Pr x,y∼U ({0,1} 2 ) Pr x ′ ,y ′ ∼U ({0,1} 2 ) π α ((x, y)) i = π β ((x ′ , y ′ )) j - Pr x,y∼U ({0,1} 2 ) Pr x ′ ,y ′ ∼U ({0,1} 2 ) π α ((x, y)) i ̸ = π β ((x ′ , y ′ )) j = 0,
due to independence of inputs to π α and π β .
Lemma F.15. At any step i of MLT(d, 2) with phrasebooks {π ℓ } d ℓ=1 , the following relation holds true for the intermediate outputs {s i } d+1
i=2 on an input s 1 ∈ {0, 1} L :
(s i+1,1 , s i+1,2 ) = π i ((s i,2 , s i,3 )).
Proof. MLT(d, 2) has 2 primary steps: Circular shift and Translate. By Circular shift, first, we first get sequence si , where for any j ∈ [L] we have si,j = s i,(j+1)%L . After Translate step,
(s i+1,1 , s i+1,2 ) = π i ((s i,1 , si,2 )) = π i ((s i,2 , s i,3 )).
Lemma F.16. At any step i of MLT(d, 2) with phrasebooks {π ℓ } d ℓ=1 , the following conditions hold true for the intermediate output s i .
E s1∼U ({0,1} L ) s i,j = 1 2 , for all 1 ≤ j ≤ L E s1∼U ({0,1} L ) s i,j ⊕ s i,j ′ = 1 2 , for all j ̸ = j ′ .
The above conditions are equivalent to showing that s i,j behaves like a uniformly random boolean variable, independent of any other character s i,j ′ for all coordinates j ′ ̸ = j .
Proof. The proof will follow by induction on the output of the translation task at each step. We will show the result for coordinate j = 1 in s i ; similar argument holds for other coordinates j.
Base condition: At layer i = 1, the s 1 represents the input sequence from U({0, 1} L ). By definition of uniform distribution, the conditions hold true for the input.
this section cite: []

Section: Operator Expected value
Expected ⊕ value with operator 2) and Esi,1 ⊕ si,2 under different πi phrasebooks, defined by copy and xor operations on si-1,2 and si-1,3.
copy 1 copy 2 xor copy 1 E s1 s i-1,2 = 1/2 - E s1 s i-1,2 ⊕ s i-1,3 = 1/2 E s1 s i-1,3 = 1/2 copy 2 E s1 s i-1,3 = 1/2 E s1 s i-1,2 ⊕ s i-1,3 = 1/2 - E s1 s i-1,2 = 1/2 xor E s1 s i-1,2 ⊕ s i-1,3 = 1/2 E s1 s i-1,3 = 1/2 E s1 s i-1,2 = 1/2 - Table 6. Esi,1 (similarly Esi,
Induction step: Argument for general i > 1: Suppose the conditions are true for all layers 1 ≤ ℓ < i. Then for layer i, we will provide an argument for the condition to hold true for j = 1 and j = 2, arguments for other js will extend similarly. By Lemma F.15,
(s i,1 , s i,2 ) = π i ((s i-1,2 , s i-1,3 )).
From Lemma F.21, we have each output character can be defined in terms of copy, xor, and not operations on input characters. Then, the relations for s i,1 , s i,2 can be computed as follows:
• If a map π α is selected from Mirrorset(π) for some π ∈ {∆ 1 , • • • , ∆ 6 }, then one can show that the conditions hold true for π i = π α if the conditions hold true for π i = π. This follows because the output of π α follows from the output of π by selective not operations to the output of π. As not operation won't change the expected values of a variable which behaves like a random boolean variable, the argument follows.
• Now, we show that for π i = π for some π ∈ {∆ 1 , • • • , ∆ 6 }, the conditions hold true. For these phrasebooks, the output characters are defined by copy and xor operations. We use the definitions of these phrasebooks from Table 7 and show the expected values s i,1 in terms of expected values of s i-1,1 and s i-1,2 , and the expected values s i,1 ⊕ s i,2 in terms of expected values of s i-1,1 and s i-1,2 in Table 6.
Both of the above arguments then can be combined to show that
E s1∼U ({0,1} L ) s i,j = 1 2 , for j ∈ {1, 2} E s1∼U ({0,1} L ) s i,1 ⊕ s i,2 = 1 2 .
We can similarly extend the argument for expectation of each individual character to other positions j > 2, i.e. E s1∼U ({0,1} L ) s i,j = 1 2 for all other possible js. We can also similarly extend the argument for joint expectation of two consecutive characters s i,2k+1 ⊕ s i,2k+2 for any general k.
The remaining argument will be to show that characters that don't form a consecutive 2-tuple are going to be independent of each other as well. Consider any two 2-tuples (s i,2k+1 , s i,2k+2 ) and (s i,2k ′ +1 , s i,2k ′ +2 ), with k ̸ = k ′ . By adapting Lemma F.15, one can show that
(s i,2k+1 , s i,2k+2 ) = π i ((s i-1,2k+2 , s i,2k+3 )) (s i,2k ′ +1 , s i,2k ′ +2 ) = π i ((s i-1,2k ′ +2 , s i,2k ′ +3 ))
The primary thing to note here is that both the tuples depend on two distinct tuples in layer i -1. By induction assumption, characters across these two 2-tuples are independent of each other. As π i is a bijective map, this will also suggest that characters across the 2-tuples in the resulting output must also be independent of each other. This will give the final argument.
Lemma F.17. Under the assumption that sequence lengths L are even, for any 2 ≤ i ≤ d + 1, where s α i,j , s β i,j denote the output after i -1st translation step for a random sequence s 1 ∼ U({0, 1} L ) at any position 1 ≤ j ≤ L under the two set of phrasebooks Π α :i := {π α ℓ } i-1 ℓ=1 and Π β :i := {π β ℓ } i-1 ℓ=1 , the following holds true for all positions j:
Correlation(s α i,j , s β i,j , U({0, 1} L )) = Correlation(s α i,(j+2k)%L , s β i,(j+2k)%L , U({0, 1} L )), for all integer k.
Proof. Fix an integer k. By Definition F.1, the necessary condition would be to show
1 -2E s1∼{0,1} L s α i,j ⊕ s β i,j = 1 -2E s1∼{0,1} L s α i,(j+2k)%L ⊕ s β i,(j+2k)%L .
We will look at the behavior of the function h(s 1 ) = s α i,j ⊕ s β i,j . Denote C k as a circular operation that takes a sequence s 1 and returns a shifted sequence ⟲ s 1 , i.e. if ⟲ s 1 = C k (s 1 ) then for all j, ⟲ s 1,j = s 1,(j+2k)%L . Note that, we can also define an inverse function C -1 k and s 1 = C -1 k (C k (s 1 )). From the definition of MLT Π for any phrasebook Π, we have for any input s 1 :
s i = T πi-1 • • • • • T π1 (s 1 ) ,
where for any level i, T πi denotes the translation step at level i that includes Circular shift and Translate using π i .
In Lemma F.18, we show that for any translation task MLT Π and any input s 1 ,
T πi-1 • • • • • T π1 (s 1 ) = C -1 k T πi-1 • • • • • T π1 • C k (s 1 ) . On input ⟲ s 1 = C k (s 1 ), if ⟲ s α
i and ⟲ s β i represent the output of translations using Π α :i := {π α ℓ } i-1 ℓ=1 and Π β :i := {π β ℓ } i-1 ℓ=1 respectively, then the above statement says that
s α i = C -1 k ⟲ s α i , s β i = C -1 k ⟲ s β i (or) C k (s α i ) = ⟲ s α i , C k s β i = ⟲ s β i
Thus, for any position j, we will have C k (s α i ) j = ⟲ s α i,j (and similarly for ⟲ s β i,j ). We can then compute the value of h on input ⟲ s 1 as follows:
h( ⟲ s 1 ) = ⟲ s α i,j ⊕ ⟲ s β i,j = C k (s α i ) j ⊕ C k s β i j = s α i,(j+2k)%L ⊕ s β i,(j+2k)%L
The last step follows from using the definition of C k . On the other hand,
E⟲ s1∼U ({0,1} L ) h( ⟲ s 1 ) = E s1∼U ({0,1} L ) h(s 1 ),
as the uniform distribution can be shown to not change under circular function C k . This will imply:
E s1∼U ({0,1} L ) s α i,(j+2k)%L ⊕ s β i,(j+2k)%L = E s1∼{0,1} L s α i,j ⊕ s β i,j =⇒ 1 -2E s1∼{0,1} L s α i,j ⊕ s β i,j = 1 -2E s1∼{0,1} L s α i,(j+2k)%L ⊕ s β i,(j+2k)%L =⇒ Correlation(s α i,j , s β i,j , U({0, 1} L )) = Correlation(s α i,(j+2k)%L , s β i,(j+2k)%L , U({0, 1} L )).
Lemma F.18. For any translation task MLT Π , at any level i ≤ d and any input s 1 ,
T πi • • • • • T π1 (s 1 ) = C -1 k (T πi • • • • • T π1 • C k (s 1 )) .
Proof. Denote C k as a circular operation that takes a sequence s 1 and returns a shifted sequence ⟲ s 1 , i.e. if ⟲ s 1 = C k (s 1 ) then for all j, ⟲ s 1,j = s 1,(j+2k)%L . Note that, we can also define an inverse function C -1 k and s 1 = C -1 k (C k (s 1 )). Recall that for any level i, T πi denotes the translation step at level i that includes Circular shift, and Translate with π i . The argument will again follow by an induction step.
Base condition: i = 1: We need to show that
T π1 (s 1 ) = C -1 k (T π1 • C k (s 1 )) .
To do so, we will look at the behavior of the first 2 characters. Argument for others can be extended. If ⟲ s 1 = C k (s 1 ), s 2 = T π1 (s 1 ), ⟲ s 2 = T π1 ( ⟲ s 1 ) , then by Lemma F.15,
(s 2,1 , s 2,2 ) = π 1 ((s 1,2 , s 1,3 )) ( ⟲ s 2,1 , ⟲ s 2,2 ) = π 1 (( ⟲ s 1,2 , ⟲ s 1,3 )).
But by definition of C k ,
( ⟲ s 1,2 , ⟲ s 1,3 ) = (s 1,(2+2k)%L , s 1,(3+2k)%L ).
On the other hand, Lemma F.15 can be adapted to give
s 2,(1+2k)%L , s 2,(2+2k)%L = π 1 (s 1,(2+2k)%L , s 1,(3+2k)%L ).
Thus, we can show by combining the above 3 steps that
( ⟲ s 2,1 , ⟲ s 2,2 ) = π 1 (( ⟲ s 1,2 , ⟲ s 1,3 )) = π 1 (s 1,(2+2k)%L , s 1,(3+2k)%L ) = s 2,(1+2k)%L , s 2,(2+2k)%L .
We can extend the above argument to show that for any position j, ⟲ s 2,j = s 2,(j+2k)%L , which by definition of C k , implies
⟲ s 2 = C k (s 2 ) or s 2 = C -1 k ⟲ s 2 ,
which can be further simplified (using the notations: ⟲ s 1 = C k (s 1 ), s 2 = T π1 (s 1 ), ⟲ s 2 = T π1 ( ⟲ s 1 ))
s 2 = C -1 k ⟲ s 2 = C -1 k T π1 ⟲ s 1 = C -1 k (T π1 (C k (s 1 ))) := C -1 k (T π1 • C k (s 1 )) .
General argument for i: Suppose the induction condition holds true for all layers ℓ < i. Then,
T πi • • • • • T π1 (s 1 ) = T πi C -1 k T πi-1 • • • • T π1 • C k (s 1 ) .
We can then follow the same argument as the base condition, and show that
T πi C -1 k T πi-1 • • • • T π1 • C k (s 1 ) = C -1 k (T πi • • • • T π1 • C k (s 1 )) .
Lemma F.19. For any 1 ≤ i ≤ d, if s α i,j , s β i,j denote the output after i -1st translation step for a random sequence s 1 ∼ U({0, 1} L ) at any position 1 ≤ j ≤ L under the two sets of random phrasebooks Π α :i := {π α ℓ } i-1 ℓ=1 and Π β :i := {π β ℓ } i-1 ℓ=1 , the following holds true for all positions j:
Pr Π α :i ,Π β :i Correlation s α i,j , s β i,j+1 , U({0, 1} L ) = 1 ≤ 1 2 Pr Π α :i ,Π β :i Correlation s α i,j , s β i,j , U({0, 1} L ) = 1 + Pr Π α :i ,Π β :i Correlation s α i,j+1 , s β i,j+1 , U({0, 1} L ) = 1 .
Proof. We will prove the required result with a counting argument. We will create a family of phrasebooks: let FAMILY(j) and FAMILY(j + 1) denote two sets of phrasebooks, such that for every phrasebook Π α :i in FAMILY(j), there exists at least one phrasebook Π β :i in FAMILY(j + 1) such that Correlation s α i,j , s β i,j+1 , U({0, 1} L ) = 1, (Equiv. to saying translations for Π α :i and Π β :i have correlations at position j and j + 1)
where s α i,j , s β i,j+1 are outputs on a random sequence s 1 ∼ U({0, 1} L ) corresponding to using Π α :i and Π β :i respectively. We then apply a grouping algorithm GROUP to group correlated phrasebooks together in each family. That is, in FAMILY(j), we create groups of phrasebooks {S 1 , S 2 , • • • } such that for any two phrasebooks Π α :i and Π α ′ :i that belong to a group S,
Correlation s α i,j , s α ′ i,j , U({0, 1} L ) = 1,
(Equiv. to saying translations for Π α :i and Π β :i have correlations at position j)
where s α i,j , s α ′ i,j+1 are outputs on a random sequence s 1 ∼ U({0, 1} L ) corresponding to using Π α ′ :i and Π α ′ :i respectively. We call the resulting output of this operation as GROUP(FAMILY(j)). Similarly, we compute GROUP(FAMILY(j + 1)).
We can observe the following two characteristics of GROUP(FAMILY(j)) and GROUP(FAMILY(j + 1)):
1. For every set S 1 ∈ GROUP(FAMILY(j)) there will exist one set S 2 ∈ GROUP(FAMILY(j + 1)), such that for all phrasebooks Π α :i ∈ S 1 and Π β :i ∈ S 2 , correlation will be 1 for output at positions j and j + 1. 2. For every set S 1 ∈ GROUP(FAMILY(j)) there can't exist two sets S 2 , S * 2 ∈ GROUP(FAMILY(j + 1)), such that the phrasebooks in S 1 are correlated to phrasebooks from both S 2 , S * 2 for output at positions j and j + 1 respectively. Otherwise, we could have merged S 2 and S * 2 under the GROUP operation.
Let CORRELATION-MAP denote the map between GROUP(FAMILY(j)) and GROUP(FAMILY(j + 1)), which connects sets S 1 ∈ GROUP(FAMILY(j)) to a set S 2 ∈ GROUP(FAMILY(j + 1)) such that any two phrasebooks in S 1 and S 2 have correlations for output at positions j and j + 1 respectively.
The result will then follow from a counting argument. The number of possible pairs (can be identical phrasebooks) that can give correlations for output at position j are given by: S1∈GROUP(FAMILY(j)) |S 1 | 2 . Similarly, the number of possible pairs that can give correlations for output at position j + 1 are given by: S2∈GROUP(FAMILY(j+1)) |S 2 | 2 . On the other hand, the number of possible pairs that can give correlations for output at position j and j + 1 respectively are given by: S1∈GROUP(j);S2=CORRELATION-MAP(S1) |S 1 ||S 2 |. Applying the AM-GM inequality, we can show that the average of the number of pairs for which correlation is 1 for output characters at either position j or j + 1 is higher than the number of pairs for which correlation is 1 for output at positions j and j + 1.
this section cite: []

Section: F.5. Proof for Statistical query lower bound for general n
We present the main theorem statement again for readability.
Theorem F.20 (SQ dimension for general n). For the translation task MLT(d, n) that has depth d and n characters per level, the statistical query dimension SQ-dim(MLT(d, n)) is atleast n Ω(d) .
Proof. We adapt the SQ-dimension proof for MLT(d, 2) to show the SQ-dimension proof for MLT(d, n). We will design a family of set of phrasebooks Π = {π
i : {0, 1, • • • , n -1} 2 → {0, 1, • • • , n -1} 2 } d
i=1 , where phrasebooks in Π are built on top of a translation task in MLT(log 2 n, 2).
For a character a ∈ {0, 1, • • • , n -1}, suppose BIT(a) ∈ {0, 1} log 2 n indicates its binary representation, and NUMERIC represents the map from binary representation to its corresponding numeric representation. Then, we design each phrasebook π using a random translation task ν ∈ MLT(log 2 n, 2). For any tuple (a, b) ∈ {0, 1, • • • , n -1} 2 , output of π is given as π(a, b) = (o 1 , o 2 ), where
o 1 = NUMERIC ν {ã i ⊕ bi } log 2 n i=1 o 2 = NUMERIC ν b ã = BIT(a) b = BIT(b)
Primarily, the phrasebooks are defined as follows:
1. On a 2-tuple of characters (a, b), we first compute their binary representations (BIT(a), BIT(b)). We then compute two intermediate outputs, one where a xor operation is applied on BIT(a), BIT(b) at each bit, and another where BIT(b) is simply copied. This operation is equivalent to applying a deterministic map on 2-tuples of binary characters (identical to ∆ 6 from Table 7), where 2-tuples are created by pairing binary bits in binary representation of a and b.
2. We then apply a random MLT task of depth log 2 n on each of the intermediate outputs. This applies a random bijective map on the sequence of bits in the intermediate outputs, using a translation task in MLT(log 2 n, 2) (Lemma E.1).
this section cite: []

Section: The final tuple of characters is returned by applying a NUMERIC operation on the binary representations.
Thus, we have narrowed our focus on a special group of tasks from MLT(d, n) that applies MLT(log 2 n, 2) on the binary representations of the characters at each level. By Theorem F.10, at any level 1 ≤ i ≤ d, we can create 2 Ω(log 2 n) phrasebooks, the output of which are pairwise uncorrelated . Now, we can compose these uncorrelated phrasebooks to give multiple set of phrasebooks that are uncorrelated. That is, following a similar proof as Theorem F.10, we can show that we can create 2 Ω(log 2 n) Ω(d) = n Ω(d) set of phrasebooks, using the above restriction, that are pairwise uncorrelated on any bit in the binary representation of the output of their corresponding translation tasks. This will translate to the output of the translation task in numeric form as well, as the mapping between binary representation and numeric form of a digit is bijective.
this section cite: []

Section: F.6. Proofs of Useful lemmas
Here we give the proofs for the useful lemmas necessary to prove Theorem F.10. We repeat the lemma statements for easier readability.
Lemma F.21 (Formulation of bijective maps for n = 2). Any bijective map π : {0, 1} 2 → {0, 1} 2 can be expressed using copy, not, and xor operations. Furthermore, from 24 possible maps for π, 1. There are 6 maps ∆ 1 , ∆ 2 , • • • , ∆ 6 for which characters in the output tuple can be defined by copy and xor operations on the characters in the input tuple.
2. Mirror maps: For each map π ∈ {∆ 1 , ∆ 2 , • • • , ∆ 6 }, there exist mirror maps π (1) , π (2) , π (3) whose output on each input tuple can be defined by selective not operations on either or both characters of the output tuple of π. We call {π, π (1) , π (2) , π (3) } as a mirror map set of π, in short, Mirrorset(π).
Proof. There are 4 possible tuples (0, 0), (0, 1), (1, 0), (1,1). By Lemma E.2, the number of possible bijective maps (phrasebooks) that connect 2-tuples are 4! = 24. We will show that the output tuple of each map can be represented by 3 operations.
Operation not creates mirror maps: For each map π, there exists 3 alternative maps π (1) , π (2) , π Table 7. The table captures the definition of 6 bijective maps (phrasebooks) on 2-tuples {0, 1} 2 → {0, 1} 2 , whose output characters can be defined in terms of copy and xor operations on the input characters. Any other bijective map can be shown to belong to Mirrorset of one of these maps.
• π (1) selectively applies the not operation to the first character in the output tuple of π on any input tuple, i.e.
π (1) (a, b) = (not(π(a, b) 1 ), π(a, b))
for all tuples (a, b) ∈ {0, 1} 2 .
• π (2) selectively applies the not operation to the second character in the output tuple of π on any input tuple, i.e.
π (2) (a, b) = (π(a, b) 1 , not(π(a, b)))
for all tuples (a, b) ∈ {0, 1} 2 .
• π (3) selectively applies the not operation to both characters in the output tuple of π on any input tuple, i.e.
π (3) (a, b) = (not(π(a, b) 1 ), not(π(a, b)))
for all tuples (a, b) ∈ {0, 1} 2 .
Thus, for each map π, there exist 3 other alternative maps that simply modify the output of map π with the not operation.
After removing the mirror maps: We now show that there 6 possible maps that apply either a xor or a copy on the input characters to get the output characters. We name them ∆ 1 , ∆ 2 , • • • , ∆ 6 . We give the output of each map on the 4 tuples in Table 7 and show that the each character in the output tuple can be represented using copy and xor operations.
Lemma F.22 (Correlation of bijective maps for n = 2). For two randomly selected maps π α , π β : {0, 1} 2 → {0, 1} 2 , the following hold true.
or
Correlation(π α (•) 1 , π β (•) 2 , U({0, 1} 2 )) = 1. Correlation(π α (•) 2 , π β (•) 1 , U({0, 1} 2 )) = 1.
Other cases are not possible, i.e. for any i ∈ {1, 2}, both Correlation(π
α (•) i , π β (•) 1 , U({0, 1} 2 )) = 1 and Correlation(π α (•) i , π β (•) 2 , U({0, 1} 2 )) = 1 can't hold true.
Proof. We prove the lemma for case 1, cases 2 and 3 can be similarly proved. From Lemma E.2, π α and π β can be randomly selected from a set of 24 possible candidates. On the other hand, Lemma F.21 shows that there are 6 maps {∆ 1 , • • • , ∆ 6 } whose output can be defined in terms of copy and xor operations of characters in the input tuple. For each π in this set, there are mirror maps π (1) , π (2) , π (3) whose output are defined by selective not operations on the output of π, and the set of 4 maps is represented by Mirrorset(π).
The proof will follow from 2 steps: first, we argue about correlations when π α and π β are selected from {∆ 1 , • • • , ∆ 6 }, and then we argue about the general case when π α and π β are selected from the general set of bijective maps.
• In Lemma F.24, we show that for two maps that are randomly selected from {∆ 1 , • • • , ∆ 6 }, the correlation is 1 with probability 1/3.
• The remaining possibility is when π α and π β belong to Mirrorset(π i ) and Mirrorset(π j ) for some π i , π j ∈ {∆ 1 , • • • , ∆ 6 }. We show in Lemma F.23, correlation of π α and π β will be equal to correlation of π i , π j .
Thus, we can combine all the observations to show that for two random maps π α , π β that belong to Mirrorset(π i ) and Mirrorset(π j ) for some
π i , π j ∈ {∆ 1 , • • • , ∆ 6 }, Correlation(π α (•) 1 , π β (•) 1 , U({0, 1} 2 )) = Correlation(π i (•) 1 , π j (•) 1 , U({0, 1} 2 )) = 1, w.p. 1/3 w.r.t. randomness in π i , π j 0, otherwise.
Lemma F.23. The following holds true for any maps π α and π β with π α , π β ∈ {∆ 1 , • • • , ∆ 6 } and for all πα ∈ Mirrorset(π α ), πβ ∈ Mirrorset(π β ):
Correlation(π α (•) 1 , π β (•) 1 , U({0, 1} 2 )) = Correlation(π α (•) 1 , πβ (•) 1 , U({0, 1} 2 )),
Proof. We prove as follows:
Correlation(π α (•) 1 , π β (•) 1 , U({0, 1} 2 )) = Pr x∼U ({0,1} 2 ) [π α (x) 1 ̸ = π β (x) 1 ] - Pr x∼U ({0,1} 2 ) [π α (x) 1 = π β (x) 1 ] = 2 Pr x∼U ({0,1} 2 ) [π α (x) 1 ̸ = π β (x) 1 ] -1 (15
) = 2 Pr x∼U ({0,1} 2 ) [π α (x) 1 ̸ = πβ (x) 1 ] -1 , if condition "c1" is true 1 -2 Pr x∼U ({0,1} 2 ) [π α (x) 1 = πβ (x) 1 ] , if condition "c2" is true (16
) = Pr x∼U ({0,1} 2 ) [π α (x) 1 ̸ = πβ (x) 1 ] - Pr x∼U ({0,1} 2 ) [π α (x) 1 = πβ (x) 1 ] = Correlation(π α (•) 1 , πβ (•) 1 , U({0, 1} 2 )),(17)
where the second and the penultimate steps follow from the law of total probability. Here, condition "c1" holds when either case is true,
πα (x) 1 = not(π α (x) 1 ), πβ (x) 1 = not(π β (x) 1 ), for all x ∈ {0, 1} 2 πα (x) 1 = π α (x) 1 , πβ (x) 1 = π β (x) 1 , for all x ∈ {0, 1} 2 .
and condition "c2" holds when either case is true,
πα (x) 1 = π α (x) 1 , πβ (x) 1 = not(π β (x) 1 ), for all x ∈ {0, 1} 2 πα (x) 1 = not(π α (x) 1 ), πβ (x) 1 = π β (x) 1 , for all x ∈ {0, 1} 2 .
One of condition "c1" or condition "c2" is true because πα and π α (similarly, πβ and π β ) are mirror maps.
Lemma F.24. The following holds true for two randomly selected maps π α and π β with π α , π β ∈ {∆ 1 , • • • , ∆ 6 } :
• With probability 1/3 w.r.t. random selection, Correlation(π α (•) 1 , π β (•) 1 , U({0, 1}2
)) > 0 (= 1).
• With probability 1/3 w.r.t. random selection,
Correlation(π α (•) 2 , π β (•) 2 , U({0, 1} 2 )) > 0 (= 1).
Proof. We prove for case 1, proof for case 2 is analogous.
Among {∆ 1 , • • • , ∆ 6 }, we can create three family of maps: F copy1 : {∆ 1 , ∆ 2 }, F copy2 : {∆ 3 , ∆ 4 }, F xor : {∆ 5 , ∆ 6 } that are identical in operation (copy 1 , copy 2 , xor respectively) at the first character in output tuple. This would imply, if the π α and π β both belong to one of these families, Correlation(π
α (•) 1 , π β (•) 1 , U({0, 1}2
)) will be 1.
On the other hand, one can show that for any operation
f 1 , f 2 ∈ {copy 1 , copy 2 , xor} with f 1 ̸ = f 2 will have Correlation(f 1 , f 2 , U({0, 1} 2 )) = 0.
That would then suggest that if π α and π β belong to different families among
F copy1 , F copy2 , F xor , then Correlation(π α (•) 1 , π β (•) 1 , U({0, 1}2
)) will be 0.
By a simple counting argument, with probability 1 3 , two maps π α and π β randomly selected from {∆ 1 , • • • , ∆ 6 } will have non-zero correlation.
this section cite: []

Section: G. Upper Bound: Context-Enhanced Learning of MLT(d, n) with Simple Surrogate Model

this section cite: []

Section: G.1. Setup of Surrogate Model with In-context Capability
Given d + 1 alphabets A 1 , . . . , A d+1 of size n and d bijective phrasebooks π i :
A 2 i → A 2 i+1
The input of the translation process is an even-length sequence in the first alphabet, which we denote as s 1 ∈ A L 1 where L is the sequence length. The translation process modifies the input string recursively from s i to s i+1 through the following 2 sub-processes:
1. Circular shift: The characters in s i ∈ A L i are shifted by 1 character leftward (and wrapped around to the end if necessary) to give sequence si ∈ A L i . Formally, for each j ∈ [1, L] we have si,j = s i,(j+1)%L . 2. Translate: Using the phrasebook π i :
A 2 i → A 2 i+1 , we translate 2-tuples (bigrams) of consecutive characters in sequence si to create s i+1 . That is, for every odd j ∈ [1, L], (s i+1,j , s i+1,j+1 ) = π i (s i,j , si,j+1 ).
this section cite: []

Section: Now let us revisit the surrogate model introduced in Section 5.1. Without loss of generality let
A 1 = A 2 = • • • = A d+1 := A = {1, 2, . . . , n}.
For any single character a ∈ A, let its vector representation be a one-hot vector e a ∈ R n such that (e a ) a = 1. For any 2-tuple (a, b) ∈ A 2 , let its vector representation be a n
2 -dimensional vector v(a, b) ≜ e a ⊗ e b . Note that v(a, b) is also a one-hot vector where v(a, b) i = 1 if i = an + b 0 elsewhere .
We use the notation ēa (long one-hot) to denote a one-hot vector in R n 2 with a-th position being 1 to avoid confusion.
Definition G.1 (Matrix Representation of Sequence). For a length-L input sequence s i = (s i,1 , . . . , s i,L ), let its matrix representation be Mat(s i ) ≜ V i ∈ R n 2 ×L/2 that V i =   | | • • • | v (s i,1 , s i,2 ) v (s i,3 , s i,4 ) • • • v (s i,L-1 , s i,L ) | | • • • |   For each j ∈ [L/2], we use V (j) i
to denote the j-th column of V i . We also denote the above conversion from a sequence s i to its matrix form as V i = Mat(s i ) and assume that V 1 serves as the input to the surrogate model.
Note that the matricization operation is invertible by construction: for each column V
(j) i , let x = arg max V (j) i , we may read off the two characters in the original alphabet by computing Mat -1 (V (j) i ) = (⌈x/n⌉, x%n).
At each level of translation, we assume the surrogate model will perform the following operations to V i :
(a) Circular shift from V i to Ṽi Definition G.2 (Circular Shifting Operator Shift). Given a matrix representation V ∈ R n 2 ×L/2 of a sequence s, the circular shifting operator Shift acts on V as Shift(V ) := Ṽ ∈ R n 2 ×L/2 where for all j ∈ [L/2], Ṽ (j) = QV (j) ⊙ Q ⊤ V ((j+1)%L) where Q = (I n ⊗ 1 n ) (1 n ⊗ I n ) ⊤ , 1 n ∈ R n×1
is the all-ones vector, and ⊙ is the Hadamard product.
this section cite: []

Section: Lemma G.3 (Equivalence of Shift and circular shift).
For any sequence s ∈ A L , let s be the circular shifted s, then
Mat (s) = Shift (Mat (s)) .
Proof of Lemma G.3. In this proof we will show that Mat (s) and Shift (Mat (s)) agrees on every column.
Fix a column j ∈ [n/2], without loss of generality let the input sequence s be (a, b, c, d) ∈ A starting from the (2j -1)-th position to the (2j + 2)-th position (wrapped around when necessary). By construction each output column of Ṽ (j) is dependent on at most V (j) and V (j+1) which corresponds to 4 characters in the sequence s.
By definition of V we then have V
(j) = v(a, b) = e a ⊗ e b and V (j+1)%L = v(c, d) = e c ⊗ e d . It follows that Ṽ (j) = QV (j) ⊙ Q ⊤ V ((j+1)%L) = (I n ⊗ 1 n ) 1 ⊤ n ⊗ I ⊤ n (e a ⊗ e b ) ⊙ (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n (e c ⊗ e d ) = (I n ⊗ 1 n ) 1 ⊤ n e a ⊗ I ⊤ n e b ⊙ (1 n ⊗ I n ) I ⊤ n e c ⊗ 1 ⊤ n e d = ((I n ⊗ 1 n ) (1 ⊗ e b )) ⊙ ((1 n ⊗ I n ) (e c ⊗ 1)) (e a , e d are one-hot, 1 ⊤ n e a = 1 ⊤ n e d = 1) = ((I n ⊗ 1 n ) (e b ⊗ 1)) ⊙ ((1 n ⊗ I n ) (1 ⊗ e c )) = (e b ⊗ 1 n ) ⊙ (1 n ⊗ e c ) = e b ⊗ e c .
(by definition of Kronecker product)
Note that e b ⊗ e c is just Mat(s) (j) since the (2j -1)-th and the 2j-th character of the shifted sequence s is now (b, c).
This concludes the proof.
(b) Translation from Ṽi to V i+1
With Shift effectively completing Merge, Circular shift, and Split, what remains is the translation leveraging π i . Since
A i = A i+1 = A = [n]
, there is a natural bijection between the space of binary column-stochastic matrix and the space of all possible (not necessarily bijective) mappings between 2-tuples from A. Concretely
Definition G.4 (Column-Stochastic Matrix Representation of phrasebook).
Given phrasebook π : A 2 → A 2 , its matrix representation is defined to be Matrix(π) ∈ R n 2 ×n 2 that for i, j ∈ [n 2 ],
Matrix(π) j,i = 1 if π(⌈i/n⌉, i%n) = (⌈j/n⌉, j%n) 0 elsewhere . Let V i+1 = Matrix Ṽi
where Matrix is a binary column-stochastic matrix, we can show that, a complete translation process for one level can be formally expressed as follows:
Lemma G.5 (Equivalence of Matrix and Translate).
this section cite: []

Section: For any sequence s
i ∈ A L , let π i be a bijective phrasebook A 2 → A 2 defined in Section 2.2, then V i+1 = Matrix(π i ) Shift (Mat (s i )) = Mat (T πi (s i )) = Mat(s i+1 ).
Proof of Lemma G.5. Fix any column j ∈ [L/2], let (a, b) be the (2j -1)-th and the 2j-th character of the shifted sequence si . Let (c, d) = π i (a, b), by the translation construction we know the (2j -1)-th and the 2j-th character of s i+1 is just c and d. Hence Mat(s i+1 ) (j) = v(c, d).
this section cite: []

Section: From Lemma G.3 we know that Shift(Mat(s
i )) (j) = v(a, b). By construction of Matrix(π i ) above, we know that Matrix(π i ) Shift(Mat(s i )) (j) is a one-hot vector at the (cn + d)-th position, i.e. V (j) i+1 = Matrix(π i ) Shift(Mat(s i )) (j) = e c ⊗ e c = v(c, d). Since Mat(s i+1 ) (j) = V (j)
i+1 for all j, we have V i+1 = Mat(s i+1 ).
this section cite: []

Section: Parameterization of the Translation Matrix P .
With the two key operations in place, now we can introduce the surrogate model in its full detail. Since the multi-level translation task is a naturally sequential operation, we model the surrogate operations as a multi-layer network as well (which also matches with the solution found by Llama-based models when trained on real data as in Section 3.3).
For each level, the surrogate model needs to present both in-context learning capability at the initialization and in-weight capability toward the end of context-enhanced learning on a certain set of phrasebooks Π * . The in-weight capability requires certain parameter to store Π * on its own that is independent of the context.
To capture both capabilities at the same time, we parameterize the translation matrix P i as a combination of in-context information C i ∈ R n 2 ×n 2 and in-weight memory W i ∈ R n 2 ×n 2 .
Definition G.6 (Effective Translation Matrix). For in-context information C i ∈ R n 2 ×n 2 and in-weight memory W i ∈ R n 2 ×n 2 at level i, the corresponding effective translation matrix P i is defined as
P i = HardMax (C i + W i )
where HardMax is the column-wise hard-max function converting C i + W i to a binary column stochastic matrix.
Note that column k in P i , which we denote as P (k) i , is equal to the one-hot vector at arg max(C
(k) i + W (k) i ).
this section cite: []

Section: Surrogate Model with In-Context Capability
Now we can formally introduce the surrogate model.
Definition G.7 (Surrogate Model for MLT). The surrogate model for MLT(d, n) can be represented by the recursive expression
V i+1 = HardMax(C i + W i ) Shift(V i )(18)
The learnable parameters for the surrogate model are the weight matrices
{W i } d i=1 := {W 1 , W 2 , . . . , W d }. We denote the surrogate model parameterized by {W i } d i=1 as SURR-MLT {Wi} d i=1 (•)
which maps the input and the in-context information to the output as
V d+1 = SURR-MLT {Wi} d i=1 (C 1 , C 2 , . . . , C d , V 1 ) ≜ HardMax(C d + W d ) Shift (HardMax(C d-1 + W d-1 ) Shift (• • • HardMax(C 1 + W 1 ) Shift(V 1 ) • • • )) .(19)
In this surrogate model, the in-context descriptive text DESC is just {C 1 , . . . , C d }, where each matrix is directly being passed into the corresponding layer. When providing information about a set of phrasebooks Π = {π i } d i=1 , we have C i = Matrix(π i ) where Matrix(π i ) is the column-stochastic matrix representation of π i as defined in Definition G.4.
When partial phrasebook information is provided (corresponding to dropping certain ab->CD entries in the language model context), we zero-out the corresponding column in C i . When no in-context information is provided for level i, we just have C i be the all-zero matrix containing no information (assuming zero as prior).
For simplicity of presentation, in Definition G.7 the in-context information C i 's are provided directly to the corresponding layers. We note that the equivalent operations can be exactly re-parameterized such that C i 's are provided in-context (in concatenation with V 1 ). The reparameterization of the surrogate model is provided below: Definition G.8 (Context-Augmented Surrogate Model for MLT). Let the context-augmented input be
X 1 = [C 1 , . . . , C d , V 1 ] ∈ R n 2 ×(dn 2 +L) ,
then we can rewrite the same surrogate model as
X i+1 = X i I dn 2 0 0 0 L×L + X i e i ⊗ I n 2 0 L×n 2 + W i Shift X i 0 n 2 d×L I L 0 dn 2 ×dn 2 0 0 I L = [C 1 , . . . , C d , 0 n 2 ×L ] + [0, . . . , 0, (C i + W i ) Shift (V i )] = [C 1 , . . . , C d , V i+1 ](20)
With the reparameterization, the model is capable of generating [C 1 , . . . ,
C d , V d+1 ] with input [C 1 , . . . , C d , V 1 ],
with all information provided in-context in the input.
this section cite: []

Section: Now let us check what does Definition 2.2 (ICL-capable) and Definition 2.1 (specific task-capable) mean in the context of the surrogate model.
To make things more rigorous we introduce two stronger notions of capabilities:
Definition G.9 (Strongly MLT(d, n)-ICL-capable surrogate model). We say a surrogate model SURR-MLT {Wi} d i=1 (•) is strongly MLT(d, n)-ICL-capable if for any set of phrasebooks Π = {π i } d i=1 in MLT(n, d), for any input sequence s 1 ∈ A L where L is even, we have SURR-MLT {Wi} d i=1 (Matrix(π 1 ), . . . , Matrix(π d ), Mat(s 1 )) = Mat(MLT Π (s 1 )). Definition G.10 (Strongly MLT Π * -capable surrogate model). For a fixed set of phrasebooks Π * = {π * i } d i=1 in MLT(n, d), we say a surrogate model SURR-MLT {Wi} d i=1 (•) is strongly MLT Π * -capable if for any input sequence s 1 ∈ A L where L is even, we have SURR-MLT {Wi} d i=1 (0, . . . , 0, Mat(s 1 )) = Mat(MLT Π * (s 1 )).
this section cite: []

Section: Now we can show the following properties of the surrogate model SURR-MLT {Wi}
d i=1 (•): Lemma G.11. When ∥W i ∥ 0 < 1 2 for all i ∈ [d], SURR-MLT {Wi} d i=1 is strongly MLT(d, n)-ICL-capable. Proof. Fix any set of phrasebooks Π = {π i } d i=1 and its corresponding matrix representations {Matrix(π i )} d i=1 . Since ∥W i ∥ 0 < 1 2 , for any column k, no entries in W (k) i can flip the argmax of Matrix(π i ) (k) + W (k) i
away from being arg max Matrix(π i ) (k) . Therefore we have
P i = HardMax(Matrix(π i ) + W i ) = Matrix(π i ) for all layers i ∈ [d].
By Lemma G.5, for all i ∈ [d] we have P i = Matrix(π i ) recovering T πi . Hence for any input sequence s 1 ∈ A L , we have SURR-
MLT {Wi} d i=1 ({(π 1 )} , . . . , {(π d )} , Mat(s 1 )) = Mat(MLT Π (s 1 )). Lemma G.12. Fix a target set of phrasebooks Π * = {π * i } d i=1 , when HardMax (W i ) = Matrix(π * i ) for all i ∈ [d], SURR-MLT {Wi} d i=1 (•) is strongly MLT Π * -capable.
Proof. By Lemma G.5, for all i ∈ [d] we have
P i = HardMax(W i + 0) = Matrix(π * i ) recovering T π * i . Hence for any input sequence s 1 ∈ A L , we have SURR-MLT {Wi} d i=1 (0, . . . , 0, Mat(s 1 )) = Mat(MLT Π * (s 1 )).
Lemma G.11 suggests that when the weight matrices have small initializations, the model has perfect ICL capability. Meanwhile Lemma G.12 suggests that when the weights W i recover Matrix(π * i ) in the column-wise hard-max sense, then the surrogate model can perform MLT Π * when no context is being provided (C i = 0).
this section cite: []

Section: G.2. Learning Π * in MLT(d, n) with Heuristics Search
In this section, we provide a brute-force algorithm that can learn any target set of phrasebooks Π * in MLT(d, n) using a single "short" sequence whose length is not exponentially dependent on d.
Before proceeding to the details, let us first investigate more on the nature of MLT and the surrogate model. For simplicity of notations, given Π * = {π * 1 , . . . , π * d }, we denote the general translation operator Matrix as P , and denote the translation operator Matrix (π * 1 ) as W * i . Also, with slight abuse of notations we use
MLT Π * (V 1 ) to denote Mat MLT Π * (Mat -1 (V 1 )) .
First, we characterize the input sequence that is good for providing learning signals.
Definition G.13 (Π * -coverable input). Fix a target set of phrasebooks Π * = {π * 1 , . . . , π * d } in MLT(d, n) and an input matrix V 1 ∈ R n 2 ×L , let Ṽ * 1 , V * 2 , Ṽ * 2 , . . . , Ṽ * d , V * d+1 ∈ R n 2 ×L be the intermediate outputs when applying MLT Π * on V 1 . We say V 1 is Π * -coverable if for all levels i ∈ [d], Ṽ * i is of rank-n 2 .
Note that as a matrix with only one-hot columns, Ṽ * i being rank-n 2 suggests that for all k ∈ [n 2 ], there exists some column j ∈ [L] such that Ṽ * (j)
i = e k . In the context of the translation process, it means that the correct translation process of a Π * -coverable input V 1 would require all entries of all phrasebooks in Π * .
Next we will show that if we use the context to condition all translation operators P l of the surrogate model to be P (π * l ) for all but one level l ∈ [d]\ {i} (i as the unconditioned level), then for the surrogate model to correctly perform MLT Π * , the operator for the unconditioned level i must also be equal to P (π * i ) . Lemma G.14 (Uniqueness of a single P i when conditioning all other levels).
P i = HardMax (C i + W i ) = W * i .
Proof. This lemma is a direct consequence of the bijective property of the translation process shown in Lemma E.1.
Let Ṽ * 1 , V * 2 , Ṽ * 2 , . . . , Ṽ * d , V * d+1 ∈ R n 2 ×L
be the intermediate outputs when applying MLT Π * on V 1 , and let Ṽ1 , V 2 , Ṽ2 , . . . , Ṽd , V d+1 ∈ R n 2 ×L be the intermediate outputs when applying SURR-MLT {Wi} d i=1 (C 1 , C 2 , . . . , C d , •) as described. Since we assume P l = P (π * l ) for all l < i, we have
V i = V * i and therefore Ṽi = Ṽ * i . On the other end, since SURR-MLT {Wi} d i=1 (C 1 , C 2 , . . . , C d , V 1 ) = MLT Π * (V 1 ) = V *
d+1 and P l = P (π * l ) for all l > i, by the invertible property of the translation process (Lemma E.1) we must have Ṽi+1 = Ṽ * i+1 and thus V i+1 = V * i+1 . Combining both ends we know that
P i Ṽi = V i+1 = V * i+1 = W * i Ṽ * i = W * i Ṽi .(21)
Since Ṽi = Ṽ * i is rank n 2 by the Π * -coverable assumption and P i ∈ R n 2 ×n 2 , it must be so that P i = W * i .
If we further condition on the held-out level P i such that we only leave one column of P
(j) l = W * (j) l for all (l, j) ∈ [d] × [n 2 ]\ {(i, k)}. Then SURR-MLT {Wi} d i=1 (C 1 , C 2 , . . . , C d , V 1 ) = MLT Π * (V 1 ) if and only if P (k) i = HardMax C (j) i + W (j) i = W * (k) i .
This suggests that if we condition everything else except for one column of the translation operator, then to match the final output on a Π * -coverable sequence, the model must recover the held-out column to be the same as the ground truth in the set of phrasebooks. Now we can introduce the search algorithm, which simply enumerate over all translation columns W (j) i as learning target, generate a contextual information that only leaves that column unconditioned, and search over all possible one-hot vectors for W (j) i until the output matches with MLT Π * . Once the output matches, by Corollary G.15 we know HardMax(W
(j) i ) recovers W * (j) i
and we move on to the next learning target. The algorithm can be formalized as follows:
Algorithm 2 Context-Enhanced Searching Algorithm for MLT(d, n) 1: Input: 2: input V 1 ∈ R n 2 ×L , label V * d+1 ∈ R n 2 ×L , descriptive text W * 1 , . . . , W * d ∈ R n 2 ×n 2 3: 4: Initialize W 1 , . . . , W d ← 0 # Start with zero initialization 5: for i = 1 to d do 6: for k = 1 to n 2 do 7: Initialize C i(k) ← W * i (I n 2 -diag(ē k )) # Create masked context matrix 8: # Search Loop 9:
for a = 1 to n 2 do 10:
W (k) i ← ēa # Search over one-hot columns 11: V d+1 ← SURR-MLT {Wi} d i=1 (W * 1 . . . , W * i-1 , C i(k) , W * i+1 . . . , W * d , V 1 ) 12: if V d+1 = V * d+1then
13: break # Break when found the right column 14: end if 15: end for 16: end for 17: end for 18: Return W 1 , . . . , W d . Theorem G.16 (Learning Π * with context-enhanced search with Π * -coverable input). For any target set of phrasebooks Π * = {π * 1 , . . . , π * d } in MLT(d, n), given an Π * -coverable input V 1 and the corresponding ground truth label V * d+1 = MLT Π * (V 1 ), Algorithm 2 terminates with W i = W * i for all i ∈ [d] with O(n 4 d) forward passes through the surrogate model. W * 1 . . . , W * i-1 , C i(k) , W * i+1 . . . , W * d will correctly condition all columns of P 's except for the P (k) i since C (k)
i(k) = W * i (I n 2 -diag(ē k )) (k) = 0.(22)
Thus by Corollary G.15, we know that the search loop will terminate when it finds W
(k) i = W * (k) i . The newly added column provides the correct inductive hypothesis on W (j) l = W * (j) l for the next enumeration step.
By induction to i = d and k = n 2 , we will be able to recover
W i = W * i for all i ∈ [d].
Given that we can learn W i effectively with Π * -coverable input, how should we construct such inputs? It turned out that with high probability, short random strings suffices.
Lemma G.17 (Distribution of intermediate sequences). Fix a target set of phrasebooks Π * = {π * 1 , . . . , π * d } in MLT(d, n). Let V 1 ∈ R n 2 ×L be a random input matrix to MLT(d, n) such that each column V (j) 1 is i.i.d. sampled from U({ē k } n 2 k=1 ) (the uniform distribution over one-hot vectors {ē k } n 2 k=1 ), the columns of intermediate random sequences Ṽ * 1 , V * 2 , Ṽ * 2 , . . . , Ṽ * d , V * d+1 ∈ R n 2 ×L obtained by passing the input V 1 through MLT Π * also follow the same i.i.d. uniform distribution.
Proof. We will prove the claim by induction on depth i. Let the inductive hypothesis be that columns in V i independently follow an uniform distribution over the one-hot vectors {ē k } n 2 k=1 . Note that the base case is just the assumption. Now we prove for the inductive step. For any j ∈ [L], we can write V k=1 ), sampling Ṽi to be rank n 2 becomes identical to the classic coupon collection problem (see Lemma G.29 from Motwani (1995)). Thus we have the following bound:
Lemma G.18 (Short Π * -coverable random sequence). A random sequence V 1 of length L ≥ 2n 2 log nd δ is Π * -coverable with probability at least 1 -δ. Proof. Let the event A i denote that Ṽi is not rank n 2 . By Lemma G.17, each column of Ṽi is i.i.d. distributed following U({ē k } n 2 k=1
). Thus making Ṽi being rank n 2 is equivalent to a coupon collecting problem (Motwani, 1995) with set size n 2 . By Lemma G.29 we know that with L = 2n 2 log nd δ , P [A i ] ≤ δ d . Thus by a simple union bound the probability that Ṽi being not Π * -coverable is
P d i=1 A i ≤ d i=1 P [A i ] ≤ d δ d = δ.(23)
Now we can apply the above result and extend Theorem G.16.
Corollary G.19 (Learning Π * with random input using heuristics search).
For any target set of phrasebooks Π * = {π * 1 , . . . , π * d } in MLT(d, n), with probability at least 1 -δ over a uniformly random input V 1 of length L = 2n 2 log nd δ , Algorithm 2 provided with ground truth label V * d+1 = MLT Π * (V 1 ) terminates with W i = W * i for all i ∈ [d] with O(n 4 d) forward passes through the surrogate model.
this section cite: ['b21']

Section: G.3. Learning Π * in MLT(2, n) with Surrogate Gradient Descent
In this section we take the analysis one step beyond the heuristics searching regime. We will show that any set of phrasebooks Π * = {π * 1 , π * 2 } can be sample-efficiently learned by a gradient-descent based algorithm. In this particular case, the surrogate model is parameterized by
V 3 = SURR-MLT {Wi} d i=1 (C 1 , C 2 , V 1 ) ≜ HardMax(C 2 + W 2 ) Shift (HardMax(C 1 + W 1 ) Shift (V 1 )) . (24)
We start with any weight initializations
W (0) 1 , W (0) 2 ∈ R n 2 ×n 2 satisfying ∥W (0) 1 ∥ 1 < 1 2 , ∥W (0) 2 ∥ 1 < 1 2 , by Lemma G.11 the initialization is strongly MLT(d, n)-ICL-capable.
For simplicity, we denote the ground truth permutation matrix induced by π * 1 as W * 1 ≜ P (π * 1 ) and similarly the ground truth permutation matrix induced by π * 2 as W * 2 ≜ P (π * 2 ) . From Lemma G.12 we know that the learning is successful if we have HardMax (W 1 ) = W * 1 and HardMax (W 2 ) = W * 2foot_4 (i.e. the maximum index of each weight column agrees with that of the ground truth).
We employ a layer-wise gradient descent algorithm for the learning process. The algorithm takes in a single fixed sequence s 1 with matrix representation V 1 and its corresponding ground truth label
V * 3 ≜ Mat(MLT Π * (s 1 )) = SURR-MLT (W (0) 1 ,W (0) 2 ) (W * 1 , W * 2 , V 1 ).(25)
Given the input and label, we employ the following gradient descent based algorithm to update the weights:
The training happens in a layer-wise and column-wise fashion: We first freeze W 2 and set W 1 as the trainable parameter.
For each entry k ∈ [n 2 ], we create a context matrix C 1(k) ≜ W * 1 (I n 2diag(e k )) which essentially creates a copy of W * 1 except of setting the k-th column to be zero. Then we take a forward pass through the surrogate model with the one-column dropped-out context and get output V 3(1,k) ≜ SURR-MLT (W1,W2) (C 1(k) , W * 2 , V 1 ). Here the subscript • (1,k) denotes the final output when the i-th column of the first context matrix is being dropped.
The weight update follows a surrogate gradient update scheme where we use the MSE loss: L = ∥V 3(1,k) -V * 3 ∥ 2 2 . Since it is difficult to take gradient through the hardmax function, we instead compute the gradient of the loss with respect to the translation matrix P 1 = HardMax W 1 + C 1 (k) and apply the update W
(k) 1 ← W (k) 1 -∂L ∂P (k) 1
. We apply such gradient update twice for each dropped column k.
For the second layer, we freeze the first layer W 1 and apply one-column dropouts to the C 2 . Similarly we apply the surrogate gradient update W 2 ← W 2 -∂L ∂P2 but we only need one gradient step per column. We claim the surrogate gradient descent update can correctly recover P * 1 and P * 2 similar to the heuristics search case.
Theorem G.24 (Learning Π * with context-enhanced surrogate GD with Π * -coverable input). To prove for Theorem G.24, we will carefully analyze the learning of the first layer and second layer respectively, and provide a similar induction argument as in the proof for the heuristics search case.
Algorithm 3 Context-Enhanced Layerwise Gradient Descent
1: Input: input V 1 ∈ R n 2 ×L , label V * 3 ∈ R n 2 ×L , descriptive text W * 1 , W * 2 ∈ R n 2 ×n 2 , init W (0) 1 , W (0) 2 ∈ R n 2 ×n 2 2:
3: # Train the first layer 4: for k = 1 to n 2 do 5:
C 1(k) ≜ W * 1 (I n 2 -diag(e k ))
# Create context matrix with k-th column dropped.
6:
for t = 1 to 2 do 7:
V 3(1,k) ← SURR-MLT (W1,W2) (C 1(k) , W * 2 , V 1 ) # Forward pass 8: L ← ∥V 3(1,k) -V * 3 ∥ 2 2 9: W (k) 1 ← W (k) 1 -∂l ∂P (k) 1
# Surrogate gradient update 10:
end for 11: end for 12: 13: # Train the second layer 14: for k = 1 to n 2 do 15:
C 2(k) ≜ W * 1 (I n 2 -diag(e k ))
# Create context matrix with k-th column dropped.
16:
V 3(2,k) ← SURR-MLT (W1,W2) (W * 1 , C 2(k) , V 1 ) # Forward pass 17: L ← ∥V 3(2,k) -V * 3 ∥ 2 2 18: W (k) 2 ← W (k) 2 -∂l ∂P (k) 2 # Surrogate gradient update 19: end for 20: Return W 1 , W 2 G.
this section cite: []

Section: LEARNING THE FIRST LAYER
To study the learning process we first need to compute the closed-form gradient ∂L ∂P (k) 1
, which requires the following lemma:
Lemma G.20 (Gradient with respect to incorrect column in P 1 ).
When only the k-th column of translation matrix P is used in the forward pass, there exists α ∈ Z + and β ∈ N such that the gradient of L with respect to P
(k) 1 is of the form ∂L ∂P (k) 1 =      (2α + 2β) (1 n ⊗ e b ) -(2α + 2β) (1 n ⊗ e b * ) + 2β (e a * ⊗ 1 n ) if a * = a, b * ̸ = b (2α + 2β) (e a ⊗ 1 n ) -(2α + 2β) (e a * ⊗ 1 n ) + 2β (1 n ⊗ e b * ) if a * ̸ = a, b * = b (2α + 2β) (e a ⊗ 1 n ) + (2α + 2β) (1 n ⊗ e b ) -2α (e a * ⊗ 1 n ) -2α (1 n ⊗ e b * ) if a * ̸ = a, b * ̸ = b.
Proof of Lemma G.20. In this proof we use ēa (long one-hot) to denote the one-hot vector in R n 2 with 1 on the a-th index and use e a (short one-hot) to denote the one-hot vector in R n with 1 on the a-th index. We use Ṽ1 , V 2 , Ṽ2 and V 3 to denote the intermediate sequences attained with translation matrix P 1 . Now we can proceed to the gradient calculations. First note that with L(P 1 ) = ∥V 3 -V * 3 ∥ 2 2 , by chain rule we have
∂L ∂P 1 = L j=1 ∂V (j) 3 ∂P 1 ⊤ ∂∥V (j) 3 -V * (j) 3 ∥ 2 2 ∂V (j) 3 = 2 L j=1 ∂V (j) 3 ∂P 1 ⊤ V (j) 3 -V * (j) 3 .(26)
Specifically for each column l ∈ [n 2 ] of P 1 we have
∂L ∂P (l) 1 = 2 L j=1 ∂V (j) 3 ∂P (l) 1 ⊤ V (j) 3 -V * (j) 3 .(27)
This is the most complicated case since the loss is contributed by two different paths. We can first decompose the negative residual as
V (j) 3 -V * (j) 3 = W * 2 QP (k) 1 ⊙ Q ⊤ P (k) 1 -W * 2 QP * (k) 1 ⊙ Q ⊤ P * (k) 1 = W * 2 QP (k) 1 ⊙ Q ⊤ P (k) 1 -W * 2 QP (k) 1 ⊙ Q ⊤ P * (k) 1 + W * 2 QP (k) 1 ⊙ Q ⊤ P * (k) 1 -W * 2 QP * (k) 1 ⊙ Q ⊤ P * (k) 1 = W * 2 diag QP (k) 1 Q ⊤ P (k) 1 -P * (k) 1 + W * 2 diag Q ⊤ P * (p) 1 Q P (k) 1 -P * (k) 1 . (40)
Combining with Equation (32), we have
∂V (j) 3 ∂P 1 ⊤ V (j) 3 -V * (j) 3 = W * 2 diag Q ⊤ P (k) 1 Q + W * 2 diag QP (k) 1 Q ⊤ ⊤ W * 2 diag QP (k) 1 Q ⊤ P (k) 1 -P * (k) 1 + W * 2 diag Q ⊤ P * (k) 1 Q P (k) 1 -P * (k) 1 = Q ⊤ diag Q ⊤ P (k) 1 diag QP (k) 1 Q ⊤ P (k) 1 -P * (k) 1 (a) + Qdiag QP (k) 1 diag QP (k) 1 Q ⊤ P (k) 1 -P * (k) 1 (b) + Q ⊤ diag Q ⊤ P (k) 1 diag Q ⊤ P * (k) 1 Q P (k) 1 -P * (k) 1 (c) + Qdiag QP (k) 1 diag Q ⊤ P * (k) 1 Q P (k) 1 -P * (k) 1 .(d)
It follows that
Q ⊤ diag Q ⊤ P (k) 1 diag QP (k) 1 Q ⊤ = (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n (diag (e b ) ⊗ diag (e a )) (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n = (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n (e b ⊗ diag (e a )) I ⊤ n ⊗ 1 ⊤ n = (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n e b ⊗ e ⊤ a = (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n e b e ⊤ a ⊗ 1 = (1 n ⊗ I n ) e b e ⊤ a ⊗ 1 ⊤ n (42
)
Plugging into (a) we have
(a) = Q ⊤ diag Q ⊤ P (k) 1 diag QP (k) 1 Q ⊤ P (k) 1 -P * (k) 1 = (1 n ⊗ I n ) e b e ⊤ a ⊗ 1 ⊤ n (e a ⊗ e b ) -(1 n ⊗ I n ) e b e ⊤ a ⊗ 1 ⊤ n (e a * ⊗ e b * ) = (1 n ⊗ I n ) e b e ⊤ a e a ⊗ 1 -(1 n ⊗ I n ) e b e ⊤ a e a * ⊗ 1 = (1 n ⊗ I n ) 1 ⊗ e b e ⊤ a e a -(1 n ⊗ I n ) 1 ⊗ e b e ⊤ a e a * = 1 n ⊗ e b when a ̸ = a * 0 otherwise (43
)
Now we are ready to provide the gradient expression for the loss over the entire sequence. Observe that for every consecutive sequence of m columns {V
* (j) 1 , V * (j+1) 1 , . . . , V * (j+m-1) 1
} that all equals to ēk , it will result in one incorrect column V (j-1) 3 in case 2, one incorrect column V (j+m-1) 3 in case 3, and m -1 incorrect columns (V
(j) 3 , . . . , V (j+m-2) 3 ) in case 4. V (j-1) 1 → Ṽ (j-1) 1 (= ēp ̸ = ēk ) P (p) 1 -→ V (j-1) 2 → Ṽ (j-1) 2 W * 2 -→ V (j-1) 3 (case 2) ↗ ↗ V (j) 1 → Ṽ (j) 1 (= ēk ) P (k) 1 -→ V (j) 2 → Ṽ (j) 2 W * 2 -→ V (j) 3 (case 4) ↗ ↗ . . . . . . ↗ ↗ V (j+m-2) 1 → Ṽ (j+m-2) 1 (= ēk ) P (k) 1 -→ V (j+m-2) 2 → Ṽ (j+m-2) 2 W * 2 -→ V (j+m-2) 3 (case 4) ↗ ↗ V (j+m-1) 1 → Ṽ (j+m-1) 1 (= ēk ) P (k) 1 -→ V (j+m-1) 2 → Ṽ (j+m-1) 2 W * 2 -→ V (j+m-1) 3 (case 3) ↗ ↗ V (j+m) 1 → Ṽ (j+m) 1 (= ēq ̸ = ēk ) P (q) 1 -→ V (j+m) 2 → Ṽ (j+m) 2 W * 2 -→ V (j+m) 3
(case 1 or 2) For illustration, one can refer to the computation graph in Figure 12. In the graph, green entries agrees with the counterfactual values with correct P * ( k)  1, Red and pink entries are incorrect entries where red entries are consequence solely dependent on P (k) 1 (case 4) and pink entries depend on other correct columns (case 2,3).
Assume that in total there are α columns in V 3 under case 2, α columns in V 3 under case 3, and β columns in V 3 under case 4, then by Equation ( 27) the total gradient can be expressed as follows:
• When a = a * , b ̸ = b * : ∂L ∂P (k) 1 = 2α (1 n ⊗ (e b -e b * ) + 2α (e a -e a * ) ⊗ 1 n ) + 2β (1 n ⊗ (e b -e b * ) + e a * ⊗ 1 n ) = (2α + 2β) (1 n ⊗ e b ) -(2α + 2β) (1 n ⊗ e b * ) + 2β (e a * ⊗ 1 n ) . (52) • When a ̸ = a * , b = b * : ∂L ∂P (k) 1 = 2α (1 n ⊗ (e b -e b * )) + 2α ((e a -e a * ) ⊗ 1 n ) + 2β (1 n ⊗ e b * + (e a -e a * ) ⊗ 1 n ) = (2α + 2β) (e a ⊗ 1 n ) -(2α + 2β) (e a * ⊗ 1 n ) + 2β (1 n ⊗ e b * ) .
(53) and therefore we have
Q ⊤ diag Q ⊤ v diag Q ⊤ v Q = (1 n ⊗ I n ) I n ⊗ e ⊤ a (I n ⊗ e a ) (1 ⊤ n ⊗ I ⊤ n ) = (1 n ⊗ I n ) (I n ⊗ 1) (1 ⊤ n ⊗ I ⊤ n ) since e ⊤ a e a = 1 = (1 n ⊗ I n ) (1 ⊗ I n ) (1 ⊤ n ⊗ I ⊤ n ) = (1 n ⊗ I n )(1 ⊤ n ⊗ I ⊤ n ) = (1 n 1 ⊤ n ) ⊗ I n(70)
Lemma G.28. For any one-hot vector
v = e a ⊗ e b ∈ R n 2 , Qdiag (Qv) diag (Qv) Q ⊤ = I n ⊗ (1 n 1 ⊤ n ).
Proof. This proof is very similar to the proof for Lemma G.27. By Lemma G.26,
Q ⊤ v = 1 n ⊗ e a . Therefore diag (Qv) = diag (e b ) ⊗ diag (1 n ) = diag (e b ) ⊗ I n . Thus diag (Qv) Q ⊤ = (diag (e b ) ⊗ I n ) (1 n ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n = (e b ⊗ I n ) I ⊤ n ⊗ 1 ⊤ n(71)
and therefore we have
Qdiag (Qv) diag (Qv) Q ⊤ = (I n ⊗ 1 n ) e ⊤ b ⊗ I n (e b ⊗ I n ) (I ⊤ n ⊗ 1 ⊤ n ) = (I n ⊗ 1 n ) (1 ⊗ I n ) (I ⊤ n ⊗ 1 ⊤ n ) since e ⊤ b e b = 1 = (I n ⊗ 1 n ) (I n ⊗ 1) (I ⊤ n ⊗ 1 ⊤ n ) = (I n ⊗ 1 n )(I ⊤ n ⊗ 1 ⊤ n ) = I n ⊗ (1 n 1 ⊤ n ).(72)
Lemma G.29 (Tail Bound for Coupon Collector Problem (Motwani, 1995)).
For a set S of size n, with probability at least 1 -δ one can cover all unique elements of S in n log n δ independent uniformly random sampling trials from S.
this section cite: ['b21']

Section: G.5. Learning Π * in MLT Π * with Gradient Descent (Empirical Evidence)
In this section we provide more details on empirically optimizing the simple surrogate model SURR-MLT {Wi} d i=1 , which was only briefly discussed in the main text by the end of Section 5.1. We will first introduce the approximations we made to the surrogate model to make gradient-based optimization easy and stable, then we will present empirical results on the model learning target sets of phrasebooks MLT Π * in MLT(5, 10), MLT(10, 10), and even MLT(20, 10).
this section cite: []

Section: G.5.1. APPROXIMATED LATENT MODEL FOR GD
Recall that with input sequence represented by V 1 ∈ R n 2 ×L , the surrogate model for a depth-d translation is being recursively defined by the translation + shifting operations
V i+1 = HardMax(C i + W i ) Shift(V i )(73)
until we reach V d+1 . While this model captures the essence of transition from ICL capability to memorization of specific set of phrasebooks, HardMax is making it not directly differentiable and hard to optimize. To address this issue, we approximate it with an column-wise softmax function with very low temperature (T = 1/25). The recursive definition in the approximated model is then
Ṽi+1 = SoftMax(25(C i + W i )) Shift( Ṽi ).(74)
We denote the recursive surrogate model with the softmax substitution as SURR-MLT {Wi} d i=1 (C 1 , C 2 , . . . , C d , V 1 ) where
Ṽd+1 = SURR-MLT {Wi} d i=1 (C 1 , C 2 , . . . , C d , V 1 ) ≜ SoftMax(25C d + 25W d ) Shift SoftMax(25C d-1 + 25W d-1 ) Shift • • • SoftMax(25C 1 + 25W 1 ) Shift( Ṽ1 ) • • • .(75)
We define the objective function as the column-wise cross-entropy loss between the final output and the input. Namely for input V 1 with prediction Ṽd+1 and ground truth label V * d+1 , the loss is computed as
L = L k=1 CrossEntropy( Ṽ (k) d+1 , V * (k) d+1 ).(76)
We follow the same masking (dropout) curriculum as described in Appendix G.2 and Appendix G.3, that at each step we zero-out a single column from a single context matrix C i . We experiment on two gradient update schemes:
• Layer-wise Training: at each step, if we are masking a column on C i , we only compute the gradient with respect to W i and update it. This training is more akin to the theoretical analysis described in Appendix G.3.
• Full Parameter Training: at any step, we compute the gradient with respect to each of the weight matrices and update all parameters. This is more akin to the real gradient-based training as we do not have the heuristics for localized update.
To allow for fast and stable training, we adopt a very large learning rate of η = 100 and apply parameter clipping between [0, 1] after each update. The complete algorithm is described as follows:
Algorithm 4 Layerwise Gradient Descent with Context-Enhanced Learning For Optimizing SURR-MLT 1: Input:
2: input V 1 ∈ R n 2 ×L , label V * d+1 ∈ R n 2 ×L , descriptive text W * 1 , . . . , W * d ∈ R n 2 ×n 2 , learning rate η, total steps T 3: 4: Initialize W 1 , . . . , W d ← 0 # Start with zero initialization 5: for t = 1 to T do 6: i ← ⌊(t -1)/n 2 ⌋%d + 1
# Get the layer to be masked 7:
k ← ((t -1)%n 2 ) + 1 # Get the column index to be masked 8:
Initialize C i(k) ← W * i (I n 2 -diag(ē k )) # Create masked context matrix 9: Ṽd+1 ← SURR-MLT {Wi} d i=1 (W * 1 . . . , W * i-1 , C i(k) , W * i+1 . . . , W * d , V 1 ) 10: L ← CrossEntropy( Ṽd+1 , V * d+1 ) 11: W i ← W i -η∇ Wi L
# Update the weight for the layer with mask 12: end for 13: Return W 1 , . . . , W d .
this section cite: []

Section: Algorithm 5
Full Parameter Gradient Descent with Context-Enhanced Learning For Optimizing SURR-MLT 1: Input:
2: input V 1 ∈ R n 2 ×L , label V * d+1 ∈ R n 2 ×L , descriptive text W * 1 , . . . , W * d ∈ R n 2 ×n 2
, learning rate η, total steps T
3: 4: Initialize W 1 , . . . , W d ← 0 # Start with zero initialization 5: for t = 1 to T do 6:
i ← ⌊(t -1)/n 2 ⌋%d + 1 # Get the layer to be masked 7:
k ← ((t -1)%n 2 ) + 1 # Get the column index to be masked 8:
Initialize C i(k) ← W * i (I n 2 -diag(ē k )) # Create masked context matrix 9: Ṽd+1 ← SURR-MLT {Wi} d i=1 (W * 1 . . . , W * i-1 , C i(k) , W * i+1 . . . , W * d , V 1 ) 10: L ← CrossEntropy( Ṽd+1 , V * d+1 ) 11: for l = 1 to d do 12: W l ← W l -η∇ W l L #
Update the weight for all layers 13: end for 14: end for 15: Return W 1 , . . . , W d . 0 2000 4000 Steps 0.0 0.2 0.4 0.6 0.8 1.0 Layer-wise GD 0 2000 4000 Steps 0.0 0.2 0.4 0.6 0.8 1.0 Full Param GD W 1 W 2 W 3 W 4 W 5 (a) MLT(5, 10) 0 2500 5000 7500 10000 Steps 0.0 0.2 0.4 0.6 0.8 1.0 Layer-wise GD 0 2500 5000 7500 10000 Steps 0.0 0.2 0.4 0.6 0.8 1.0 Full Param GD W 1 W 2 W 3 W 4 W 5 W 6 W 7 W 8 W 9 W 10 (b) MLT(10, 10) 0 2500 5000 7500 10000 12500 15000 17500 20000 Steps 0.0 0.2 0.4 0.6 0.8 1.0 Layer-wise GD 0 2500 5000 7500 10000 12500 15000 17500 20000 Steps 0.0 0.2 0.4 0.6 0.8 1.0 Full Param GD W 1 W 2 W 3 W 4 W 5 W 6 W 7 W 8 W 9 W 10 W 11 W 12 W 13 W 14 W 15 W 16 W 17 W 18 W 19 W 20 (c) MLT(20, 10) After right shift operation, we will represent the output of the self-attention computation as <THINK>, [o 2 ; 0], [o 3 ; 0], • • • , [o L/2+1
; 0], and we will ignore the <THINK> embedding. Note that the second half of the output embeddings will still contain 0s and we will ignore them in the current computation. Then, the above computation can be rephrased as
o j = QV (j-1) i ⊙ Q ⊤ V (j) i , for all 2 ≤ j ≤ L/2. o L/2+1 = QV (L/2) i ⊙ Q ⊤ V (1) i
Self-attention layer: The computation of o j , for 2 ≤ j ≤ L/2, requires the computation of (1
n ⊗ I n ) ⊤ V (j-1) i and (I n ⊗ 1 n ) V (j)
i . This will require 2 attention heads, one head that attends to itself, and another that attends to previous embedding at each position. We will outline both below. We will require one additional head, as computing o L/2+1 will require the model to compute
(I n ⊗ 1 n ) V (1) i . 1. Attention Head 1 computes (1 n ⊗ I n ) ⊤ V (j-1) i at position j for all 2 ≤ j ≤ L/2 + 1.
This can be done using a self-attention head (Definition H.1) that sets query and key matrices W query , W key , and biases {b i } t≤i≤0 such that the attention score between embeddings at any two positions p 1 , p 2 is given as follows:
a p1,p2 = 1, if p 2 -p 1 = -1, 0 otherwise W value is set such that for any input x, the output of W value x is given by (W value x) p1 = n-1 j=0 x n•j+p1 .
In simple words, this operation simply adds up the values in dimensions p 1 , p 1 + n, p 1 + 2n, • • • and stores them at position p 1 for all 1 ≤ p 1 ≤ n.
this section cite: []

Section: Attention Head 2 computes (I
n ⊗ 1 n ) V (j) i
at position j for all 2 ≤ j ≤ L/2. This can be done using a self-attention head (Definition H.1) that sets W query , W key , W value and biases {b i } t≤i≤0 such that the attention score between embeddings at any two positions p 1 , p 2 is given as follows: a p1,p2 = 1, if p 2 -p 1 = 0, 0 otherwise W value is set such that for any input x, the output of W value x is given by
(W value x) p1+n = n j=1 x j+p1n-n .
In simple words, this operation simply adds up the values in dimensions 1 + (p 1 -1)n, 2 + (p 1 -1)n, • • • and stores them at dimension p 1 + n for all 1 ≤ p 1 ≤ n.
this section cite: []

Section: Attention head 3 will compute (I
n ⊗ 1 n ) V (1) i
and store in o L/2+1 . This can be done by a self-attention layer which activates only between positions p 1 and p 2 that represent the start and the end tokens of the sequence, i.e. contain b 1 and b 2 as start and end indicator embeddings, and is 0 otherwise. W value is set same as attention head 2.
The output of the three heads are simply added up. Hence, at each position 2 ≤ j ≤ L/2 + 1, the output o j has (1
n ⊗ I n ) ⊤ V (j-1) i
in [1, n] dimensions and
(I n ⊗ 1 n ) V (j%L) i in [n + 1, 2n] dimensions.
this section cite: []

Section: MLP layer:
The objective with the MLP layer (Definition H.2) will be to multiply (1 1, 2n] dimensions in each position j. This can be done by using an MLP layer with GELU activation by using Lemma H.3. The weights of the MLP layer are set as follows: W inner is set such that for all input x, we have
n ⊗ I n ) ⊤ V (j-1) i present in [1, n] dimensions and (I n ⊗ 1 n ) V (j%L) i present in [n +
(W inner x) i = 1 N x i + 1 N x n+i , for all 1 ≤ i ≤ n, (W inner x) i+n = 1 N x i , for all 1 ≤ i ≤ n, (W inner x) i+2n = 1 N x n+i , for all 1 ≤ i ≤ n. W outer is set such that for all x ∈ R 3n (W outer x) i = N 2 (x i -x i+n -x i+2n ), for all 1 ≤ i ≤ n.
All other coordinates in these matrices are set as 0s. N is set as a large number (say 100). By Lemma H.3, the output of the MLP layer will be <THINK>, o 2 , • • • , o L/2+1 , with o j containing
Ṽ (j-1) i + O(N -4 ) := (1 n ⊗ I n ) ⊤ V (j-1) i ⊙ (I n ⊗ 1 n ) V (j%L) i + O(N -4 ) at each position 2 ≤ j ≤ L/2.
this section cite: []

Section: Embedding Name Dimension size First segment values Second segment values (In-context information)
(Input query sequence embeddings)
Token 2n 2 {[e j ; C 1 e j ]} n 2 j=1 , • • • , {[e j ; C d e j ]} n 2 j=1 <THINK>, [ Ṽ (1) i ; 0], • • • , [ Ṽ (L/2) i ; 0] Context matrix index indicator d {l 1 } [1,n 2 ] , {l 2 } [1,n 2 ] , • • • {l d } [1,n 2 ] 0, 0, • • • , 0 Start and End indicator 2 0, • • • , 0 0, b 1 , 0, • • • , 0, b 2 Segment 2 g 1 , • • • , g 1 0, g 2 , • • • , g 2
Table 10. Output of CIRCULAR SHIFT-MODULE in MLT-MODULE that simulates Circular shift, i.e. computes Shift(Vi) at second segment token embeddings. <THINK> represents a null output and won't be attended to in the future modules. We ignore this symbol for simplicity, when analyzing any module. ej indicates a one-hot n 2 dimensional vector that contains 1 in dimension j. The attention between any two input sequence embedding o j and o j ′ is computed as 0s. The distinction between the attention scores of pairs of embeddings in second segment, o j and o j ′ , v/s attention scores between a token embedding in second segment and a token embedding in first segment, o j and [e r ; C ℓ e r ], can be done by using the segment indicator embeddings g 1 and g 2 used to differentiate token embeddings in first segment and the input sequence embedding vectors.
Matrix W value is set such that the columns of each C ℓ s are picked from the token embeddings in the first segment: {{[e r ; C ℓ e r ]} n 2 r=1 } d ℓ=1 .
this section cite: []

Section: 2.
The second attention head simply copies the input Ṽ (j-1) i : This can be done with an attention head that attends to itself at each position j and copies Ṽ (j-1) i to output.
The output of the two attention heads are simply added up. The output embeddings will now look as follows: <THINK>, {[C i Ṽ (j-1)
i ; Ṽ (j-1) i ]} L/2 j=1 .
MLP to represent HardMax(C i + W i ) Ṽ (j-1) i : Our current token embeddings at any position j contain both C i Ṽ (j-1) i and Ṽ (j-1) i . The first layer of MLP can be used to compute (C i + W i ) Ṽ (j-1) i by setting the weights of the layer using W i . We simulate HardMax operation as follows: õj / ∥õ j ∥ 2 , where õj = GELU((C i + W i ) Ṽ (j-1)
i )
The ℓ 2 normalization is equivalent to RMSnorm operation (Zhang & Sennrich, 2019). This is an approximation of the HardMax function, which are equivalent only under the following conditions: for each column j 1. either C (j) i or W
(j) i are all 0s.
this section cite: ['b54']

Section: C
(j) i and W
(j) i are both one-hot vectors and they match at the corresponding activated dimension.
this section cite: []

Section: I. Additional Experiment Setups

this section cite: []

Section: I.1. Format of Training Text
Here we present examples of training data used for experiments in Section 3
INPUT: "<|begin of text|> <|begin of text|> <|begin of text|> <|start header id|> user <|end header id|> You are performing a special translation task called language task 2. The subset of dictionaries used are as follows: Dictionary used from language 1 to language 2: D A -> N J; A D -> J I; B C -> O I; C C -> N N; H E -> P K; F E -> M L; G H -> L N; E G -> P J; A F -> K M; E C -> K P; B E -> K I; F D -> M N; E D -> P O; B A -> P L; B D -> I P; D G -> I I; D H -> I O; E F -> L M; H B -> J K; E A -> K J; A H -> L J; G C -> I K; F B -> P I; F G -> J O; Dictionary used from language 2 to language 3: L P -> V X; M K -> R W; L I -> X R; N O -> R R; M O -> U Q; I L -> W V; O K -> Q W; P M -> R U; J I -> V S; I M -> X W; O I -> Q U; N P -> R Q; I N -> Q X; N L -> R S; I J -> Q T; N J -> S T; L O -> U V; P J -> T X; J L -> Q R; P K -> W R; N M -> S U; M I -> Q S; P O -> S S; K K -> U U; P L -> X U; Dictionary used from language 3 to language 4: X U -> Y d; R W -> a c; T Q -> Y Y; U R -> Z Y; T X -> e d; W W -> e Y; U X -> f f; X R -> b Y; Q Q -> b c; S S -> c c; V U -> Z d; R S -> f a; V W -> Y e; R U -> f c; U S -> c a; U W -> c f; W X -> d Y; R Q -> c Z; V Q -> Z f; W S -> b d; V R -> a a; R X -> f Z; U Q -> d c; Q V -> d Z; S W -> Y b; Dictionary used from language 4 to language 5: c a -> m j; b Y -> n k; c f -> k m; c b -> l n; Z d -> g h; d c -> n g; b f -> i k; f Z -> i i; Z Y -> g l; Y a -> l l; b e -> l i; Y Y -> m g; d Z -> g g; d e -> k l; f c -> j g; f f -> k n; b Z -> n h; e a -> i m; c e -> j h; Y d -> i h; Y b -> h i; Y f -> h k; a Y -> k j; a a -> m k; c Y -> h m;
Dictionary used from language 5 to language 6: k g -> p s; h h -> r q; j m -> o o; i h -> q s; l m -> p o; i j -> o q; l j -> q u; k m -> v u; g h -> p p; g n -> v p; n h -> s s; m m -> s o; m k -> v t; m i -> u t; h k -> t o; n k -> r u; j n -> t u; g l -> t s; j g -> v v; h g -> u s; n l -> s u; l k -> q o; h l -> t p; i i -> p q; k i -> r o; l h -> u p;
Now please perform language task 2 translation from the following sequence in language 1 to language 6. Do not use code! You must only reponse in the form: "Sequence in language 1: [the sequence in language 1]; Sequence in language 2: [the sequence in language 2]; Sequence in language 3: [the sequence in language 3]; Sequence in language 4: [the sequence in language 4]; Sequence in language 5: [the sequence in language 5]; Sequence in language 6: [the sequence in language 6]". The sequence you need to translate from language 1 is: * * * * ; Sequence in language 5: l l k m k j n h k n l n h m m g h k i i h i j h h k g h * * * * * * * * * * * * ; Sequence in language 6: q o v t t u t o s u s s s o p p r o q s o q r q p s t p;<|eot id|>" , INPUT: "<|begin of text|> <|begin of text|> <|begin of text|> <|start header id|> user <|end header id|> You are performing a special translation task called language task 2. The subset of dictionaries used are as follows: Dictionary used from language 5 to language 6: k g -> p s; h h -> r q; j m -> o o; i h -> q s; l m -> p o; i j -> o q; l j -> q u; k m -> v u; g h -> p p; g n -> v p; n h -> s s; m m -> s o; m k -> v t; m i -> u t; h k -> t o; n k -> r u; j n -> t u; g l -> t s; j g -> v v; h g -> u s; n l -> s u; l k -> q o; h l -> t p; i i -> p q; k i -> r o; l h -> u p;
C B E F E B D E C B C A H E F B C A D F G B D G H E D E.
Now please perform language task 2 translation from the following sequence in language 1 to language 6. Do not use code! You must only reponse in the form: "Sequence in language 1: [the sequence in language 1]; Sequence in language 2: [the sequence in language 2]; Sequence in language 3: [the sequence in language 3]; Sequence in language 4: [the sequence in language 4]; Sequence in language 5: [the sequence in language 5]; Sequence in language 6: [the sequence in language 6]". The sequence you need to translate from language 1 is: Dictionary used from language 1 to language 2: ; Dictionary used from language 2 to language 3: ; Dictionary used from language 3 to language 4: ; Dictionary used from language 4 to language 5: ; Dictionary used from language 5 to language 6:;
Now please perform language task 2 translation from the following sequence in language 1 to language 6. Do not use code! You must only reponse in the form: "Sequence in language 1: [the sequence in language 1]; Sequence in language 2: [the sequence in language 2]; Sequence in language 3: [the sequence in language 3]; Sequence in language 4: [the sequence in language 4]; Sequence in language 5: [the sequence in language 5]; Sequence in language 6: [the sequence in language 6]". The sequence you need to translate from language 1 is:
this section cite: []

Section: References
Ref_id:b0 Title: Metadata conditioning accelerates language model pre-training Year: (2025)
Ref_id:b1 Title: Finding the dominant winning ticket in pre-trained language models Year: (2022)
Ref_id:b2 Title: Think before you speak: Training language models with pause tokens Year: (2023)
Ref_id:b3 Title: Training large language models to reason in a continuous latent space Year: (2024)
Ref_id:b4 Title: Gaussian error linear units (gelus) Year: (2016)
Ref_id:b5 Title: Learning with side information through modality hallucination Year: (2016)
Ref_id:b6 Title: Patterns for learning with side information Year: (2015)
Ref_id:b7 Title: Copyright violations and large language models Year: (2023)
Ref_id:b8 Title: Efficient noise-tolerant learning from statistical queries Year: (1998)
Ref_id:b9 Title: Pretraining of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b10 Title: Practically secure feistel ciphers Year: (1993)
Ref_id:b11 Title: Learning with side information: Part i Year: (2002)
Ref_id:b12 Title: A-tip: attribute-aware text infilling via pre-trained language model Year: (2022)
Ref_id:b13 Title: How do transformers learn topic structure: Towards a mechanistic understanding Year: (2023)
Ref_id:b14 Title: Internalizing symbolic knowledge for distilling better cot capabilities into small language models Year: (2024)
Ref_id:b15 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b16 Title: Unifying distillation and privileged information Year: (2015)
Ref_id:b17 Title: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b18 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b19 Title: Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity Year: (2021)
Ref_id:b20 Title: Understanding lupi (learning using privileged information) Year: (2018)
Ref_id:b21 Title: Randomized Algorithms Year: (1995)
Ref_id:b22 Title: Progress measures for grokking via mechanistic interpretability Year: (2023)
Ref_id:b23 Title: Differential learning kinetics govern the transition from memorization to generalization during in-context learning Year: (2024)
Ref_id:b24 Title: Taskspecific skill localization in fine-tuned language models Year: (2023)
Ref_id:b25 Title: -context learning of representations Year: (2024)
Ref_id:b26 Title: On the theory of learnining with privileged information Year: (2010)
Ref_id:b27 Title: Let's think dot by dot: Hidden computation in transformer language models Year: (2024)
Ref_id:b28 Title: Measuring and narrowing the compositionality gap in language models Year: (2022)
Ref_id:b29 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b30 Title: Compositional capabilities of autoregressive transformers: A study on synthetic, interpretable tasks Year: (2023)
Ref_id:b31 Title: The mechanistic basis of data dependence and abrupt learning in an in-context classification task Year: (2024)
Ref_id:b32 Title: Learning using partially available privileged information and label uncertainty: application in detection of acute respiratory distress syndrome Year: (2020)
Ref_id:b33 Title: Contextual games: Multi-agent learning with side information Year: (2020)
Ref_id:b34 Title: Prompting gpt-3 to be reliable Year: (2022)
Ref_id:b35 Title: The transient nature of emergent in-context learning in transformers Year: (2023)
Ref_id:b36 Title: What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation Year: (2024)
Ref_id:b37 Title: Asymptotic lower bounds for ramsey functions Year: (1977)
Ref_id:b38 Title: Controllable fast and slow thinking by learning with randomized reasoning traces Year: (2024)
Ref_id:b39 Title: Selective annotation makes language models better few-shot learners Year: (2014)
Ref_id:b40 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b41 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b42 Title: A new learning paradigm: Learning using privileged information Year: (2009)
Ref_id:b43 Title: Learning using privileged information: similarity control and knowledge transfer Year: (2015)
Ref_id:b44 Title: Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization Year: (2024)
Ref_id:b45 Title: Assessing the brittleness of safety alignment via pruning and low-rank modifications Year: (2024)
Ref_id:b46 Title: Emergent abilities of large language models Year: (2022)
Ref_id:b47 Title: Self-adaptive incontext learning: An information compression perspective for in-context example selection and ordering Year: (2022)
Ref_id:b48 Title: End-to-end learning of driving models from large-scale video datasets Year: (2017)
Ref_id:b49 Title: Supervised knowledge makes large language models better in-context learners Year: (2023)
Ref_id:b50 Title: Do large language models latently perform multi-hop reasoning? arXiv preprint Year: (2024)
Ref_id:b51 Title: Self-attention networks can process bounded hierarchical languages Year: (2021)
Ref_id:b52 Title: Skill-mix: A flexible and expandable family of evaluations for ai models Year: (2023)
Ref_id:b53 Title: Distilling system 2 into system 1 Year: (2024)
Ref_id:b54 Title: Root mean square layer normalization Year: (2019)
Ref_id:b55 Title: Joint learning of fuzzy k-means and nonnegative spectral clustering with side information Year: (2018)
Ref_id:b56 Title: Do transformers parse while predicting the masked word? Year: (2023)
Ref_id:b57 Title: The clock and the pizza: Two stories in mechanistic explanation of neural networks Year: (2023)
Ref_id:b58 Title: What algorithms can Transformers learn? A study in length generalization Year: (2023)
Ref_id:b59 Title: Promptintern: Saving inference costs by internalizing recurrent prompt during large language model fine-tuning Year: (2024)
Ref_id:b60 Title: for which the functions obtained after the first T SGD updates, f θ1 , . . . , f θ T , remain statistically independent (correlation 0) from MLT Π * despite training on it Year: ()
Ref_id:b61 Title: LT B is the product of input length, total training steps, and batch size Year: ()
Ref_id:b62 Title: we mention some useful lemmas for charaterizing the bijective maps on {0, 1} 2 → {0, 1} 2 that we will regularly use for the proof of Theorem F.10. The first lemma will show that each bijective map can be represented by three operations on input characters, copy, xor, and not operations. The second lemma measures correlation on the outputs of two randomly sampled bijective maps. The proofs are given in Appendix F.6. We define the following necessary operations to describe the random bijective maps on {0, 1} 2 → {0, 1} 2 : 1. copy: Given a tuple (x 1 , x 2 ), and a position argument i ∈ {1, 2}, the operation returns value of x i as output Year: ()
Ref_id:b63 Title: Given a variable x i , this operation returns flipped value of x i . That is, if x i = 1, then it returns 0 and vice-versa Year: ()
Ref_id:b64 Title: Given a tuple (x 1 , x 2 ), this operation returns x 1 ⊕ x 2 Year: ()
Ref_id:b65 Title: Any bijective map π : {0, 1} 2 → {0, 1} 2 can be expressed using copy, not, and xor operations. Furthermore, from 24 possible maps for π, 1. There are 6 maps ∆ 1 , ∆ 2 , • • • , ∆ 6 for which characters in the output tuple can be defined by copy and Year: ()
Ref_id:b66 Title: Mirror maps: For each map π ∈ {∆ 1 , ∆ 2 , • • • , ∆ 6 }, there exist mirror maps π (1) , π (2) , π (3) whose output on each input tuple can be defined by selective not operations on either or both characters of the output tuple of π. We call {π Year: ()
Ref_id:b67 Title: Mirror map set definition is general and isn't restricted to maps π ∈ {∆ 1 , ∆ 2 , • • • , ∆ 6 }. Because mirror maps are defined in terms of not operation, a mirror map set Mirrorset(π) for π ∈ {∆ 1 , ∆ 2 , • • • , ∆ 6 } is also equivalent to Year: ()
Ref_id:b68 Title: } 2 → {0, 1} 2 can be grouped into 6 family of maps, each containing 4 maps and represented by a unique map from {∆ 1 , ∆ 2 , • • • , ∆ 6 }. Each family is defined by mirror map set of their corresponding representative map Year: ()
Ref_id:b69 Title: 22 (Correlation of bijective maps for n = 2) Year: ()
Ref_id:b70 Title: Fix an i, j ∈ {0, 1}. , i.e. Correlation(π α (•) i , π β (•) j , U({0, 1} 2 )) := Pr x∼U Year: ()
Ref_id:b71 Title: Correlation at both output character pairs: With probability 1 3 w.r.t. the random selection of the maps, both the characters in the outputs of π α , π β have perfect correlation, i.e. either one of the cases hold true Correlation(π α (•) 1 , π β (•) 1 , U({0, 1} 2 )) = 1 Year: ()
Ref_id:b72 Title: Suppose f, g : {0, 1} k → {0, 1} 2 denote 2 functions for some arbitrary k, satisfying conditions for uniformity in output distribution Year: ()
Ref_id:b73 Title: {0,1} k ) g(x) 1 ⊕ g(x) 2 = 1/2 for all j ∈ {1, 2}. Then, the following holds true: 1. Both output character pairs have correlations between f and g: If (either) Correlation(f (•) 1 , g(•) 1 , U({0, 1} k )) = 1 and Correlation(f (•) 2 , g(•) 2 , U({0, 1} k )) = 1 (or) Correlation(f (•) 1 , g(•) 2 , U({0, 1} k )) = 1 and Correlation(f (•) 2 , g(•) 1 , U({0, 1} k )) = 1, then for two randomly picked phrasebooks π α , π β , (a) Correlation of an output character pair of π α (f (•)) and π β (g(•)): Fix an i, j ∈ {1, 2} Year: ()
Ref_id:b74 Title: With probability 1/3 w.r.t. the random selections of π α , π β , one of the following two conditions hold true. (either) Correlation(π α (f (•)) 1 , π α (g(•)) 1 , U({0, 1} k )) = 1 and Correlation(π α (f (•)) 2 , π α (g(•)) 2 , U({0, 1} k )) = 1 (or) Correlation(π α (f (•)) 1 , π α (g(•)) 2 , U({0, 1} k )) = 1 and Correlation Year: ()
Ref_id:b75 Title: Only a pair of characters have correlations between f and g: If there exists only one pair i, j ∈ {0, 1} such that Correlation(f (•) i , g(•) j , U({0, 1} k ) = 1, and for all other i ′ , j ′ , Correlation(f (•) i ′ , g(•) j ′ , U({0, 1} k )) = 0, then (a) Fix any i ′ , j ′ ∈ {1, 2}. With probability 1 Year: ()
Ref_id:b76 Title: If the above condition holds true, for all other ī, j pairs Year: ()
Ref_id:b77 Title: No correlations between f and g: If for all pairs i, j ∈ {1, 2}, Correlation(f (•) i , g(•) j , U({0, 1} k )) = 0, then for all pairs i ′ , j ′ ∈ {1, 2}, Correlation(π α (f (•)) i ′ , π β (g(•)) j ′ , U({0, 1} k )) = 0 Year: ()
Ref_id:b78 Title: We will prove each case separately Year: ()
Ref_id:b79 Title: Both output character pairs have correlations between f and g: We will consider the case when Correlation(f (•) 1 , g(•) 1 , U({0, 1} k )) = 1 and Correlation(f (•) 2 , g(•) 2 , U({0, 1} k )) = 1, proof for the other case is similar. Then, there are 4 cases possible. Subcase 1: f (x) 1 = g(x) 1 , f (x) 2 = g(x) 2 for all x ∈ {0, 1} k Subcase 2: f (x) 1 = not(g(x) 1 ), f (x) 2 = g(x) 2 for all x ∈ {0, 1} k Subcase 3: f (x) 1 = g(x) 1 , f (x) 2 = not(g(x) 2 ) for all x ∈ {0 Year: ()
Ref_id:b80 Title: Fix an i, j ∈ {0, 1}. , i.e. Correlation(π α (•) i , π β (•) j , U({0, 1} 2 )) := Pr x∼U Year: ()
Ref_id:b81 Title: Correlation at both output character pairs: With probability 1 3 w.r.t. the random selection of the maps, both the characters in the outputs of π α , π β have perfect correlation, i.e. either one of the cases hold true Correlation(π α (•) 1 , π β (•) 1 , U({0, 1} 2 )) = 1. Correlation(π α (•) 2 , π β (•) 2 , U({0, 1} 2 )) = 1. Algorithm 7 TRANSLATE-MODULE: Self-attention and MLP layers for Translate Require Year: ()
