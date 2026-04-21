Title: Characterizing the Expressivity of Fixed-Precision Transformer Language Models
Abstract: Transformer-based language models (LMs) have achieved widespread empirical success, but their theoretical expressive power remains only partially understood. In this work, we analyze a restricted idealization of fixed-precision transformers with strict future masking, soft attention, and no positional encodings. We establish that this class of models is exactly as expressive as a specific fragment of linear temporal logic that contains only a single temporal operator: the past operator. We further connect this fragment to established classes in formal language theory, automata theory, and algebra, yielding a unified framework for understanding transformer expressivity under this idealization. Finally, we present empirical results that align closely with our theory: transformers trained on languages within their characterized expressive capacity generalize reliably across sequence lengths, while they consistently fail to generalize on languages beyond it. 1

Section: Introduction
Transformer-based language models (LMs) have demonstrated remarkable empirical success [46,36,12] on a wide variety of natural language tasks [47, 19, 41, inter alia]. This success has sparked growing interest in understanding the theoretical expressive power of transformers, i.e., what languages they can and cannot recognize, and, by extension, what tasks they can and cannot perform. A significant body of work approaches this question by relating transformers to well-established frameworks such as formal languages, logic, and circuit complexity [18,31,50,42]. To facilitate their theoretical analysis, theoreticians often propose idealizations of transformers. For instance, while practical implementations of transformers operate under fixed precision, e.g., single (32-bit) or half (16-bit) precision, many authors assume arbitrary [38,18,34] or length-dependent precision [32,7]. Although such idealizations capture key aspects of transformers, they tend to overestimate their expressive power [38].
A recent step toward a more faithful theoretical understanding of the expressive power of transformers comes from Yang et al. [50], who show that fixed-precision transformers with strict future masking and unique hard attention (UHA) are exactly as expressive as linear temporal logic LTL[P, F, S, U], which includes four temporal operators: P (past), F (future), S (since), and U (until). However, UHA still deviates from the soft attention used in practice. To address this gap, Yang and Chiang [49] analyze fixed-precision transformers with strict future masking and soft attention, an idealization that most closely reflects the models deployed in real-world applications. Yang and Chiang [49] show that such models are upper bounded by C-RASP, a counting-based programming language, though a precise characterization of these models' expressivity remains open.
In this paper, we close this gap by providing an exact characterization of the expressive power of fixed-precision transformers with soft attention, strict masking, and no positional encodings (NoPE). We show they are precisely characterized by LTL[P], a restricted fragment of LTL[P, F, S, U] that  uses only the past operator (P). We further demonstrate that LTL[P] is equivalent in expressivity to partially ordered deterministic finite automata (PODFAs), which are characterized by R-trivial monoids and recognize left-deterministic polynomials. These results offer a detailed and principled characterization of the expressive power of this idealization, delineating its strengths and limitations. Crucially, our findings imply that many simple languages, e.g., bounded Dyck languages, which have been shown to be recognizable under more permissive idealizations, are beyond the reach of the models we study. We also extend our theoretical results to transformer LMs, showing that their expressivity matches that of transformer recognizers. A visual overview of the theoretical landscape is provided in Fig. 1.
To arrive at a compelling theory, it is essential to show that it faithfully reflects the behavior of models trained under standard machine learning paradigms. To this end, we provide empirical evidence using the length generalization framework, a widely used method for gauging neural network expressivity [10,6,22]. We construct a suite of languages spanning a fine-grained hierarchy of formal language classes. Our results (Tab. 1) exhibit strong alignment between theory and practice: for all languages that transformers are predicted to recognize, the models generalize perfectly over lengths (100% accuracy); for languages beyond their theoretical capacity, they consistently make generalization errors, regardless of learning rates or random seeds.
this section cite: ['b45', 'b35', 'b11', 'b17', 'b30', 'b49', 'b41', 'b37', 'b17', 'b33', 'b31', 'b6', 'b37', 'b49', 'b48', 'b48', 'b9', 'b5', 'b21']

Section: Background
In this section, we present the necessary background knowledge that underpins our analysis.
this section cite: []

Section: Strings and Languages
An alphabet, denoted as Σ, is a finite, non-empty set of symbols. A string over Σ is a finite sequence of symbols drawn from Σ. The set of all strings over Σ is denoted by its Kleene star Σ * . A subset of Σ * is called a language. A regular expression is a declarative way to describe a language, defined recursively as follows:
• ∅ and each a ∈ Σ are regular expressions;
• If α and β are regular expressions, so are the union α + β, concatenation αβ, the Kleene star α * , and complement α c .
A language is regular if and only if it can be described by a regular expression [26]. A regular language is said to be star-free if it can be described by a regular expression without the Kleene star [28]. As an example, Σ * , the set of all strings over Σ, is star-free, as it can be described by ∅ c .
this section cite: ['b25', 'b27']

Section: LTL[P]
Linear temporal logic LTL[P, F, S, U] [25] is a modal logic, with modalities referring to time. The full definition is given in §A.1. In this paper, we define a fragment of LTL[P, F, S, U]-denoted as LTL[P]-that includes only one temporal operator P (past). Formulas in LTL[P] are composed of atomic formulas π a for every a ∈ Σ, Boolean connectives ∧, ¬, and a temporal operator P. The disjunction ∨ is definable in terms of ∧ and ¬ as:
ψ 1 ∨ ψ 2 def = ¬(¬ψ 1 ∧ ¬ψ 2 ).(1)
When defined over strings, the formulas are interpreted with respect to a string w = w 1 • • • w N at a position n ∈ {1, . . . , N }. The semantics are defined inductively. Given a string w, a position n, and a formula ψ, we define w, n |= ψ (w satisfies ψ at position n) as follows:
w, n |= π a if and only if the n th symbol of w is an a. w, n |= ψ 1 ∧ ψ 2 if and only if w, n |= ψ 1 and w, n |= ψ 2 . w, n |= ¬ψ if and only if w, n ̸ |= ψ. w, n |= Pψ if and only if there exists m such that m < n and w, m |= ψ.
Example:
• abb, 1 |= π a as the first symbol is a.
• abb, 3 |= Pπ a as an a occurs before position 3.
To say a string w of length N satisfies a formula ψ, written as w |= ψ, we evaluate from position N + 1 (the position after the final symbol):
w |= ψ if and only if w, N + 1 |= ψ.
The language defined by a formula ψ is the set L(ψ) of all strings that satisfy ψ.
Example: Pa defines all the strings that contain at least one occurrence of a, i.e.,
L(Pa) = Σ * aΣ * .(3)
this section cite: []

Section: PFO 2 [<]
First-order logic FO[<] can also be used to describe formal languages. In contrast to (linear) temporal logic, where formulas are interpreted at a single position, FO[<] formulas may contain multiple position variables, making it better-suited to simulating mechanisms in transformers that involve more than one position, such as attention; see § § B.2 and 3.2. In this paper, we introduce a fragment of FO[<] called PFO 2 [<]-the past fragment of first-order logic with two variables. The atomic formulas of PFO 2 [<] consist of unary predicates π a for each a ∈ Σ and a binary numerical predicate < that can be used to determine the sequential order between variables. Formulas in PFO 2 [<] can have at most two distinct variables, x and y. In addition to the usual Boolean connectives ∧ and ¬, there is a bounded existential quantifier ∃y < x, where y is the variable bounded by the quantifier, and x is the free variable. In case there is no free variable, the bounded existential quantifier behaves like an ordinary existential quantifier. The formulas in PFO 2 [<] are constructed inductively as follows:
• Every atomic formula is a formula;
• A Boolean combination of formulas is a formula if it does not contain more than two variables;
• If ϕ(x, y) is a formula with two variables x, y, ∃y < x : ϕ(x, y) and ∃x < y : ϕ(x, y) are formulas;
• If ϕ(x) is a formula with one variable x, ∃x < y : ϕ(x) is a formula;
• If ϕ(x) is a formula with one variable x, ∃x : ϕ(x) is a formula.
The bounded universal quantifier ∀y < x can be defined using ∃y < x and ¬. For instance:
∀y < x : ϕ(x, y) def = ¬∃y < x : ¬ϕ(x, y).(4)
A formula with no free variables is called a sentence. We write w |= ϕ if ϕ is satisfied for w. A sentence ϕ defines a language, the set of strings over Σ satisfying it, denoted as L(ϕ).
this section cite: []

Section: Example:
The sentence below defines all the strings that contain abc as a subsequence (symbols appearing in order, possibly non-contiguously):
∃x : (π c (x) ∧ ∃y < x : (π b (y) ∧ ∃x < y : π a (x))) .(5)
Note that the variable x is reused, as permitted in PFO 2 [<]. The formulas below, however, are not definable in PFO 2 [<]:
• x < y < z (more than two variables);
• ∀x : (∃y ≥ x : π a (y)) (disallowed use of the quantifier ∃y ≥ x).
PFO 2 [<] is specifically designed to match the expressive power of LTL[P], allowing us to leverage both formalisms in our analysis of transformer expressivity.
Theorem 2.1. PFO 2 [<] and LTL[P] have the same expressive power. Proof. See §A.2.3. ■ 3 Transformers In this section, we formally define the transformer idealization under consideration and establish two central expressivity results: (i) every transformer can be translated into an equivalent PFO 2 [<] formula, and (ii) every LTL[P] formula can be simulated by a transformer. Combined with Theorem 2.1, these results imply that transformers are expressively equivalent to both LTL[P] and PFO 2 [<].
this section cite: []

Section: The Transformer Architecture
The transformer considered in this section follows the architecture described by Yang et al. [50], with the exception that we use soft attention in place of unique hard attention (UHA). Before formally defining the architecture, we first state explicitly the assumptions that will govern our model as follows.
Assumption 3.1. We assume that the transformer satisfies the following conditions:
1. Fixed precision: There exists a finite set of values that the floating-point numbers can assume. We denote this set as F = {f 1 , f 2 , . . .}.
this section cite: ['b49']

Section: 2.
No positional encoding (NoPE): Positional encodings are omitted for now, but are addressed in §F.3.
this section cite: []

Section: Soft attention:
Attention weights are computed using the softmax function. We also treat average hard attention (AHA) in §F.1, showing that AHA is equally expressive as soft attention.
4. Strict future masking: As in Yang et al. [50], the attention is strictly future-masked. Strict masking is shown to be more expressive than non-strict masking in §F.2.
this section cite: ['b49']

Section: 5.
Recognition: Following standard practice in expressivity studies [42], the transformer is treated as a language recognizer. We will extend the analysis to transformer language models in §5.
Transformers take strings as input. Given a string w = w 1 • • • w N over an alphabet Σ, we append a special end-of-sentence symbol EOS / ∈ Σ to form a completed string
w def = w 1 • • • w N EOS over the extended alphabet Σ def = Σ ∪ {EOS}. An initial embedding layer E : Σ N +1 → F D×(N +1) maps each
input symbol including EOS to a D-dimensional representation. We define the embedding layer as the 0 th layer of the transformer:
H (0) def = E. (6
)
The embedding is followed by a stack of L hidden layers. Each hidden layer contains two sublayers: a self-attention mechanism A (ℓ) : F D×(N +1) → F D×(N +1) , and a feedforward network F (ℓ) : F D×(N +1) → F D×(N +1) . These components are composed sequentially. For each ℓ ∈ {0, 1, . . . , L -1}, we define:
H (ℓ+0.5) def = LN A (ℓ) H (ℓ) + H (ℓ) ,(7a)
H (ℓ+1) def = LN F (ℓ) H (ℓ+0.5) + H (ℓ+0.5) .(7b)
where LN denotes layer normalization [1]. Finally, a classification head C : F D → F maps the representation at the EOS position in the final layer to a scalar output:
o(w) def = C H (L) (w) :,N +1 .(8)
We use H (L) (w) d,n to denote the (d, n) th entry in the matrix. Similarly, H (L) (w) :,n refers to the column vector at position n, and H (L) (w) d,: refers to the row vector corresponding to dimension d. Consequently, a transformer defines a function of type Σ N +1 → F, where N is a parameter of the type. We say a string w is accepted by the transformer if o(w) > 0. A detailed specification of the architecture is provided in §B.1.
this section cite: ['b41', 'b0']

Section: From Transformers to PFO 2 [<]
In this subsection, we discuss the following theorem.
this section cite: []

Section: Theorem 3.2. Every transformer can be simulated by PFO 2 [<].
Proof. See §B.2.
this section cite: []

Section: ■
Our proof closely follows the approach of Chiang et al. [9,Section 5], with one key difference: the simulation of summation in soft attention. While prior work makes use of counting quantifiers to simulate summation, Li et al. [27] demonstrate that summation with iterative rounding can be simulated with FO[<]. We take this a step further by showing that the specific summation involved in soft attention can be simulated by PFO 2 [<]. Here, we provide a high-level overview of the proof. We identify two summations in soft attention. The first occurs in the denominator of the softmax function, defined for a D-dimensional vector x, as follows:
softmax(x) d def = exp(x d ) D i=1 exp(x i )
, for d ∈ {1, . . . , D}.
exp is a deterministic function. Under the fixed precision assumption, the possible input and output values of exp form a finite set, allowing PFO 2 [<] formulas to encode exp by explicit enumeration of these values. Since exp outputs only non-negative values, the summation in the denominator reduces to a sum of non-negative terms, which can be simulated by PFO 2 [<] (Lemma B.5). Additionally, the second summation is a weighted sum, in which the weights are produced by the softmax function.
Under fixed precision, the output of a softmax contains a bounded number of non-zero entries (Proposition B.6). Therefore, the weighted sum involves only a bounded number of terms and can similarly be simulated by PFO 2 [<] (Lemma B.7).
this section cite: ['b8', 'b26']

Section: From LTL[P] to Transformers
In this subsection, we discuss the following theorem.
Theorem 3.3. Every LTL[P] formula can be simulated by a transformer. Proof. See §B.3. ■ Our proof is adapted from that of Yang et al. [50, Appendix C.1]. Intuitively, each LTL[P] formula is encoded in a dedicated coordinate d ∈ {1, . . . , D} of the transformer's hidden state. At each position, this coordinate takes the value 1 if the formula is satisfied at that position, and 0 otherwise. The key challenge in proving Theorem 3.3 lies in simulating the temporal operator P. Prior constructions typically employ uniform attention, which assigns equal weight to all preceding positions, to simulate such operators. However, under fixed precision, soft attention has a limited attention span, as the output of the softmax function contains only a bounded number of non-zero entries. We refer to this bound as the maximum attention span, denoted by N max . To overcome this limitation, previous work has relied on non-fixed numerical precision-such as arbitrary precision [9,49] or log precision [31]-to prevent attention weights from vanishing. In contrast, we present a construction that overcomes this issue without requiring increased numerical precision.
We begin by describing a base construction that applies when attention weights do not vanish, and then extend it to handle cases where vanishing weights may occur.
Base Construction. Suppose a formula ψ is simulated by coordinate d 1 at the ℓ th layer, i.e., by H (ℓ) d1,:
(omitting the input string w for brevity), and let Pψ be the formula we aim to simulate. We construct an attention sublayer that uniformly attends to all previous positions m < n such that H (ℓ) d1,m = 1. If such positions exist, we set an unused coordinate d P to 1, i.e., H (ℓ+0.5) d P ,n = 1 and otherwise to 0.
Handling Vanishing Weights. However, when too many (more than N max ) previous positions satisfy ψ, the resulting attention weights will underflow to 0. This results in H (ℓ+0.5) d P ,n = 0 even when Pψ is true, leading to incorrect behavior. To address this, we first compute the logical conjunction of d 1 and d P and store it in d ∧ , i.e.,
H (ℓ+1) d∧,n = 1 if H (ℓ) d1,n = 1 and H (ℓ+0.5) d P ,n = 1, 0 otherwise. (10
)
The number of positions n with H
d∧,n = 1 is bounded by N max , since beyond the (N max + 1) th position, the first attention sublayer will suffer from vanishing attention weights, causing H (ℓ+0.5) d P ,n = 0.
Next, we construct a second layer in which the attention sublayer uniformly attends to all positions m < n where H (ℓ+1) d∧,m = 1. This produces a coordinate d PP that simulates P(ψ ∧ Pψ). Since the number of positions satisfying H (ℓ+1) d∧,m = 1 is bounded, this second attention sublayer is not subject to vanishing attention weights and therefore computes its output reliably.
The formula P(ψ ∧ Pψ) effectively asks whether there are at least two positions before n that satisfy ψ. Thus, if there is exactly one position m < n satisfying ψ, then P(ψ ∧ Pψ) differs from the target formula Pψ. In this case, however, the first attention sublayer (producing d P ) correctly simulates Pψ, since the attention only needs to cover one position. We therefore take the logical disjunction of d P and d PP , implemented via the subsequent feedforward sublayer. The resulting coordinate d ∨ correctly simulates the formula Pψ. This completes the proof sketch.
this section cite: ['b8', 'b48', 'b30']

Section: Example:
Assume the attention span is limited to 1 position. The following example illustrates the simulation process, with defective simulations highlighted in red.
formula coordinate position ψ d 1 1 0 1 0 1 Pψ d P 0 1 1 0 0 ψ ∧ Pψ d ∧ 0 0 1 0 0 P(ψ ∧ Pψ) d PP 0 0 0 1 1 Pψ = P(ψ ∧ Pψ) ∨ Pψ d ∨ 0 1 1 1 14
this section cite: []

Section: Characterizations of LTL[P]
We have established the expressive equivalence between transformers and LTL[P]. This connection becomes especially compelling when paired with precise and rich characterizations of LTL[P]. To that end, we prove the following theorem. Theorem 4.1. Let L ⊆ Σ * be a regular language, M be its syntactic monoid, and A be the DFA accepting it. The following assertions are equivalent: (1) L is a left-deterministic polynomial, (2) M is R-trivial, (3) A is partially ordered, and (4) L is definable by LTL[P].
Proof. See §C.
this section cite: []

Section: ■
In the remainder of this section, we focus on the PODFA characterization, which will be central to later developments. The other characterizations are defined in §C, which also includes a discussion of superclasses of LTL[P], summarized in Tab. 2. Definition 4.2. A deterministic finite automaton (DFA) is a 5-tuple A = (Σ, Q, q I , F, δ) where
• Σ is an alphabet;
• Q is a finite set of states;
• q I ∈ Q is the initial state;
• F ⊆ Q is the set of final states;
• δ : Q × Σ → Q is a total transition function;
• q R ∈ Q \ F is a rejecting sink state, i.e., δ(q R , a) = q R for every a ∈ Σ.
Given an automaton A = (Σ, Q, q I , F, δ), we say A is in state q upon reading a string w = w 1 • • • w N ∈ Σ * if and only if there exists a sequence of states q 0 , . . . , q N such that
• q 0 = q I ; • q n+1 = δ(q n , w n+1 ) for n ∈ [N -1]; • q N = q.
A string w is accepted by A if it reaches one of the final states upon reading w. Definition 4.3. A DFA is said to be partially ordered if there exists a partial order ⪯ on Q such that for every state q ∈ Q and symbol a ∈ Σ, we have q ⪯ δ(q, a).
Informally, in a partially ordered DFA, once the automaton leaves a state, it never revisits it. An example of a non-partially ordered DFA is provided in §C.3.
this section cite: []

Section: Transformer Language Models
In this section, we extend our analysis to transformer LMs. Formally, an LM constitutes a distribution over Σ * , typically factorized autoregressively as follows:
p(w) def = - → p (EOS | w) N n=1 - → p (w n | w <n ).(11)
where w <n def = w 1 • • • w n-1 . We define an LM p's prefix probability p prefix as:
p prefix (w) def = w ′ ∈Σ * p(ww ′ ).(12)
To adapt a transformer for language modeling, we feed the input string w into the model autoregressively, and the classification head is replaced with a language modeling head L : F D → F |Σ| , which maps the representation in the last layer L at the most recent position n -1 to a probability distribution over Σ. This defines the local distribution -→ p as follows:
- → p (w n | w <n ) def = L H (L) (w <n ) :,n-1 wn . (13
)
where L(H (L) (w <n ) :,n-1 ) wn denotes the probability assigned to the symbol w n . Under fixed precision, this distribution may not be perfectly normalized, as the sum of its components can deviate from 1 due to rounding errors.
As transformer LMs do not introduce any new computational components beyond those already present in transformer recognizers, the same constructions and proofs apply. Therefore, every transformer LM can be simulated by PFO 2 [<].
Theorem 5.1. Every transformer LM can be simulated by PFO 2 [<].
Proof. See §D.
this section cite: []

Section: ■
The converse direction is more subtle. Yang and Chiang [49] show how to compile a transformer LM from C-RASP. Here, we present an alternative approach that is more intuitive. Our construction leverages the PODFA characterization of LTL [P]. Given a PODFA A = (Σ, Q, q I , F, δ), we can define a corresponding LM p A by specifying the local distribution -→ p A as follows: Suppose A is in state q upon reading the prefix w <n :
• If q / ∈ F and q ̸ = q R , -→ p A (a | w <n ) assigns uniform probability over all symbols in {a | δ(q, a) ∈ Q \ {q R }};
• If q ∈ F , - → p A (a | w <n ) assigns uniform probability over all symbols in {a | δ(q, a) ∈ Q \ {q R }} ∪ {EOS}; • If q = q R , - → p A (a | w <n ) = 0 for all a ∈ Σ, i.e., p prefix (w <n ) = 0.
Note that in practice, a softmax cannot produce an all-zero vector. For technical convenience, we therefore extend the Σ with a special symbol UNK / ∈ Σ, which receives all the probability mass when A reaches the rejecting state, i.e., -→ p A (UNK | w <n ) = 1 if q = q R . Accordingly, we extend the LM head L to accommodate this special symbol: L : F D → F |Σ∪{UNK}| . Theorem 5.2. Every PODFA LM can be simulated by a transformer LM.
Proof. See §D. ■ Intuitively, for each state q ∈ Q, we can construct a LTL[P] formula ψ q such that w <n , n -1 |= ψ q if and only if the automaton is in state q after reading the prefix w <n . By Theorem 3.3, each formula ψ q can be encoded into a designated coordinate d q of the transformer's hidden state. The language modeling head, extended to include the special token UNK, then maps these coordinates to the corresponding probability distribution specified by -→ p A .
this section cite: ['b48']

Section: Experiments
The experiments are divided into two parts: language recognition and language modeling.
this section cite: []

Section: Language recognition
We have shown that a transformer can recognize only left-deterministic polynomials. In this section, we conduct a series of language classification experiments to empirically validate this claim. We consider five language classes, arranged in a strict inclusion hierarchy-each class is a proper subset of the one preceding it. For each class, we select one or more representative languages. These languages are listed in Tab. 1, and their detailed definitions are provided in §E.1.2.
this section cite: []

Section: Experimental setup
We implement the transformer that we theorize about (Assumption 3.1). For comparison, we also train a long short-term memory (LSTM) [20]. Models are trained on strings up to length 40, and tested on strings of length 41 --500. Each experiment is run with 5 different random seeds and 3 learning rates. We consider a transformer to have successfully recognized a language if it achieves 100% accuracy in at least one of the runs. Details of the experimental setup and model configurations are provided in §E.1.1.
this section cite: ['b19']

Section: Results
We compute classification accuracy and report both the maximum and mean values across all runs in Tab. 1. The LSTM achieves perfect accuracy on all tasks, consistent with previous work showing that LSTMs can recognize regular languages [29] and implement counting mechanisms [48]. This confirms that the tasks are learnable given the available training data. Results on transformers align precisely with our theoretical predictions: under fixed precision, transformers with soft attention and NoPE can recognize exactly the class of left-deterministic polynomials. They achieve perfect accuracy on all LTL[P]-definable languages but consistently fail on tasks outside this class, even though prior work has shown that some of these languages are theoretically recognizable under more permissive idealizations [9,31,49,51,44]. Notably, we use single-precision (32-bit) floating-point numbers, and the string lengths never exceed the maximum attention span of the transformer. That is, attention can uniformly cover all prior positions without numerical underflow or overflow. Yet, despite these favorable conditions, the transformer exhibits no expressive power beyond what is predicted by our formal characterization. A more detailed breakdown of the results can be found in §E.1.3.
this section cite: ['b28', 'b47', 'b8', 'b30', 'b48', 'b50', 'b43']

Section: Language modeling
We now turn to experiments on language modeling, focusing on three representative languages: RDP-1, LDP-2, and LDP-1. The corresponding automata are illustrated in Fig. 3. We train the transformer LMs using the standard cross-entropy loss. For evaluation, a predicted symbol is considered correct if it has non-zero probability under the target distribution -→ p A induced by the DFA. Per-token accuracies are reported in Tab. 3. The transformer LM successfully learns LDP-1 and LDP-2 with perfect accuracy. For RDP-1, the best performance reaches 98.3%, but the model consistently falls short of achieving 100.0%. This gap becomes more evident upon inspecting the hidden states of the model.
Table 1: Language recognition experiments. Language classes are ordered by decreasing complexity. For each class, examples are chosen to be minimally more complex than those in the immediately lower class. Maximum and mean accuracies (± standard deviation) are reported. Exact values of 100.0% accuracy are highlighted in bold. Class Language Transformer LSTM max (%) mean (%) max (%) mean (%) Counter languages CNT 83.3 53.6 ± 8.6 100.0 86.9 ± 20.9 We extract the representation H at each position n by concatenating the outputs of all self-attention and feedforward sublayers:
H :,n = H (0.5) (w) :,n ⊤ • • • H (L) (w) :,n ⊤ ⊤ .(14)
To assess whether the model internally tracks the states in the DFAs, we train a linear classifier to predict the state the DFA is in upon reading w <n , given H :,n-1 . The classifier consists of two linear layers with no intermediate nonlinearity: the first projects to F 2 for visualization, and the second performs classification. In Fig. 2, we plot the 2D projections and overlay the decision boundaries of the classifier.
The visualizations reveal that, for LDP-1 and LDP-2, the transformer does distinguish the states, as their representations are linearly separable. In contrast, for RDP-1, state representations are intermixed. Notably, even when we train a neural probe with nonlinearity on the representations, perfect probing accuracy remains elusive. This suggests that the transformer does not learn to fully separate the states in the non-partially ordered DFA corresponding to RDP-1.
this section cite: []

Section: A Background
In this section, we provide background information that was omitted from the main text for brevity. We include it here for the sake of completeness.
this section cite: []

Section: A.1 Temporal logic
Temporal logic [37] is a special case of modal logic, with modalities referring to time.
this section cite: ['b36']

Section: A.1.1 LTL[P, F, S, U]
Linear temporal logic LTL[P, F, S, U] includes four temporal operators: P (past), F (future), S (since), and U (until). Notably, P, F, and one of S or U can be omitted without loss of expressive power [16]. We write ⊤ to denote TRUE and ⊥ to denote FALSE. The semantics of the formulas are defined inductively as follows:
• w, n |= π a iff the n th symbol of w is an a.
• w, n |= ψ 1 ∧ ψ 2 iff w, n |= ψ 1 and w, n |= ψ 2 .
• w, n |= ¬ψ iff w, n ̸ |= ψ.
• w, n |= Pψ iff there exists m such that m < n and w, m |= ψ.
• w, n |= Fψ iff there exists m such that m > n and w, m |= ψ.
• w, n |= ψ 1 Sψ 2 iff there exists m such that m < n, w, m |= ψ 2 , and for every k such that m < k < n, w, k |= ψ 1 .
• w, n |= ψ 1 Uψ 2 iff there exists m such that m > n, w, m |= ψ 2 , and for every k such that n < k < m, w, k |= ψ 1 .
Example:
• abb, 1 |= π a as the first symbol is a.
• abb, 3 |= Pπ a as an a occurs before position 3.
• abb, 3 |= π b Sπ a as π a holds at position 1 (before position 3), and π b holds at every position between 1 and 3.
To say a string w of length N satisfies a formula ψ, written as w |= ψ, we imagine we start at a position outside the string, e.g., position 0 or N + 1.
Example:
• abb |= π b Sπ a as the formula is satisfied at position 4, outside the string.
• abb |= π a Uπ b as the formula is satisfied at position 0, outside the string.
The language defined by a formula ψ is the set of all strings that satisfy ψ, denoted by L(ψ):
L(ψ) def = {w ∈ Σ * | w |= ψ}.(15)
Example:
L (π b Sπ a ) = Σ * ab * (16a) L (π a Uπ b ) = a * bΣ * (16b
)
We define the operator depth of a formula ψ, denoted od(ψ), as the maximum number of nested temporal operators in ψ. It is defined recursively as follows:
od(π a ) = 0 od(ψ 1 ∧ ψ 2 ) = max(od(ψ 1 ), od(ψ 2 )) od(¬ψ) = od(ψ) od(Pψ) = od(ψ) + 1 od(Fψ) = od(ψ) + 1 od(ψ 1 Sψ 2 ) = max(od(ψ 1 ), od(ψ 2 )) + 1 od(ψ 1 Uψ 2 ) = max(od(ψ 1 ), od(ψ 2 )) + 1 A.1.2 LTL[P, F]
Unary temporal logic LTL[P, F] is the fragment of LTL[P, F, S, U] that excludes the binary operators S and U. This restriction renders LTL[P, F] strictly less expressive than LTL[P, F, S, U] [13,14].
this section cite: ['b15', 'b12', 'b13']

Section: Example:
Consider the language consisting of all words that begin with a and end with b. To enforce that a word starts with a, we require that a occurs at the beginning of the word, i.e., it is preceded by nothing. This can be expressed as P(π a ∧ ¬P⊤). Symmetrically, ending with b can be expressed using the dual condition involving F. Thus, the formula
L(P(π a ∧ ¬P⊤) ∧ P(π b ∧ ¬F⊤)) = aΣ * b (17
)
defines the desired language.
this section cite: []

Section: A.1.3 LTL[P]
We define LTL[P] as the past fragment of LTL[P, F], which excludes the F operator.
Proposition A.1. LTL[P] is strictly less expressive than LTL[P, F]. Proof. Since LTL[P] is a syntactic fragment of LTL[P, F], every formula expressible in LTL[P] is also expressible in LTL[P, F]. To establish strictness, it suffices to exhibit a language definable in LTL[P, F] that is not definable in LTL[P]. Consider again the language aΣ * b. It is not definable in LTL[P], as expressing the end of a string requires the ability to look forward, which LTL[P] lacks. ■ A.2 First-order logic
The seminal work of Büchi [5] demonstrated how properties of languages can be described using logical formulas, while McNaughton and Papert [28] was the first to restrict these formulas to first-order logic (FO[<]).
this section cite: ['b4', 'b27']

Section: A.2.1 FO[<]
The atomic formulas of FO[<] consist of unary predicates π a for each a ∈ Σ and a binary numerical predicate < that can be used to determine the sequential order between variables. Formulas in FO[<] may use arbitrarily many variables. However, it has been shown that three variables are both necessary and sufficient to define all first-order definable languages [25,24]. In addition to the usual Boolean connectives ∧ and ¬, there is an existential quantifier ∃. The formulas in FO[<] are constructed inductively as follows:
• Every atomic formula is a formula;
• A Boolean combination of formulas is a formula;
• If ϕ(x, . . .) is a formula, so is ∃x : ϕ(x, . . .);
The universal quantifier ∀ can be defined as follows:
∀x : ϕ(x, . . .) def = ¬∃x : ¬ϕ(x, . . .).(18)
A formula with no free variables is called a sentence. We write w |= ϕ if ϕ is satisfied for w under the interpretation, where variables range over the positions in w, unary predicates π a (x) hold if the symbol at position x is a, and < is interpreted as the natural order on positions. A sentence ϕ can define a language, the set of strings over Σ satisfying it, denoted as L(ϕ).
this section cite: ['b24', 'b23']

Section: Example:
The formula below defines the language aΣ * b:
∃x∀y∃z : (π a (x) ∧ x ≤ y ∧ y ≤ z ∧ π b (z))(19)
where
x ≤ y is shorthand for x < y ∨ x = y.
The quantifier depth of a formula ϕ, denoted qd(ϕ), is defined recursively as follows:
qd(π a ) = 0 qd(ϕ 1 ∧ ϕ 2 ) = max(qd(ϕ 1 ), qd(ϕ 2 )) qd(¬ϕ) = qd(ϕ) qd(∃x : ϕ(x, . . .)) = qd(ϕ) + 1
It is well known that FO[<] and LTL[P, F, S, U] are equivalent in expressive power [25] and they both define exactly the class of star-free languages [28].
this section cite: ['b24', 'b27']

Section: A.2.2 FO 2 [<]
FO 2 [<] is the fragment of FO[<] restricted to using only two distinct variables (typically reused via quantification).
this section cite: []

Section: Example:
The formula Eq. ( 19) can be written as a formula in FO 2 [<] as follows:
∃x : (π a (x) ∧ ∀y : (x ≤ y ∧ ∃x : (y ≤ x ∧ π b (x)))) ,(20)
which uses only two distinct variables x and y.
It is shown that LTL[P, F] and FO 2 [<] recognize precisely the same languages [14].
this section cite: ['b13']

Section: A.2.3 PFO 2 [<]
We define PFO 2 [<] as the past fragment of FO 2 [<], in which formulas are restricted so that whenever a free variable x is present, every existential quantifier must be of the form ∃y < x.. This ensures that all quantification is limited to positions at or before x, i.e., only past positions relative to the free variable are accessed.
PFO 2 [<] has precisely the same expressive power as LTL[P]. We establish this equivalence by providing translations in both directions. Lemma A.2. Every formula ψ in LTL[P] can be translated into an equivalent single-variable formula ϕ(x) in PFO 2 [<].
Proof. We proceed by structural induction on the formula ψ.
Base case. If ψ = π a , we translate it to ϕ(x) = π a (x).
this section cite: []

Section: Induction step.
Assume ψ 1 and ψ 2 can be translated into ϕ 1 (x) and ϕ 2 (x) respectively. We translate compound formulas as follows:
• If ψ = ψ 1 ∧ ψ 2 , then ϕ(x) = ϕ 1 (x) ∧ ϕ 2 (x).
• If ψ = ¬ψ 1 , then ϕ(x) = ¬ϕ 1 (x).
• If ψ = Pψ 1 , then ϕ(x) = ∃y < x : ϕ 1 (y)
In each case, the resulting formula ϕ(x) belongs to PFO 2 [<], since it uses at most two variables (the free variable x and a bound variable y), both of which can be reused.
■ Lemma A.3. Every single-variable formula ϕ(x) in PFO 2 [<] can be translated into an equivalent formula ψ in LTL[P].
Proof. We proceed by induction on the quantifier depth qd(ϕ).
Base case. If qd(ϕ) = 0, then ϕ(x) is a Boolean combination of atomic formulas such as π a (x), which are directly translatable into LTL[P] formulas.
this section cite: []

Section: Induction step.
Assume that every formula of quantifier depth at most k can be translated into an equivalent LTL[P] formula. Consider a formula ϕ(x) of depth k + 1. The formula can be written as a Boolean combination of:
• Subformulas of quantifier depth at most k, which can be translated into LTL[P] by the inductive hypothesis;
• Subformulas of the form ∃y < x : ϕ 1 (y), where ϕ 1 has depth at most k. By the inductive hypothesis, ϕ 1 (y) translates to a LTL[P] formula ψ 1 , and the whole subformula translates to ψ = Pψ 1 .
this section cite: []

Section: ■
Combining the two lemmas, we obtain the following theorem.
Theorem 2.1. PFO 2 [<] and LTL[P] have the same expressive power. Proof. The result follows directly from Lemma A.2 and Lemma A.3. To define languages, LTL[P] formulas are interpreted at position N + 1. Similarly, PFO 2 [<] formulas with a single variable x can be converted into sentences by prefixing them with ∃x, which is semantically equivalent to ∃x < N + 1. ■
this section cite: []

Section: B Transformers
In this section, we describe the transformer idealization under consideration in more detail. The formal proofs for the following two results are also given: (i) every transformer can be translated into an equivalent PFO 2 [<] formula, and (ii) every LTL[P] formula can be simulated by a transformer.
this section cite: []

Section: B.1 Transformer architecture
The assumptions we make are restated as follows: Assumption 3.1. We assume that the transformer satisfies the following conditions:
1. Fixed precision: There exists a finite set of values that the floating-point numbers can assume. We denote this set as F = {f 1 , f 2 , . . .}.
this section cite: []

Section: 2.
No positional encoding (NoPE): Positional encodings are omitted for now, but are addressed in §F.3.
3. Soft attention: Attention weights are computed using the softmax function. We also treat average hard attention (AHA) in §F.1, showing that AHA is equally expressive as soft attention.
4. Strict future masking: As in Yang et al. [50], the attention is strictly future-masked. Strict masking is shown to be more expressive than non-strict masking in §F.2.
5. Recognition: Following standard practice in expressivity studies [42], the transformer is treated as a language recognizer. We will extend the analysis to transformer language models in §5.
We denote the set of non-negative floating-point numbers as F ≥0 ⊂ F. Similar notation is used for positive numbers, negative numbers, and other subsets. The set F includes two special values, ∞ and -∞, representing positive and negative infinity. Their behaviors are defined as follows:
• ∀f ∈ F \ {-∞}, ∞ + f def = ∞;
• ∀f ∈ F \ {-∞}, -∞ + f def = -∞; • ∀f ∈ F >0 , f • ∞ def = ∞ and f • (-∞) def = -∞; • ∀f ∈ F <0 , f • ∞ def = -∞ and f • (-∞) def = ∞; • ∀f ∈ F \ {∞, -∞}, f /∞ def = 0; • exp(∞) def = ∞ and exp(-∞) def = 0.
Transformers take strings as inputs. Given a string w = w 1 • • • w N over an alphabet Σ, we append a special end-of-sentence symbol EOS / ∈ Σ to form the extended string
w def = w 1 • • • w N EOS over the extended alphabet Σ def = Σ ∪ {EOS}.
The transformer consists of an input layer, followed by a stack of L hidden layers and a final output layer.
this section cite: ['b49', 'b41']

Section: B.1.1 Input layer
An embedding function e of type Σ → F D maps each symbol to a D-dimensional column vector. The embedding layer applies e to each symbol in w to produce the input representation: E(w) ∈ F D×(N +1) , i.e., E(w) :,n def = e(w n ), n ∈ {1, . . . , N + 1}.
where w N +1 def = EOS.
this section cite: []

Section: B.1.2 Hidden layers
Each hidden layer contains two sublayers: a self-attention mechanism and a feedforward network.
The self-attention mechanism is a function of type F D×(N +1) → F D×(N +1) . Given input H(w) ∈ F D×(N +1) , we compute:
Q def = Θ Q H(w),(22a)
K def = Θ K H(w), (22b
) V def = Θ V H(w),(22c)
where Θ Q , Θ K , Θ V ∈ F D×D are learnable parameter matrices.
A pairwise compatibility score S ∈ F (N +1)×(N +1) is computed via scaled dot product:
S n,m def = Q :,n • K :,m √ D .(23)
We then compute attention weights α ∈ F (N +1)×(N +1) using softmax:
α n,m def = exp(S n,m ) i<n exp(S n,i ) .(24)
In practice, the softmax is numerically stabilized by subtracting the maximum score from all scores before exponentiation:
α n,m def = exp(S n,m -max i<n S n,i ) j<n exp(S n,j -max i<n S n,i ) . (25
)
The attention output at position n is then computed as a weighted sum of the value vectors:
A(H(w)) :,n def = m<n α n,m V :,m .(26)
We assume a single attention head, since multiple heads do not increase expressive power [27].
The feedforward sublayer is a function of type F D×(N +1) → F D×(N +1) and consists of two linear transformations with a ReLU nonlinearity:
F(H(w)) :,n def = Θ F2 ReLU Θ F1 H(w) :,n + b F1 + b F2 ,(27)
where
Θ F1 ∈ F D ′ ×D , b F1 ∈ F D ′ , Θ F2 ∈ F D ′ ×D , b F2 ∈ F D are trainable parameters, D ′
is the hidden size of the intermediate layer.
this section cite: ['b26']

Section: B.1.3 Output layer
The final output layer is a function of type F D×(N +1) → F that computes a scalar score based on the representation at the final position (corresponding to EOS):
C H (L) (w) = θ C ⊤ H (L) (w) :,N +1 + b C (28
)
where θ C ∈ F D and b C ∈ F are learnable parameters.
this section cite: []

Section: B.2 From transformers to PFO 2 [<]
In this sub-section, we show that any transformer can be simulated by PFO
2 [<]. We begin by formalizing what it means for a function to be simulated by PFO 2 [<]. With a slight abuse of notation, we write D to refer either to the model's hidden dimensionality or to 1, in the case of a scalar classification output. Definition B.1. A function H : Σ N +1 → F D×(N +1) is said to be simulated by PFO 2 [<] if for every dimension d ∈ {1, . . . , D} and every floating-point value f ∈ F, there exists a single-variable formula ϕ(x) in PFO 2 [<] such that for every position n ∈ {1, . . . , N + 1}, H(w) d,n = f if and only if w, n |= ϕ(x).
Similarly, a function H of type Σ N +1 → F (N +1)×(N +1) is said to be simulated by PFO 2 [<] if for every floating-point value f ∈ F, there exists a two-variable formula ϕ(x, y) in PFO 2 [<] such that for every pair of positions n, m ∈ {1, . . . , N + 1}, H(w) n,m = f if and only if w, n, m |= ϕ(x, y).
We will prove the following proposition, which we will use repeatedly.
Proposition B.2. Let H 1 , H 2 be functions of type Σ N +1 → F D , and let H be either a function
F D → F D ′ (where D ′ is not necessarily equal to D) or F D × F D → F.
If both H 1 and H 2 can be simulated by PFO 2 [<], then so can H(H 1 ) and H(H 1 , H 2 ).
Proof. Since the inputs are fixed-dimensional vectors over F, the number of possible value combinations is finite. For any given output value f ∈ F, one can enumerate all input configurations for which the function yields f , and construct a PFO 2 [<] formula expressing their disjunction. ■ This proposition accounts for all components of the transformer architecture, such as the feedforward sublayers, projection operations, dot products, elementwise operations such as addition, scalar operations such as division, the exp function, and the classification head-except for the embedding layer, the summations involved in the attention computation, and the operation of identifying the maximum score introduced by the stabilized softmax.
We handle the embedding layer as follows:
Lemma B.3. The embedding layer can be simulated by PFO 2 [<].
Proof. For any dimension d and position n, the embedding layer outputs a fixed value f if the symbol at position n is mapped to f by the embedding function e in the d-th coordinate. Since Σ is finite and e is fixed, we can express this using a disjunction over all symbols that are mapped to f in coordinate d:
E(w) d,n = f if and only if w, n |= a∈{a|e(a) d =f } π a (x).(31)
this section cite: []

Section: ■
We next address the summations in the attention mechanism. We begin by establishing the following result: Proposition B.4. PFO 2 [<] can count up to a threshold.
Proof. The threshold counting quantifiers can be defined as follows.
• The quantifier for "there exists at least one" coincides with the standard bounded existential quantifier:
∃ ≥1 y < x : ϕ(y) def = ∃y < x : ϕ(y).(32)
• "There exist at least two" can be defined as:
∃ ≥2 y < x : ϕ(y) def = ∃y < x : (ϕ(y) ∧ ∃x < y : ϕ(x)).(33)
• We can define "exactly one" by combining the above:
∃ =1 y < x : ϕ(y) def = ∃ ≥1 y < x : ϕ(y) ∧ ¬∃ ≥2 y < x : ϕ(y).(34)
• Additional counting quantifiers up to a threshold can be defined in a similar fashion.
this section cite: []

Section: ■
Soft attention involves two types of summations, which we address in turn. The first appears in the denominator of the softmax function, used to compute the attention weights α (Eq. ( 24)). Since exp outputs non-negative values, this summation consists entirely of non-negative numbers and can be simulated by PFO 2 [<].
Lemma B.5. PFO 2 [<] can simulate a sum of non-negative fixed-precision floating-point numbers.
Proof. Since F ≥0 is finite, there exists a finite set of possible sums. Moreover, because all values are non-negative, for each such sum there are only finitely many combinations of input values that yield it. Each combination can be characterized using the threshold-counting quantifiers introduced in Proposition B.4: Too many positive entries lead to overflow, while 0's do not affect the sum. Taking a finite disjunction over all such combinations defines a PFO 2 [<] formula that holds exactly when the prefix sum equals a given value. ■
The second summation occurs in the computation of attention outputs (Eq. ( 26)). The value matrix V ∈ F D×(N +1) may include both positive and negative values, so we cannot apply the previous approach directly. However, as pointed out by Merrill and Sabharwal [31]: Proposition B.6. Under fixed precision, there exists an upper limit on the number of non-zero entries in the output of a softmax function. We refer to the upper bound as the maximum attention span N max .
Proof. This upper bound is given by:
N max = min(1, max(F \ {∞})) min(F >0 ) ,(35)
where ⌊•⌋ is the floor function.
this section cite: ['b30']

Section: ■
Consequently, the number of nonzero attention weights is bounded. It follows that Eq. ( 26) computes the sum over a bounded number of terms, which can be simulated by PFO 2 [<]:
Lemma B.7. PFO 2 [<] can simulate a sum of a bounded number of fixed-precision floating-point numbers.
Proof. Since the number of summands is bounded, we can enumerate all possible combinations of inputs. This behavior can be simulated using the threshold counting quantifiers introduced in Proposition B.4. ■
Finally, we account for the numerical-stability adjustment in Eq. ( 25). The only additional operation introduced by this stabilization is the identification of the maximum score among a finite set of floating-point values, which can be expressed in PFO 2 [<] as follows:
Proposition B.8. PFO 2 [<] can identify the maximum score max i<n S n,i .
Proof. Let the ordered set of floating-point values be F = {-∞, f 1 , f 2 , . . . , f K , ∞}. For each f ∈ F, let ϕ f (x) be a PFO 2 [<]-formula that holds iff the value at position x equals f . Since the set of floating-point numbers is finite, we can enumerate all possible maximum values. For each candidate maximum value f , we can construct a PFO 2 [<] formula ϕ max f (x) that holds if and only if the scores among all positions y < x are less than or equal to f , and at least one value equals f . They can be defined inductively over the total order of F as follows:
ϕ max ∞ (x) def = ∃y < x : ϕ ∞ (y), (36a
)
ϕ max f K (x) def = (¬∃y < x : ϕ max ∞ (y)) ∧ ∃y < x : ϕ f K (y),(36b)
. . .
ϕ max f1 (x) def =   ¬∃y < x : f ∈{∞,f K ,...,f2} ϕ max f (y)   ∧ ∃y < x : ϕ f1 (y),(36c)
ϕ max -∞ (x) def =   ¬∃y < x : f ∈{∞,f K ,...,f1} ϕ max f (y)   ∧ ∃y < x : ϕ -∞ (y).(36d)
this section cite: []

Section: ■
We have now completed the proof of the following theorem:
Theorem 3.2. Every transformer can be simulated by PFO 2 [<]. Proof. By Lemma B.3, the embedding layer can be simulated by PFO 2 [<]. By Proposition B.2, simulation by PFO 2 [<] is closed under all subsequent operations in the transformer, except for summations. By Lemma B.5, Proposition B.6, and Lemma B.7, the summation operations involved in attention computation can also be simulated by PFO 2 [<]. Therefore, the entire transformer can be simulated by PFO 2 [<]. ■
this section cite: []

Section: B.3 From LTL[P] to transformers
In this subsection, we show that every LTL[P] formula can be simulated by a transformer. The central idea is to encode the truth value of the formula at each position as a dedicated coordinate within the transformer's hidden state: the value 1 represents ⊤, and 0 represents ⊥.
The formal definition is as follows.
Definition B.9. A LTL[P] formula ψ is said to be simulated by a function H of type Σ N +1 → F D×(N +1) if there exists a coordinate d ∈ {1, . . . , D} such that for every position n ∈ {1, . . . , N +1} and every input string w ∈ Σ * ,
H(w) d,n = 1, if w, n |= ψ, 0, otherwise.(37)
We introduce a one-hot encoding function , which maps an index to a column vector of zeros except for a single entry equal to 1 at that index. Let 0 denote the all-zero column vector.
We first show that the atomic formulas can be translated into an embedding layer.
Lemma B.10. The atomic formulas π a for a ∈ Σ can be simulated by an embedding layer.
Proof. We assign a distinct coordinate d to every distinct symbol a ∈ Σ with a map g : Σ → {1, . . . , Σ }. Since the biases in attention mechanisms are sometimes omitted, we include a dedicated bias coordinate d b that always holds a constant value 1. The embedding layer is then defined as follows:
H (0) (w) d,n =    1, if d = d b , 1, if d = g(w n ), 0, otherwise.(38)
Coordinates that remain inactive at this stage are initialized to 0; they will later be used to store logical formulas introduced in subsequent layers. ■ Next, we show that the Boolean connectives can be simulated.
this section cite: []

Section: Lemma B.11.
If ψ 1 and ψ 2 can be simulated by a transformer, then their Boolean combinations can also be simulated by a transformer.
Proof. Assume ψ 1 and ψ 2 are simulated by H
d1,: and H
d2,: respectively. For brevity, we write x = H (ℓ) (w) d1,: and y = H (ℓ) (w) d2,: , so x, y ∈ F 1×(N +1) are row vectors indexed by positions. All operations below are applied elementwise.
• ¬ψ 1 : Construct a layer that computes, in a fresh coordinate d ¬ ,
-x + 1.(39)
This can be implemented with an FFN as follows. Select a hidden unit d ′ in the intermediate hidden state {1, . . . , D ′ } and set the first linear transformation to:
Θ F1 d,: = -d 1 ⊤ if d = d ′ , 0 ⊤ otherwise, b F1 = d ′ . (40
)
The second linear transformation maps d ′ to the output coordinate d ¬ :
Θ F2 d,: = d ′ ⊤ if d = d ¬ , 0 ⊤ otherwise, b F2 = 0.(41)
The attention sublayer is bypassed by setting Θ Q , Θ K , Θ V to zero matrices.
• ψ = ψ 1 ∧ ψ 2 : Compute, in a fresh coordinate d ∧ , ReLU (x + y -1) . (42) Implement this with a single FFN as follows. Select a hidden unit d ′ and set the first linear transformation to:
Θ F1 d,: = d 1 ⊤ + d 2 ⊤ if d = d ′ , 0 ⊤ otherwise, b F1 = -d ′ .(43)
The second linear transformation connects d ′ to the output coordinate d ∧ :
Θ F2 d,: = d ′ ⊤ if d = d ∧ , 0 ⊤ otherwise, b F2 = 0.(44)
Again, the attention sublayer is bypassed.
this section cite: []

Section: ■
Finally, we address the temporal operator P. Lemma B.12. If ψ can be simulated by a transformer, then Pψ can also be simulated by a transformer.
Proof. We start by introducing the base construction and then discuss two issues arising from fixed-precision arithmetic, together with their corresponding fixes. Assume ψ is simulated by H (ℓ) d1,: .
this section cite: []

Section: Base construction.
We construct an attention mechanism such that each position n attends uniformly to all previous positions m < n where H (ℓ) (w) d1,m = 1. Specifically, we need the attention scores to be:
S n,m = 0 if H (ℓ) (w) d1,m = 1, -f large if H (ℓ) (w) d1,m = 0,(45)
where f large ∈ F is a large enough positive floating-point number such that exp(-f large ) = 0. This ensures that positions where ψ is ⊥ receive zero attention weight.
This behavior can be implemented by configuring the query and key vectors as follows:
∀n, Q d,n = f large if d = d 1 , 0 otherwise,(46)
and
K = H (ℓ) (w) -1.(47)
where the subtraction by 1 is applied elementwise.
The corresponding attention projection weights are:
Θ Q d,: = f large d b ⊤ if d = d 1 , 0 ⊤ otherwise,(48)
and
Θ K d,: = d ⊤ -d b ⊤ ,(49)
where d b denotes the bias coordinate.
Next, we copy the values from H (ℓ) d1,: into an unused coordinate d P via the value projection:
Θ V d,: = d 1 ⊤ if d = d P , 0 ⊤ otherwise. (50
)
Thus, V becomes:
V d,: = H (ℓ) (w) d1,: if d = d P , 0 ⊤ otherwise. (51
)
Let N ⊤ denote the number of positions m < n such that H (ℓ) (w) d1,m = 1. If N ⊤ > 0, then α n,m = 1/N ⊤ , giving H (ℓ+0.5) (w) d P ,n = 1. Otherwise, if N ⊤ = 0, the attention weights are distributed uniformly over previous positions, all of which satisfy H (ℓ) (w) d1,m = 0, yielding H (ℓ+0.5) (w) d P ,n = 0. Hence, H (ℓ+0.5) d P ,:
simulates Pψ in the ideal case.
this section cite: []

Section: Patch 1: Vanishing Attention Weights
When too many previous positions with H (ℓ) d1,m = 1, i.e., N ⊤ > N max , the resulting attention weights α n,: will underflow to 0 ⊤ . This causes H (ℓ+0.5) (w) d P ,n = 0 even when Pψ is true, leading to incorrect behavior.
To fix this, we first compute the logical conjunction of d 1 and d P and store the result in d ∧ via the feedforward sublayer. When H (ℓ+1) d∧,n = 1, it indicates that the position n satisfies ψ and there exists at least one prior position that also satisfies ψ. The number of such positions is bounded by N max , as starting from the (N max + 1) th position, the first attention sublayer will suffer from vanishing attention weights, causing H (ℓ+0.5) (w) d P ,n = 0.
Next, we construct a second layer in which the attention sublayer uniformly attends to positions m < n where H (ℓ+1) (w) d∧,m = 1. This constructs a coordinate d PP simulating P(ψ ∧ Pψ). Since the number of positions where H (ℓ+1) d∧,m = 1 is bounded, this second attention sublayer is not subject to vanishing attention weights and reliably computes its output.
Note that P(ψ ∧ Pψ) evaluates to ⊤ when at least two earlier positions satisfy ψ. If exactly one earlier position satisfies ψ, then Pψ holds but P(ψ ∧ Pψ) does not. In this case, however, the first attention sublayer (producing d P ) already correctly simulates Pψ, since the attention only needs to cover one position. Thus, we take the logical disjunction of d P and d PP , implemented via the feedforward sublayer. The resulting coordinate d ∨ correctly simulates the formula Pψ. Patch 2: Rounding Errors Fixed-precision arithmetic introduces rounding errors when computing the attention weights α n,m . The fraction 1/N ⊤ may not be exactly representable by a finite-precision floating-point number. Moreover, the iterative summation in Eq. ( 26) can further accumulate rounding errors. Consequently, the output H (ℓ+0.5) d P ,n may equal a value very close to 1, when it should be exactly 1. This could lead to incorrect results in subsequent computations.
We address this with a thresholding mechanism implemented via a feedforward sublayer. Define round(1) as the set of floating-point numbers that deviates from 1 by at most δ:
round(1) = {f ∈ F | |f -1| ≤ δ} . (52
)
In standard floating-point systems, such as IEEE 754 [23], δ is on the order of the machine precision, e.g., approximately 2 -24 for single precision and 2 -11 for half precision. Let f small be a small positive floating-point number such that 1/f small is also representable, f small • 1 fsmall = 1, and min(round(1))f small > 0. We rectify the value at d P using: Let f small be a small positive floating-point number such that 1/f small is also representable, f small • 1 fsmall = 1, and min(round(1)) -f small > 0. We rectify the value at d P using:
- 1 f small
ReLU H (ℓ+0.5) d P ,:
-f small -ReLU H (ℓ+0.5) d P ,:
.
This function outputs 1 if H (ℓ+0.5) d P ,n ∈ round(1), and 0 if H (ℓ+0.5)
d P ,n = 0.
To implement this with an FFN, select two hidden units d ′ 1 , d ′ 2 and define:
Θ F1 d,: = d P ⊤ if d = d ′ 1 or d = d ′ 2 , 0 ⊤ otherwise, b F1 = -f small d ′ 1 . (54
)
and
Θ F2 d,: = -1 fsmall ( d ′ 1 ⊤ + d ′ 2 ⊤ ) if d = d P , 0 ⊤ otherwise, b F2 = 0. (55
)
this section cite: ['b22']

Section: ■
So far, we have ignored layer normalization. We now show that its presence does not affect the representational power of transformers.
Proposition B.13. If a LTL[P] formula ψ can be simulated by a transformer without layer normalization, then it can also be simulated by a transformer with layer normalization.
Proof. Let x be a vector. Layer normalization is defined as follows:
LayerNorm(x) = x -µ √ σ 2 + ϵ • γ + β,(56)
where µ and σ are the mean and standard deviation of the elements in x, γ and β are parameters, and ϵ is a small constant for numerical stability.
We introduce, for every logical coordinate d, a mirror coordinate d and enforce
H (ℓ) (w) d,n = 1 -H (ℓ) (w) d,n(57)
for all sublayers ℓ and positions n. This guarantees that for every position n,
µ n = 1 2 and σ 2 n = 1 4 . (58
)
independently of the particular truth assignment.
With this structure, we can choose the parameters of layer normalization such that the operation becomes the identity function (i.e., has no effect):
γ = σ 2 n + ϵ = 1 4 + ϵ, β = µ n = 1 2 . (59
)
However, the attention sublayer may produce values close to 1 but not exactly equal to it. Consequently, layer normalization can slightly perturb these values away from the intended Boolean values. We denote by round(0) and round(1) the sets of floating-point numbers that deviate from 0 and 1, respectively, by at most a small constant δ, which is again on the order of the machine precision. This deviation is harmless, since the subsequent feedforward sublayer can rectify the outputs using the following thresholding function: let d 1 be the coordinate we hope to rectify and choose f 1 , f 2 ∈ F with max(round(0)) < f 1 < f 2 < min(round(1)). Apply:
1 f 2 -f 1 H (ℓ) (w) d1,: -f 1 -ReLU H (ℓ) (w) d1,: -f 2 . (60
)
To implement this with an FFN, select two hidden units d ′ 1 , d ′ 2 and define:
Θ F1 d,: = d 1 ⊤ if d = d ′ 1 or d = d ′ 2 , 0 ⊤ otherwise, b F1 = -f 1 d ′ 1 -f 2 d ′ 2 .(61)
and
Θ F2 d,: = 1 f2-f1 ( d ′ 1 ⊤ -d ′ 2 ⊤ ) if d = d 1 , 0 ⊤ otherwise, b F2 = 0.(62)
This yields 1 if H (ℓ) (w) d1,n ∈ round(1), and 0 if H (ℓ) (w) d1,n ∈ round(0). ■ Now, we obtain the following theorem: Theorem 3.3. Every LTL[P] formula can be simulated by a transformer.
Proof. By applying structural induction to the formula, using the lemmas Lemma B.10, Lemma B.11, and Lemma B.12, we demonstrate that each formula ψ can be simulated by dimension d ψ in the transformer's hidden state. Furthermore, by Proposition B.13, we can nullify the effect of layer normalization.
Finally, the classification head only needs to copy the relevant dimension d ψ as the output. ■
this section cite: []

Section: C Characterizations of LTL[P]
In this section, we introduce three alternative formalisms and show that they are equivalent in expressive power to LTL[P].
this section cite: []

Section: C.1 Left-deterministic polynomials
A monomial over an alphabet Σ is a language of the form:
Σ * 0 a 1 Σ * 1 • • • a n Σ * n ,(63)
where a 1 , . . . , a k ∈ Σ and Σ 0 , Σ 1 , . . . , Σ n ⊆ Σ.
A monomial is called
• left-deterministic if for every k ∈ {1, . . . , n}, a k ̸ ∈ Σ k-1 ,
• right-deterministic if for every k ∈ {1, . . . , n}, a k ̸ ∈ Σ k ,
• unambiguous if it is either left-deterministic or right-deterministic.
A polynomial is a finite union of monomials, and it is said to be left-deterministic (resp. rightdeterministic or unambiguous) if it is a finite disjoint union of left-deterministic (resp. rightdeterministic or unambiguous) monomials.
this section cite: []

Section: Example:
The language FIRST bΣ * is left-deterministic and therefore definable in LTL[P]. In contrast, the language LAST Σ * b is right-deterministic but not left-deterministic and thus not recognizable by LTL[P].
this section cite: []

Section: C.2 R-trivial monoids
A semigroup is a set endowed with an associative operator, and a monoid is a semigroup additionally endowed with an identity element. A canonical example is the set Σ * , with string concatenation as the associative operation and the empty string ε as the identity element. A monoid M is said to be R-trivial (resp. L-trivial) if for all elements s, t ∈ M, the condition sM = tM (respectively, Ms = Mt) implies s = t.
We now define an equivalence relation on Σ * known as the syntactic congruence. Given a language L ⊆ Σ * , two strings s, t ∈ Σ * are syntactically equivalent, written s≡ L t, if and only if:
∀u, v ∈ Σ * : usv ∈ L ⇔ utv ∈ L.(64)
The equivalence class of a string w is denoted as [w]. The quotient monoid Σ * /≡ L , i.e., the set of equivalence classes under ≡ L , is called the syntactic monoid of the language L.
this section cite: []

Section: C.3 Partially ordered DFAs
PODFA is introduced in §4. Constructing the DFA for a given language is a useful method for assessing whether the language is definable in LTL[P].
Example: It is non-trivial to determine whether certain languages can be defined in LTL[P]. However, it becomes clearer when we consider its corresponding automaton. For instance, consider the language DYCK-(1, 1), written as (ab) * in regular expression, the Dyck language over one type of parentheses with nesting depth limited to 1. The minimal DFA accepting this language is shown below:
q 0 q 1 a b
This DFA is not partially ordered, as it can revisit both q 1 and q 0 . Therefore, DYCK-(1, 1) is not definable in LTL[P].
this section cite: []

Section: C.4 Equivalence
We first show that every PODFA can be defined in LTL[P]. Lemma C.1. Let A = (Σ, Q, q I , F, δ) be a PODFA and w ∈ Σ a string of length N . For every q ∈ Q, there exists a formula ψ q> in LTL[P] such that A is in q upon reading w <n if and only if w, n -1 |= ψ q> .
Proof. We first construct a formula ψ =q for each state q ∈ Q, such that w, n -1 |= ψ =q if and only if A is in q after reading w <n-1 , i.e., right before reading w n-1 . We proceed by induction on the partial order ⪯ defined over Q.
Base case. The initial state q I can be defined by ensuring that no prior symbol caused a transition out of q I :
ψ =q I = a∈{a|δ(q I ,a)̸ =q I } ¬Pπ a .(65)
Induction step. Assume that for all q ′ ⪯ q with q ′ ̸ = q, the formulas ψ =q ′ have already been constructed. Then, A has entered q at some prior point if:
ψ <q = q ′ ∈Q a∈{a|δ(q ′ ,a)=q} P(ψ =q ′ ∧ π a ).(66)
To define ψ =q , we must ensure that q has been entered and it is not exited yet:
ψ =q = ψ <q ∧ a∈{a|δ(q,a)̸ =q} ¬P(ψ <q ∧ π a ).(67)
Once we obtain ψ =q , we can define ψ q> as follows:
ψ q> = q ′ ∈Q a∈{a|δ(q ′ ,a)=q} ψ =q ′ ∧ π a(68)
Finally, a string w is accepted by A if it reaches a final state:
q∈F ψ q .(69)
this section cite: []

Section: ■
Now we move on to prove the direction from LTL[P] to R-trivial monoid. Our proof is inspired by the proof of Proposition 4 in Diekert et al. [11].
A key characterization of R-trivial monoids is given by Brzozowski and Fich [4]: Proposition C.2 (Brzozowski and Fich [4]). M is R-trivial if and only if there exists an integer K > 0 such that for all s, t ∈ M, we have (st
) K s = (st) K .
Thus, we prove the following lemma as preparation: Lemma C.3. Let ψ be a formula in LTL[P] with operator depth at most K and K > 0. For every u, v, s, t ∈ Σ * , we have
u(st) K sv |= ψ if and only if u(st) K v |= ψ.(70)
Proof. The case s = ε is trivial, so we assume s ̸ = ε.
Let w = u(st) K sv and w ′ = u(st) K v. As w ′ is a subsequence of w, we align positions in w ′ with a subset of positions in w. Define N = |w|. We consider position N + 1 to point just beyond the end of w.
We define a pair (n, n ′ ) ∈ {1, . . . , N + 1} as legal if
(n = n ′ = N + 1) ∨ w n = w n ′ ∧ (n ′ < u(st) K ∨ n ′ > u(st) K s ) .(71)
Next, define the middle block of w as:
B k = n u(st) k < n < u(st) K s ,(72)
A legal pair (n, n ′ ) is said to be k-close if:
n = n ′ ∨ n, n ′ ∈ B k .(73)
We now prove the following inductive claim: Let ψ be a formula in LTL[P] with operator depth at most k, for some 0 ≤ k ≤ K. For every k-close pair (n, n ′ ), we have
w, n |= ψ if and only if w ′ , n ′ |= ψ.(74)
We proceed by induction on k.
Base case. k = 0: In this case, ψ is a Boolean combination of atomic formulas π a . (n, n ′ ) being a legal pair implies either n = n ′ = N + 1 or w n = w n ′ , so the claim holds trivially.
this section cite: ['b10', 'b3', 'b3']

Section: Induction step.
Assume the claim holds for depth k -1. Let ψ 1 = Pψ where ψ has depth k -1. Suppose w, n |= ψ 1 . Then there exists a position m < n such that w, m |= ψ. For w ′ , n ′ |= ψ 1 to hold, we need to identify a position m ′ < n ′ such that w ′ , m ′ |= ψ. By the inductive hypothesis, it suffices for m ′ < n ′ and for (m, m ′ ) to be (k -1)-close. There are two cases:
• Case 1: m / ∈ B k . Then we can set m ′ = m. (m, m ′ ) is (k -1)-close by definition. We now need to show that m ′ < n ′ .
-Subcase 1.1: n, n ′ ∈ B k . Since m / ∈ B k and m < n, we can conclude m ≤ u(st) k . Thus,
m ′ = m < n ′ because n ′ ∈ B k . -Subcase 1.2: n = n ′ . Then we have m ′ = m < n = n ′ . • Case 2: m ∈ B k . Then, as B k-1 \ B k covers all symbols in B k , there exists m ′ ∈ B k-1 \ B k such that w m ′ = w m . Again, (m, m ′ ) is (k -1)-close since m, m ′ ∈ B k-1 . We need to show that m ′ < n ′ : -Subcase 2.1: n, n ′ ∈ B k . Since m ′ ∈ B k-1 \ B k and n ′ ∈ B k , we have m ′ < n ′ . -Subcase 2.2: n = n ′ . Since m ′ ∈ B k-1 \ B k and m ∈ B k , we have m ′ < m < n = n ′ .
The converse direction, from w ′ , n ′ |= ψ to w, n |= ψ follows analogously. By applying the claim with k = K, n = N + 1, and n ′ = N + 1, we conclude that w |= ψ if and only if w ′ |= ψ.
this section cite: []

Section: ■
This result then follows straightforwardly: Lemma C.4. Every LTL[P]-definable language has a R-trivial syntactic monoid.
Proof. Lemma C.3 suffices to show that every LTL[P]-definable language satisfies the characterization in Proposition C.2. ■
We now have all the pieces in place to prove the main theorem. Theorem 4.1. Let L ⊆ Σ * be a regular language, M be its syntactic monoid, and A be the DFA accepting it. The following assertions are equivalent: (1) L is a left-deterministic polynomial, (2) M is R-trivial, (3) A is partially ordered, and (4) L is definable by LTL[P].
Proof. Brzozowski and Fich [4] showed the equivalence of ( 1), (2), and (3). To complete the picture, it remains to incorporate (4), which follows from Lemma C.1 and Lemma C.4.
this section cite: ['b3']

Section: ■
The relationships between various temporal logics, first-order logics, formal languages, monoids, and finite automata are summarized in Tab. 2. We treat the last row (for LTL[P]) in this paper. For further details on the other classes (LTL[P, F] and LTL[P, F, S, U]), interested readers can refer to McNaughton and Papert [28] and Tesson and Thérien [45].
this section cite: ['b27', 'b44']

Section: D Transformer Language Models
The transformer architecture follows the specification in §B.1, with one modification: the final output layer is replaced by a language modeling head L : F D → F |Σ∪{UNK}| . L is defined as follows:
L(H (L) (w <n )) = softmax Θ L H (L) (w) :,n-1 + b L (75
)
where Θ L ∈ F (|Σ|+1)×D and b L ∈ F |Σ|+1 are learnable parameters.
We have laid the groundwork that makes the proofs of the following two theorems straightforward.
Theorem 5.1. Every transformer LM can be simulated by PFO 2 [<].
Proof. The mapping L consists of a linear layer followed by a softmax. Since both the linear layer and softmax are definable in PFO 2 [<] as shown in §B.2, the result follows directly. ■ Theorem 5.2. Every PODFA LM can be simulated by a transformer LM.
Proof. Let A = (Σ, Q, q I , F, δ) be a PODFA, and let p A be its induced LM. By Lemma C.1, for each state q ∈ Q, there exists a LTL[P] formula ψ q that characterizes the automaton being in state q. Then, by Theorem 3.3, each such formula can be simulated by a dedicated dimension in the hidden state of a transformer.
Since A is deterministic, at each position, there is exactly one coordinate that takes the value 1, while the rest are 0. Therefore, the final prediction head effectively acts as a lookup table that maps each dimension to the target distribution -→ p A induced by A, which depends solely on the state A is in. ■
this section cite: []

Section: E Experiments
In this section, we present additional empirical results that align with our theoretical findings.
this section cite: []

Section: E.1 Language recognition
We supplement §6.1 with detailed descriptions of the experimental setup, tasks, and results.
this section cite: []

Section: E.1.1 Experimental setup
Our experimental setup follows Deletang et al. [10], Butoi et al. [6]. We use a transformer with soft attention, strict future masking, L = 5 layers, model size D = 64, and NoPE. Training strings are of length up to 40, while test strings range from length 41 to 500. The model is trained for 1,000,000 steps with a batch size of 128. For evaluation, we generate 512 samples per test length. For comparison, we also train a long short-term memory (LSTM) [20] with a hidden size of 256. Each experiment is run with 5 different random seeds and 3 learning rates (1 × 10 -4 , 3 × 10 -4 , 5 × 10 -4 ).
All experiments were conducted on a single GPU with 24 GB of memory, each taking approximately one hour to complete. Our code is adapted from https://github.com/google-deepmind/  neural_networks_chomsky_hierarchy, licensed under the Apache License, Version 2.0.
this section cite: ['b9', 'b5', 'b19']

Section: E.1.2 Languages
We consider five language classes, arranged in a strict inclusion hierarchy-each class is a proper subset of the one preceding it. For each class, we select one or more representative languages.
this section cite: []

Section: Counter languages.
Counter languages are languages that can be recognized by counter machinesfinite-state automata augmented by a number of counters [15,30]. In our experiments, we choose one of the simplest counter languages: CNT a n b n , the set of strings consisting of an arbitrary number of as followed by an equal number of bs.
this section cite: ['b14', 'b29']

Section: Regular languages.
A defining characteristic of non-star-free regular languages is modular counting. For instance, PARITY a * (ba * ba * ) * , the language of binary strings with an even number of bs, is one of the simplest instances of this class.
Star-free languages. Languages that are definable in LTL[P, F, S, U]. We focus on three common examples that are not definable in LTL[P, F]:
• DYCK-(1, 2): (a(ab) * b) * , the Dyck language (well-balanced parentheses) with 1 pair of parentheses, limited to depth 2.
• DYCK-(1, 1): (ab) * , the Dyck language with 1 pair of parentheses, limited to depth 1. This is also a canonical example of strictly locally testable languages, where string membership depends only on adjacent symbol blocks [28].
• LT-2: Σ * abΣ * with |Σ| > 2, the set of strings containing ab as a substring (symbols appearing contiguously). This is an example of a locally testable language [28].
Unambiguous polynomials. Languages that are definable in LTL[P, F]. We are interested in those not definable in LTL[P], i.e., right-deterministic but not left-deterministic polynomials. We select two such languages:
• RDP-1: (Σ \ {b 0 }) * a(Σ \ {a, b 1 }) *
, a simple right-deterministic monomial.
• LAST: Σ * b, the language of strings ending with b. It can be seen as the simplest representative of the class.
Left-deterministic polynomials. We now consider languages that are definable in LTL[P], selecting five examples that serve as contrasts to the previously discussed ones.:
• PT-2: Σ * aΣ * bΣ * with |Σ| > 2, the language of strings that contain ab as a subsequence. Languages of this type are known as piecewise testable languages [40].
• LT-1: Σ * aΣ * , the set of strings that contain a as a substring. This is the simplest case in both locally testable and piecewise testable languages.
• LDP-1: (Σ \ {a, b 0 }) * a(Σ \ {b 1 }) * , a left-deterministic monomial symmetrical to RDP-1.
• Although LDP-1 and RDP-1 are symmetrical, the former can be recognized by a DFA with only 2 states, whereas the latter requires 3 (see Fig. 3). For a fair comparison, we also include LDP-2
(Σ \ {a 1 , b 0 }) * a 1 (Σ \ {a 2 , b 1 }) * a 2 (Σ \ {b 2 }) *
, which also requires 3 states.
• FIRST: bΣ * , the set of strings beginning with b.
this section cite: ['b27', 'b27', 'b39']

Section: Sample generation.
For each language, we construct both positive and negative samples. Some examples are constructed adversarially to increase the difficulty of the classification task.:
• CNT: Negative samples contain one fewer a or b.
• PARITY: Negative samples contain an odd number of bs.
• DYCK-(1, 2), DYCK-(1, 1): One symbol in a positive sample is flipped. Since these languages require even-length strings, we only use even-length inputs.
• LT-2 and LT-1: Negative samples omit ab (resp. a) as a substring. Positive samples are constrained to include exactly one such occurrence.
• RDP-1, LDP-1, and LDP-2: Negative samples contain a single incorrect symbol.
• LAST and FIRST: Negative samples do not end or begin with b, respectively.
• PT-2: Negative samples lack the ab subsequence.
this section cite: []

Section: E.1.3 Results
We compute classification accuracy and report both the maximum and mean values across all runs in Tab. 1. The LSTM achieves perfect accuracy on all tasks, consistent with previous work showing that LSTMs can recognize regular languages [29] and implement counting mechanisms [48]. This confirms that the tasks are learnable given the available training data.
Counter languages. While several papers have shown that counting quantifiers can be simulated by arbitrary-precision [9,49] or log-precision transformers [31], our results indicate that a fixed-precision transformer cannot recognize CNT. The model achieves a maximum accuracy of only 83.3%. This finding contrasts with Bhattamishra et al. [3], who report that transformers can recognize a n b n c n . The discrepancy may stem from our evaluation on longer input lengths.
this section cite: ['b28', 'b47', 'b8', 'b48', 'b30', 'b2']

Section: Regular languages.
In line with Hahn [17], we find that the transformer completely fails to learn the non-star-free regular language PARITY, reaching at most 52.1% accuracy. Although Chiang and Cholak [8] design a transformer with customized positional encodings capable of recognizing PARITY, these encodings are not always representable under fixed precision, and the resulting architecture is not learnable under standard training conditions.
Star-free languages. Yang et al. [50] show that fixed-precision transformers with UHA recognize exactly the star-free languages. Similarly, Yao et al. [51] demonstrate that transformers with positional encodings of the form n/N -which are not always representable under fixed precision-can recognize bounded Dyck languages, a family of star-free languages. In contrast, we prove that transformers with soft attention and NoPE cannot recognize even the simplest bounded Dyck languages (DYCK-(1, 2) and DYCK-(1, 1)), nor the locally testable language LT-2. These results are corroborated by our experiments: the transformer fails to learn any of these three languages. The best performance is on DYCK-(1, 1), with a maximum accuracy of 87.7%, which still indicates poor generalization, especially considering the simplicity of the task.
Unambiguous polynomials. As predicted by our theory, the transformer fails to learn unambiguous polynomials that are not left-deterministic. The model achieves a maximum accuracy of 90.0% on RDP-1 and 64.8% on LAST.
Left-deterministic polynomials. Although the transformer cannot learn LT-2, it achieves perfect accuracy on both PT-2 and LT-1. In contrast to the poor performance on RDP-1 and LAST, the model learns their symmetrical counterparts (LDP-1, LDP-2, and FIRST) with 100% accuracy. Notably, Chiang and Cholak [8] construct unmasked transformer encoders capable of recognizing FIRST, but report difficulty in training such models in practice. Our results show that masked transformer decoders can learn FIRST easily and consistently, suggesting that masking may offer a more robust source of positional information than positional encodings.
this section cite: ['b16', 'b7', 'b49', 'b50', 'b7']

Section: Summary
The empirical results align fully with our theoretical predictions. Importantly, we use single-precision (32-bit) floating-point numbers, and the string lengths never exceed the maximum 100.0 98.9 ± 3.7 attention span of the transformer. That is, attention can uniformly cover all prior positions without numerical underflow or overflow. Yet, despite these favorable conditions, the transformer exhibits no expressive power beyond what is predicted by our formal characterization.
this section cite: []

Section: E.2 Language modeling
The DFAs corresponding to the languages used in our experiments are shown in Fig. 3, and their maximum and mean per-token accuracies are reported in Tab. 3. As predicted, the transformer language model learns left-deterministic polynomials perfectly but fails on the right-deterministic polynomial.
this section cite: []

Section: F Additional results
In this section, we present further theoretical results of potential interest.
this section cite: []

Section: F.1 Hard attention
Average hard attention (AHA) uniformly attends to positions m < n with the maximum score S n,m . Formally, Eq. ( 24) is modified as follows:
α n,m def = 1 {S n,m = max i<n S n,i } j<n 1 {S n,j = max i<n S n,i } ,(76)
where 1 {•} is the indicator function.
We now show that AHA is also definable in PFO 2 [<].
Theorem F.1. Every transformer with AHA can be simulated by PFO 2 [<]. Proof. By Proposition B.8, PFO 2 [<] can identify the maximum score max i<n S n,i . The denominator in Eq. (76) is a sum of non-negative terms and can therefore be simulated by PFO 2 [<] via Lemma B.5. As with the soft-attention case in Eq. (24), the resulting attention weights may vanish when the number of positions exceeds the model's bounded attention span; hence, Lemma B.7 applies here as well. The remaining steps of the construction follow identically to those in §B.2. ■ The other direction is straightforward. Theorem F.2. Every LTL[P] formula can be simulated by a transformer with AHA.
Proof. Recall that the attention mechanism we constructed in the proof of Lemma B.12 uniformly attends to all positions with the highest score S n,m = 0, effectively making it equivalent to AHA. Therefore, LTL[P] formulas can also be simulated by transformers with AHA. ■ Now, we turn to UHA, another widely studied attention mechanism where each position n attends to a single position m < n with the highest S n,m . In the case of ties, the rightmost position is selected. Yang et al. [50] prove that transformers with UHA are as expressive as LTL[P, F, S, U].
Combining the results from above, we have: Corollary F.3. Transformers with AHA are as expressive as those with soft attention, which are strictly less expressive than those with UHA.
this section cite: ['b49']

Section: F.3 Positional encodings
Positional encodings are introduced in Vaswani et al. [46] to inject information about the positions into the transformer. In general, there are two categories of positional encodings:
• Absolute positional encoding: This is a function p : {1, . . . , N + 1} → F D . It is typically injected into the input layer by modifying Eq. ( 21) as follows:
E(w) :,n def = e(w n ) + p(n), n ∈ {1, . . . , N + 1}.(82)
• Relative positional encoding: This is a function p : {1, . . . , N + 1} × {1, . . . , N + 1} → F D . It is typically injected into the attention sublayer by modifying Eq. ( 23) as follows [39,21]:
S n,m def = Q :,n • K :,m + Q :,n • p(n, m) √ D .(83)
In formal logic, a numerical predicate is a predicate that depends solely on the positions in a string, not the symbols in those positions. Numerical predicates that depend on one (resp. two) position(s) are referred to as unary (resp. binary) numerical predicates. For instance, the relation < is a binary numerical predicate.
We now show that all absolute (resp. relative) positional encodings can be simulated by unary (resp. binary) numerical predicates.
Theorem F.5. Let p be an absolute (resp. relative) positional encoding and L ⊆ Σ * be a regular language. There exists a collection of unary (resp. binary) numerical predicates P such that the following assertions are equivalent:
1. L can be recognized by a transformer with positional encoding p.
this section cite: ['b45', 'b38', 'b20']

Section: 2.
L is definable by PFO 2 [<, P], i.e., PFO 2 [<] extended with the numerical predicates in P.
Proof. Positional encodings, under fixed precision, have a finite image. Therefore, for every dimension d ∈ 1, . . . , D and every floating-point number f ∈ F, we can define a numerical predicate r such that:
r(n) = ⊤ if p(n) d = f, (84
) or r(n, m) = ⊤ if p(n, m) d = f.(85)
this section cite: []

Section: ■
The reverse direction and a precise logical characterization of commonly used positional encodings, e.g., sinusoidal and rotary [43], are left for future work.
this section cite: ['b42']

Section: G Related work
The expressivity of transformers in the context of formal methods has been extensively studied in recent literature [42]. Various transformer variants have been explored, and different assumptions have been made to enhance the expressivity of transformers.
this section cite: ['b41']

Section: Chain of thought (CoT).
CoT reasoning, which involves generating intermediate steps before arriving at a final answer, has become a popular approach. Pérez et al. [38] demonstrated that a transformer with both an encoder and a decoder, when allowed a polynomial number of intermediate computation steps, is Turing complete. Similarly, Merrill and Sabharwal [33], Nowak et al. [35], Li et al. [27] show that the expressivity of transformers can be improved with various forms of CoT reasoning. In this work, along with many others, we do not allow intermediate steps, which restricts expressivity. In such a case, Hao et al. [18], Merrill et al. [34], Merrill and Sabharwal [32], Chiang et al. [9], Yang and Chiang [49] have established various upper bounds on expressivity.
this section cite: ['b37', 'b32', 'b34', 'b26', 'b17', 'b33', 'b31', 'b8', 'b48']

Section: Non-fixed precision.
If a transformer is allowed arbitrary precision, or if the precision increases with input length, Bhattamishra et al. [3] proved that it is more expressive than simplified and stateless counter machines. Additionally, Chiang et al. [9] and Yang and Chiang [49] showed that a transformer with such precision can recognize any language definable by temporal counting logic. In contrast, our work focuses on the scenario where precision is fixed, which imposes more limitations on expressivity.
Bespoke positional encodings. Some prior studies have employed bespoke positional encodings, many of which cannot be represented under fixed precision, to overcome certain limitations of transformers. For example, Chiang and Cholak [8] used the encoding n/N to enable transformers to recognize parity, and Barcelo et al. [2] demonstrated that transformers with a specially designed positional encoding can achieve a lower bound of FO[<] with unary numerical predicates. In contrast, our constructions and proofs do not rely on any form of positional encoding.
Hard attention. Two forms of hard attention have been explored in the literature. Yang et al. [50] showed that masked fixed-precision transformers with UHA and NoPE recognize exactly the star-free languages. Barcelo et al. [2] established a similar lower bound for transformers with AHA and certain positional encodings. Prior to our work, it was unclear, under fixed precision, whether these two hard attention mechanisms were more or less expressive than, or comparable to, the standard soft attention. Surprisingly, we find that UHA is strictly more expressive than soft attention, while AHA is as expressive as soft attention.
this section cite: ['b2', 'b8', 'b48', 'b7', 'b1', 'b49', 'b1']

Section: H Limitations
The primary limitation of this work lies in the omission of positional encodings. While we briefly discuss their role as numerical predicates, the exact numerical predicates simulated by commonly used positional encodings remain unknown. We hope to explore this in future work. Nevertheless, we believe it is important to first understand the expressivity of a barebones transformer architecture, as this forms the foundation for systematically incorporating various forms of positional encoding later on.
Another limitation-particularly from an empirical perspective-is our stringent evaluation criteria for transformer performance. While certain levels of accuracy might be considered successful in empirical studies, we require perfect accuracy. This is motivated by a formal perspective: a model (e.g., an automaton or logical formula) either recognizes a language or it does not; anything short of perfection is regarded as failure, suggesting that some form of approximation or shortcut has been employed. We interpret our results as identifying a class of tasks that transformers have the full capacity to solve.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Code included in the supplementary material.
Guidelines: • The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: § § E and 6 Guidelines:
this section cite: []

Section: 
Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Johan Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew M. Dai, Andrew La, Andrew Kyle Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakaş, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartłomiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Bryan Orinion, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, Cesar Ferri, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Christopher Waites, Christian Voigt, Christopher D Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, C. Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Dylan Schrader, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodolà, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germàn Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Xinyue Wang, Gonzalo Jaimovitch-Lopez, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Francis Anthony Shevlin, Hinrich Schuetze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B Simon, James Koppel, James Zheng, James Zou, Jan Kocon, Jana Thompson, Janelle Wingfield, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Batchelder, Jonathan Berant, Jörg The strict inequality soft attention < UHA follows directly from the fact that LTL[P] is strictly less expressive than LTL[P, F, S, U]. ■
this section cite: []

Section: F.2 Non-strict future masking
Most commonly used transformers adopt non-strict future-masked soft attention, where each position is allowed to attend to itself. In this case, Eq. ( 24) and Eq. ( 26) become:
α n,m def = exp(S n,m ) i≤n exp(S n,i ) ,(77)
and A(H(w)) :,n
def = m≤n α n,m V :,m .(78)
respectively.
This modification, however, limits the expressiveness of the model. Theorem F.4. Transformers with non-strict future masking are strictly less expressive than those with strict future masking.
Proof. Since we have already established that transformers with strict masking are as expressive as PFO 2 [<], it suffices to show that any transformer with non-strict masking is strictly less expressive than PFO 2 [<].
We begin by observing that transformers with non-strict masking can be simulated by PFO 2 [<]. The required modification to Proposition B.4 involves including the current position in the threshold counting quantifiers:
∃ ≥1 y ≤ x : ϕ(y) def = ∃ ≥1 y < x : ϕ(y) ∨ ϕ(x),(79a)
∃ ≥2 y ≤ x : ϕ(y) def = ∃ ≥2 y < x : ϕ(y) ∨ ∃ ≥1 y < x : ϕ(y) ∧ ϕ(x) ,(79b)
. . . Next, we construct a language that is definable in PFO 2 [<] but cannot be recognized by a transformer with non-strict masking. Consider the language bbΣ * , which can be defined in PFO 2 [<] by the following sentence:
∃x : (π b (x) ∧ ∃ =1 y < x : π b (y) ∧ ¬∃y < x : ¬π b (y)).(80)
On the other hand, given any transformer H with non-strict masking, and for any string w ∈ bbΣ * , we have:
H (ℓ) (w) :,2 = H (ℓ) (w) :,1 for all layers ℓ ∈ {1, . . . , L}(81)
This is because, under non-strict masking, position n attends to all positions ≤ n, including itself. When the prefix consists entirely of identical symbols, each attention pattern and subsequent computation depend solely on the identical sequence of embeddings, resulting in identical representations at each position. However, differentiating the two leading bs is essential for recognizing bbΣ * , as the corresponding DFA will enter a different state upon reading the first symbol b. ■ Empirically, we confirm this limitation by training transformers-one with strict masking and one with non-strict masking-on the language bbΣ * . The results are consistent with our theoretical prediction:
• The strictly masked transformer achieves 100.0% maximum accuracy and 96.3% ± 5.8% average accuracy.
• The non-strictly masked variant reaches only 95.8% maximum accuracy and 74.4%±8.2% average accuracy.
this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Claims made in abstract and §1 accurately reflect the paper's contributions and scope.
Guidelines: • The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: §H.
Guidelines: • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Assumptions: § § B.1, 3.1 and 5, proofs: § § B.2, B.3, C, D and F.
Guidelines: • The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: A description of the experimental setup and data generation is provided in § § E.1.1, E.1.2 and 6.
Guidelines: • The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: Standard deviations are reported in Tabs. 1 and 3 and §F.2.
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Computer resources are reported in §E.1.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: We foresee no societal impact of this work.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper poses no such risks Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: Our code is adapted from Deletang et al. [10]. The authors are properly credited and the license and terms are explicitly mentioned and properly respected in §E.1.1.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: This paper does not introduce new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. Justification: This research does not involve crowdsourcing.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: ['b9']

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This research does not involve human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA] Justification: This research does not involve LLMs.
Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Layer normalization Year: (2016)
Ref_id:b1 Title: Logical languages accepted by transformer encoders with hard attention Year: (2024)
Ref_id:b2 Title: On the Ability and Limitations of Transformers to Recognize Formal Languages Year: (2020-11)
Ref_id:b3 Title: Languages of R-trivial monoids Year: (1980)
Ref_id:b4 Title: On a Decision Method in Restricted Second Order Arithmetic Year: (1990)
Ref_id:b5 Title: Training neural networks as recognizers of formal languages Year: (2025)
Ref_id:b6 Title: Transformers in uniform TC 0 . Transactions on Machine Learning Research Year: (2025)
Ref_id:b7 Title: Overcoming a theoretical limitation of self-attention Year: (2022-05)
Ref_id:b8 Title: Tighter bounds on the expressivity of transformer encoders Year: (2023-07)
Ref_id:b9 Title: Neural networks and the chomsky hierarchy Year: (2023)
Ref_id:b10 Title: A survey on small fragments of first-order logic over finite words Year: (2008)
Ref_id:b11 Title:  Year: (2024)
Ref_id:b12 Title: An until hierarchy for temporal logic Year: (1996)
Ref_id:b13 Title: First-order logic with two variables and unary temporal logic Year: (2002)
Ref_id:b14 Title: Counter machines and counter languages Year: (1968)
Ref_id:b15 Title: On the temporal analysis of fairness. POPL '80 Year: (1980)
Ref_id:b16 Title: Theoretical limitations of self-attention in neural sequence models Year: (2020)
Ref_id:b17 Title: Formal language recognition by hard attention transformers: Perspectives from circuit complexity Year: (2022)
Ref_id:b18 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b19 Title: Long short-term memory Year: (1997)
Ref_id:b20 Title: Music transformer Year: (2019)
Ref_id:b21 Title: A formal framework for understanding length generalization in transformers Year: (2025)
Ref_id:b22 Title: Ieee standard for floating-point arithmetic Year: (2008)
Ref_id:b23 Title: Definability with bounded number of bound variables Year: (1989)
Ref_id:b24 Title: Tense Logic and the Theory of Linear Order Year: (1968)
Ref_id:b25 Title: Representation of Events in Nerve Nets and Finite Automata Year: (1956)
Ref_id:b26 Title: Chain of thought empowers transformers to solve inherently serial problems Year: (2024)
Ref_id:b27 Title: Counter-free Automata Year: (1971)
Ref_id:b28 Title: Sequential neural networks as automata Year: (2019-08)
Ref_id:b29 Title: On the linguistic capacity of real-time counter automata Year: (2020)
Ref_id:b30 Title: A logic for expressing log-precision transformers Year: (2023)
Ref_id:b31 Title: The parallelism tradeoff: Limitations of log-precision transformers Year: (2023)
Ref_id:b32 Title: The expressive power of transformers with chain of thought Year: (2024)
Ref_id:b33 Title: Saturated transformers are constantdepth threshold circuits Year: (2022)
Ref_id:b34 Title: On the representational capacity of neural language models with chain-of-thought reasoning Year: (2024-08)
Ref_id:b35 Title: OpenAI. GPT-4 technical report. Computing Research Repository Year: (2023)
Ref_id:b36 Title: The temporal logic of programs Year: (1977)
Ref_id:b37 Title: On the turing completeness of modern neural network architectures Year: (2019)
Ref_id:b38 Title: Self-attention with relative position representations Year: (2018-06)
Ref_id:b39 Title: Piecewise testable events Year: (1975)
Ref_id:b40 Title: Vikas Raunak, Vinay Venkatesh Ramasesh, vinay uday prabhu Year: (2023)
Ref_id:b41 Title: What formal languages can transformers express? a survey Year: (2024-05)
Ref_id:b42 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b43 Title: Transformers can represent n-gram language models Year: (2024-06)
Ref_id:b44 Title: Diamonds are forever: The variety DA Year: (2002)
Ref_id:b45 Title: Attention is all you need Year: (2017)
Ref_id:b46 Title: Superglue: A stickier benchmark for general-purpose language understanding systems Year: (2019)
Ref_id:b47 Title: On the practical computational power of finite precision RNNs for language recognition Year: (2018-07)
Ref_id:b48 Title: Counting like transformers: Compiling temporal counting logic into softmax transformers Year: ()
Ref_id:b49 Title: Masked hard-attention transformers recognize exactly the star-free languages Year: (2024)
Ref_id:b50 Title: Self-attention networks can process bounded hierarchical languages Year: (2021-08)
