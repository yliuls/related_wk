Title: From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers
Abstract: Transformers can implement both generalizable algorithms (e.g., induction heads) and simple positional shortcuts (e.g., memorizing fixed output positions). In this work, we study how the choice of pretraining data distribution steers a shallow transformer toward one behavior or the other. Focusing on a minimal triggeroutput prediction task -copying the token immediately following a special trigger upon its second occurrence -we present a rigorous analysis of gradientbased training of a single-layer transformer. In both the infinite and finite sample regimes, we prove a transition in the learned mechanism: if input sequences exhibit sufficient diversity, measured by a low "max-sum" ratio of trigger-to-trigger distances, the trained model implements an induction head and generalizes to unseen contexts; by contrast, when this ratio is large, the model resorts to a positional shortcut and fails to generalize out-of-distribution (OOD). We also reveal a trade-off between the pretraining context length and OOD generalization, and derive the optimal pretraining distribution that minimizes computational cost per sample. Finally, we validate our theoretical predictions with controlled synthetic experiments, demonstrating that broadening context distributions robustly induces induction heads and enables OOD generalization. Our results shed light on the algorithmic biases of pretrained transformers and offer conceptual guidelines for data-driven control of their learned behaviors.

Section: Introduction
Large language models (LLMs) leverage circuits of attention heads [VSP + 17] to perform (implicit) algorithmic reasoning. Certain attention heads implement discrete algorithms -notably induction heads [ENO + 21, OEN + 22], which scan for previously seen token patterns in the context to predict subsequent tokens. Such heads enable in-context learning behaviors [BMR + 20], allowing a transformer to continue a sequence such as [A, B, . . . , A] → B purely by leveraging patterns in the context. By contrast, attention can also implement positional mechanisms that select tokens based solely on their location in the sequence [VTM + 19, AWKA24]. These mechanisms can yield contrasting generalization performance [CBKZ24], and we expect the pretraining data distribution to play a central role in determining which mechanisms a model learns to rely on: depending on structural properties of the corpus, a transformer may either discover generalizable strategies (content-based retrieval) or adopt position-based shortcuts.
-Solution 1: positional shortcut -Solution 2: induction head ⋯ t t ⋯ ℓ ⋯ t o t ⋯ p (T+1)/2 p T ⋯ t o t ⋯ p (T+1)/2 p T ⋯ t o t ⋯ e t e t ✖ 1) Match position T ↦ (T + 1)/2 2) Output here 1) Match trigger 2) Output next ℓ ⋯ t t ⋯ o o o o Pretraining Data ? ? ⋯ t t ⋯ o e t e t ? ? ℓ 1 ⏞ ℓ 2 OOD Test Data × Not Generalizable ✔ Generalizable
Figure 1: Two mechanisms for the associative copying task [..., t, o, ..., t] → o. In the pretraining data, the size of irrelevant tokens before the occurrence of the first and second trigger ℓ remains fixed per sequence, hence allowing two solutions: (i) positional shortcut that outputs the token at position (T + 1)/2 for input length T ; and (ii) induction head using token embedding e, which finds the queried token and returns the ensuing token. Whereas on OOD sequences with varying ℓ1 ̸ = ℓ2, only (ii) remains a valid solution.
Motivation. We theoretically study how pretraining data influences the implemented circuit and out-of-distribution (OOD) generalization performance of the transformer. This perspective is motivated from the empirical observation that pretrained models often leverage shortcut solutions that are brittle beyond the training distribution [MPL19, GJM + 20, LAG + 22]. For instance, a transformer might utilize the aforementioned position-based attention head to memorize that a certain output tends to occur at a particular position in the training text, instead of learning the underlying association (induction head); such positional shortcut is a double-edge sword in algorithmic tasks: transformers can achieve near-perfect accuracy in distribution, but struggle on test sequences of unseen lengths or structures. Since it is empirically known that the learned mechanism heavily depends on the structure of pretraining data [GTLV22, RLIGS22, RPCG23, WNB + 25], we ask the following question.
How does the data structure decide whether a pretrained transformer implements a generalizable mechanism (e.g.,induction head) or a shortcut that fails OOD (e.g., positional memorization)?
1.1 Our Contributions Trigger-output Copying. To investigate this question in a controlled setting, we introduce a minimal trigger-output copying task inspired by [BCB + 23]. In this synthetic task, each input sequence contains a special trigger token that appears twice. The model must predict the token that immediately follows the first trigger when the trigger appears the second time. For example, given
. . . [trigger][X] . . . [trigger][?] . . . ,
the correct prediction is X. Depending on the structure of the input sequence, this task admits multiple solutions. We focus on two mechanisms -see Figure 1.
• Induction head. The model attends back to the location of the previous trigger and copies the token following it; this works for arbitrarily long gaps between trigger occurrences (up to contextlength limit).
• Positional shortcut. When the position of the first trigger is inferable from the second (e.g., under periodic structure), the model may copy the token using positional information alone. This shortcut is valid in-distribution but does not reflect the underlying association. For this task, we define out-of-distribution (OOD) generalization as performance on test sequence with altered structure, where the trigger appears at positions not seen during pretraining (e.g., longer or aperiodic sequences). The induction head mechanism is robust to such shifts as it learns the correct association, whereas the positional shortcut typically fails OOD. Our goal is to identify a data-dependent transition between these two mechanisms that governs OOD generalization: intuitively, increasing the diversity of pretraining sequences -by varying the distances between trigger occurrences -dilutes positional signals and discourages the shortcut; conversely, as the number of trigger tokens in the data grows, the effective signal for induction weakens. We make these intuitions precise in our theoretical analysis.
this section cite: []

Section: Main Findings.
We provide a quantitative account of how pretraining data diversity shapes the mechanism learned by a pretrained transformer in the trigger-output copying task introduced above. Specifically, we rigorously analyze the in-distribution and out-of-distribution performance of a shallow (single-layer) transformer trained on this synthetic task. By studying an "early-phase" simplification of gradient descent in both the infinite-data (population loss) and finite-data (empirical loss) regimes, we show that the pretraining distribution directly selects the model's algorithm: when pretraining data are sufficiently "diverse" -as measured by a max-sum ratio of trigger distances -the transformer learns an induction head; when diversity is low, the model adopts a positional shortcut that fails to generalize OOD. Using this diversity measure that governs the phase transition, we discuss various tradeoffs and how to choose a pretraining distribution that induces the desired induction mechanism with minimal computational cost. Finally, we empirically probe the learned circuits by visualizing attention scores, and present evidence that a similar mechanism transition arises under standard gradient-based training beyond our theoretical setting.
this section cite: []

Section: Related Works
The induction head mechanism in transformers was first presented in the mechanistic interpretability literature [ENO + 21, OEN + 22], and followup theory investigates when such circuits emerge under simplified training dynamics and tasks [BCB + 23, ETE + 24, NDL24, Red24, CSWY24]. Empirical studies on algorithmic tasks (copying, arithmetic, sorting) demonstrate that transformers often rely on spurious "shortcut" solutions that fail to generalize, often due to poor use of positional information [ZBB + 22, JdDE + 23, ZBL + 23, GJB + 25]; the OOD brittleness of shortcut solution is also documented in [LAG + 22]. A complementary thread links the structure of pretraining data to in-context behaviors: the function classes a transformer implements in context and the sensitivity of performance to data statistics such as corpus coverage and frequency [GTLV22, RLIGS22, MLH + 22] or task diversity [RPCG23, LLZV + 25]. Our analysis aligns with this view by making explicit how diversity in trigger distances steers the learned mechanism. Methodologically, we borrow the "early-phase" simplification of training dynamics and study the loss improvement after the first few gradient descent step [BES + 22, DLS22, ORST23, BCB + 23].
this section cite: []

Section: Problem Setting
Notations. For a positive integer N , we denote [N ] := {1, 2, . . . , N }. For integers N 1 ≤ N 2 , we define [N 1 : N 2 ] := {N 1 , N 1 + 1, . . . , N 2 }. The Softmax function for an N -dimensional vector v ∈ R N is defined as Softmax(v) i :=
e v i N j=1 e v j . For a vector v, we write v = O 2 (f (N )) if ∥v∥ 2 = O(f (N )), and v = O ∞ (f (N )) if max i |v i | = O(f (N ))
. Similar notation is used for a matrix A, where ∥A∥ 2 and ∥A∥ ∞ denote its ℓ 2 → ℓ 2 spectral and max norms, respectively.
this section cite: []

Section: Data Generating Process
We study the trigger-output setting to investigate how transformers acquire the induction head mechanism. Let N ∈ N denote the vocabulary size and L ∈ N the maximum input sequence length. We designate special tokens as trigger tokens. We define our data model as follows: Definition 1 (Data Distribution). Let ℓ 1 , ℓ 2 ∈ N such that T := ℓ 1 + ℓ 2 + 3 ≤ L -1. Let N trg ≤ N denote the number of trigger tokens. A sequence z 1:T +1 ∈ [N ] T +1 is sampled as follows:
1. Sample a trigger token t ∈ [N trg ] and an output token o ∈ [N trg + 1 : N ] uniformly at random, where N trg = o(N 1/3 ).
2. Construct the sequence:
z 1:T +1 = ( z 1 , . . . , z ℓ1 ℓ1irrelevant
tokens , t, o trigger-output pair , z ℓ1+3 , . . . , z ℓ1+ℓ2+2 ℓ2 irrelevant tokens , t, o trigger-output pair
)
where irrelevant token z i (i ∈ [1 :
ℓ 1 ] ∪ [ℓ 1 + 3 : ℓ 1 + ℓ 2 + 2]) is drawn i.i.d. from [N trg + 1 : N ].
We refer to such a sequence as a trigger-output model with subtext lengths ℓ 1 and ℓ 2 .
In our data model, the task is to identify the output token z T +1 = o from the sequence z 1:T = (z 1 , . . . , z ℓ1 , t, o, z ℓ1+3 , . . . , z ℓ1+ℓ2+2 , t). This can be achieved by implementing the induction head mechanism [ENO + 21, OEN + 22], which copies the token that follows the first occurrence of the trigger token and outputs it upon encountering the second occurrence of the same trigger. Due to structure of the input sequence, transformer may also rely on positional shortcuts to achieve low loss; in particular, when the lengths of irrelevant tokens are identical within each sequence, i.e., ℓ 1 = ℓ 2 = ℓ, a transformer can achieve 100% training accuracy simply by inferring the correct position to attend to (T + 1)/2 = ℓ + 2 from the position of the second trigger T = 2ℓ + 3. Such positional solution does not make use of the semantic information and generally fails when ℓ 1 ̸ = ℓ 2 .
To study the transition between the two mechanisms, we assume the pretraining data consists of a mixture of sequences with different lengths determined by
ℓ = ℓ 1 = ℓ 2 . Definition 2. Consider a language model p θ (• | z 1 z 2 • • • z T ) that is pretrained on M sequences z (i) 1:T (i) +1
M i=1 generated as follows:
• Sample ℓ (i) from a distribution D ℓ .
• Generate z (i) 1:T (i) +1 according to Definition 1 with ℓ 1 = ℓ 2 = ℓ (i) , i.e., z
1:T (i) +1 = (z 1 , . . . , z ℓ (i) , t, o, z ℓ (i) +3 , . . . , z 2ℓ (i) +2 , t, o).
this section cite: []

Section: OOD Generalization.
Note that the pretraining distribution (defined by D ℓ ) may not cover all possible sequences. We say that p θ generalizes out-of-distribution (OOD) if it implements the correct copying mechanism across all possible ℓ's, that is, for any ℓ 1 , ℓ 2 such that ℓ 1 + ℓ 2 + 3 ≤ L -1 (possibly ℓ 1 ̸ = ℓ 2 ), and for any test sequence z 1:T +1 generated from the trigger-output unigram model with subtext lengths ℓ 1 and ℓ 2 (Definition 1), we have arg max
k∈[N ] p θ (k | z 1 z 2 • • • z T ) = z T +1 .
this section cite: []

Section: Gradient-based Training of Single-layer Transformer
Architecture and Embedding. We consider a single-layer transformer block f TF defined as
f TF (X 1:t ; W KQ , W V ) = W V X 1:t Softmax(X ⊤ 1:t W KQ x t ) ∈ R N , (2.1)
where W KQ ∈ R D×D , W V ∈ R N ×D and X 1:t = (x 1 • • • x t ) ∈ R D×t denotes the input embeddings of z 1:t , with embedding dimension D. We define the embedding as follows:
Definition 3. Let D = L + 2N . Let p t ∈ R L denote the one-hot vector with a 1 at the t-th position (representing the positional embedding), and let e z ∈ R N denote the one-hot vector with a 1 at the z-th position (representing the token identity).
We then construct the input embedding x t as
x t = p t e zt e zt-1 ∈ R L+2N . (2.2)
The prediction probability is given by
p (W KQ ,W V ) (z T +1 = k | z 1 • • • z T ) = [Softmax(f TF (X 1:t ; W KQ , W V ))] k .
Remark 1. We make the following remarks on the design of our architecture and embedding.
• The architecture (with the FFN is absorbed into the value matrix W V , and tied key and query projections) is commonly used in theoretical analyses and mechanistic studies [LLR23, BCB + 23, NDL24]; the simplification allows us to focus on the inductive bias by simple attention mechanisms, while retaining sufficient expressiveness to implement algorithmic behaviors.
• Two-layer architecture is typically needed to implement the induction head mechanism, where the first layer often learns to detect the trigger and identify of the following token via attention to the previous token [SHT24]. To reflect this inductive step in our simplified single-layer setting, we explicitly encode the identity of the previous token z t-1 in the third component of the embedding x t . This choice also echoes recent empirical developments that incorporate information of previous tokens directly into the current state, such as Mamba [GD23], RWKV [PAA + 23], and convolution augmentations [LZHO25,All25].
Algorithm 1: Gradient-based training of single-layer transformer Input :
Learning rate η KQ , η V Initialize W KQ (0) = O (L+2N )×(L+2N ) , W V (0) = O N ×(L+2N ) Gradient descent on W V W V (1) ← W V (0) -η V ∇ W V 1 M V M V i=1 L(X (i) 1:T (i) ; W KQ (0), W V (0)) Gradient descent on W KQ W KQ (1) ← W KQ (0) -η KQ ∇ W KQ 1 M KQ M V +M KQ i=M V +1 L(X (i) 1:T (i) ; W KQ (0), W V (1)) Output: Prediction f TF (•)
Gradient-based Learning Algorithm. We use gradient descent (Algorithm 1) on the crossentropy loss to pretrain our shallow transformer (2.1),
L(X (i) 1:T (i) ; W KQ , W V ) = CrossEntropy(e z T (i) +1 , Softmax(f TF (X 1:T (i) ; W KQ , W V ))).
In Algorithm 1, we apply a single gradient descent step with large learning rate on the value and key-query matrices. This is motivated by recent studies [ORST23, BCB + 23, WS24] showing that the first gradient step can induce associative memory tied to specific components of the input embedding. In particular, the gradient can often be expressed as a linear combination of outer products wv ⊤ , where either w or v corresponds to embedding vectors such as e zt , e zt-1 , or p t . Such a gradient structure is sufficient to construct simple forms of associative memory within the model. We remark that similar single-step update is commonly used in the analysis of feature learning in shallow neural networks [BES + 22, DLS22, BEG + 22] and transformers [OSSW24, NSO + 25, WNB + 25].
this section cite: ['b20', 'b7', 'b0']

Section: Main Result: Data-driven Transition Between Mechanisms

this section cite: []

Section: Positional Shortcut vs. Induction Head
In this section, we illustrate how the diversity of pretraining distribution influences which algorithm the trained transformer implements -either the positional shortcut or the induction head. The following quantity plays a central role in our characterization. Definition 4. For each ℓ, let q ℓ denote the probability mass assigned under D ℓ , and S the support of D ℓ . We define the max-sum ratio as
R(D ℓ ) = max ℓ∈S ℓ -1 q ℓ ℓ∈S ℓ -1 q ℓ .
this section cite: []

Section: Interpretation of max-sum ratio.
The max-sum ratio can be seen as a diversity measure of D ℓ .
The following example provides an intuitive illustration: Example 1. Let D ℓ = Unif({ℓ 0 , ℓ 0 + 1, . . . , ℓ 0 + K -1}). Then the max-sum ratio is given by
R(ℓ 0 , K) = ℓ -1 0 K-1 k=0 (ℓ 0 + k) -1 , (3.1)
which monotonically decreases with K; hence greater diversity of D ℓ gives smaller max-sum ratio.
Note that the max-sum ratio does not merely capture the width of the distribution: in Example 1, increasing ℓ 0 while keeping K fixed decreases the proportion of ℓ -1 0 in [ℓ -1 0 , . . . , (ℓ 0 +K -1) -1 ], thus reducing the max-sum ratio. Hence, even with a narrow range, shifting the distribution rightwardplacing more probability on larger ℓ -naturally yields a smaller max-sum ratio. This is because the max-sum ratio weights each probability mass q ℓ by ℓ -1 .
Learning under Population Loss. The next theorem shows the existence of a threshold in the max-sum ratio that determines whether OOD generalization is achieved, in the infinite-data limit. Theorem 5 (Infinite Sample Setting). Suppose we run Algorithm 1 on the expected loss
E[L(X 1:T ; W KQ , W V )] with learning rates η V ≲ 1, η V η KQ ≳ N 3 N 3 trg log N . Then, there exist ϵ 1 (N trg ), ϵ 2 (N trg ) = Θ(N -1
trg ) such that:
• If R(D ℓ ) < ϵ 1 , then the pretrained transformer generalizes OOD, as defined in Definition 2.
• If R(D ℓ ) > ϵ 2 , then there exist OOD test sequences such that the pretrained transformer fails. Remark 2.
• Note that the training data only contain sequences with ℓ 1 = ℓ 2 , and thus a positional shortcut (as illustrated in Figure 1) can still achieve 100% training accuracy. However, since the OOD test data include sequences with ℓ 1 ̸ = ℓ 2 , such shortcuts inevitably fail. Our main theorems show that the pretrained transformer avoids such shortcuts when the max-sum ratio is below a certain threshold, i.e., when the data distribution is sufficiently diverse.
• We also provide a tight Θ(N -1 trg ) characterization of the max-sum ratio threshold, indicating that increasing the number of possible triggers makes OOD generalization more difficult. The underlying mechanism is discussed in the ensuing subsection.
Learning under Empirical Loss. Our next result establishes (via gradient concentration) similar transition behavior in the finite-sample setting. Theorem 6 (Finite Sample Setting). Suppose we run Algorithm 1 with the same learning rate scaling as in Theorem 5, and with sample sizes M KQ ≳ poly log N •
N 3 N 2 trg ℓ √ q ℓ 2 and M V ≳ poly log N • N 5 N 2 trg ℓ∈S √ q ℓ ℓ∈S q ℓ ℓ -1 2 .
Then, with probability at least 0.99
there exist ϵ ′ 1 (N trg ), ϵ ′ 2 (N trg ) = Θ(N -1 trg )
such that the assertion of Theorem 5 holds by substituting (ϵ ′ 1 , ϵ ′ 2 ) for (ϵ 1 , ϵ 2 ).
this section cite: []

Section: Mechanism of Algorithm Selection
Now we take a closer look at how the positional shortcut and the induction head are implemented in the attention. We begin with the case where the support of D ℓ is a singleton and N trg = 1. After a single gradient step, the parameter matrix W KQ can be shown to implement a form of associative memory over the relevant embedding vectors. Lemma 7 (Informal). Let D ℓ = {ℓ}, and assume the trigger consists of a single token w. After one gradient step of Algorithm 1, W KQ takes the form
W KQ ∝ T (ℓ) -1 (p ℓ+2 + p ℓ+3 ) 0 e w p ⊤ T (ℓ) e ⊤ w 0 ,
where T (ℓ) = 2ℓ + 3 denotes the position of the second occurrence of the trigger token.
To further simplify the exposition, we ignore the cross terms between p and e and assume that W KQ takes the following form:
W KQ ∝ T (ℓ) -1 (p ℓ+2 + p ℓ+3 ) 0 0 p ⊤ T (ℓ) 0 ⊤ 0 ⊤ positional shortcut + T (ℓ) -1 0 0 e w 0 e ⊤ w 0 induction head (3.2)
Now consider an OOD test sequence as in Figure 1, whose total length matches the training sequence but whose first and second subtext lengths differ:
ℓ 1 + ℓ 2 = 2ℓ, ℓ 1 ̸ = ℓ 2 .
In this case, the two terms in (3.2) contribute to the attention score
Softmax X ⊤ 1:Ttest W KQ x Ttest with T test = ℓ 1 + ℓ 2 + 3 = T (ℓ)
as follows (see Figure 2), noting that x Ttest = p T (ℓ) e w * ⊤ :
• 1st term (positional shortcut). Regardless of ℓ 1 , it attends to the positions ℓ + 2 = (T test + 1)/2 and ℓ + 3 = (T test + 3)/2. In particular, for the former, even though ℓ 1 ̸ = ℓ 2 , the transformer incorrectly associates the second trigger position
T test with (T test + 1)/2 as if ℓ 1 = ℓ 2 . 2 p t e z t e z t-1 123456789 123456789 trg trg trg trg z = [* * * t o * * * t ?] 1 2 3 4 5 6 7 8 9 positional shortcut (a) Attention heatmap for ℓ ∼ Unif({3}). p t e z t e z t-1 123456789 123456789 trg trg trg trg z = [* * * t o * * * t ?] [* * * * t o * * * * t ?] [* * * * * t o * * * * * t ?] [* * * * * * t o * * * * * * t ?] 1 2 3 4 5 6 7 8 9 111315 11 13 15 induction head (b) Attention heatmap for ℓ ∼ Unif({3, 4, 5, 6}). In the left figure, there is a strong positional shortcut that links position 9 to position 5 (the correct position in pretraining data), whereas in the right figure, the trigger positions are more dispersed, weakening this shortcut. Instead, a signal corresponding to induction head -detecting tokens after trigger -becomes dominant.
• 2nd term (induction head). It attends to tokens whose third embedding block equals e w , i.e., tokens whose previous token is the trigger w. In other words, it scans for the trigger w = z Ttest and then attends to its next token -this is precisely the desired induction head behavior.
Thus, the learned attention matrix implements a mixture of positional shortcut and induction head, and the relative strength of these components determines which algorithm is ultimately selected. Two factors affect this balance: the diversity of irrelevant token length ℓ and the trigger size N trg .
Length distribution D ℓ . Equation (3.2) describes the case where ℓ is deterministic. When ℓ is distributed according to D ℓ , W KQ becomes a superposition over ℓ:
W KQ (1) ∝ ℓ q ℓ T (ℓ) -1 (p ℓ+2 + p ℓ+3 ) 0 0 p ⊤ T (ℓ) 0 ⊤ 0 ⊤ + E T (ℓ) -1 0 0 e w 0 e ⊤ w 0 .
Here, the first term spreads its mass across multiple positions and is consequently weakened, whereas the second term does not depend on ℓ and retains its strength. As a result, the magnitude of the former is at most max ℓ q ℓ T (ℓ) -1 , while that of the latter is ℓ q ℓ T (ℓ) -1 . Since T (ℓ) ≍ ℓ, the ratio between the strengths of positional memory and the induction head is nothing but the max-sum ratio R(D ℓ ). This explains why the max-sum ratio governs algorithm selection.
Trigger size N trg . In (3.2), when the trigger size N trg ≥ 2, the second term is replaced by
N -1 trg w∈[Ntrg] T (ℓ) -1 0 0 e w 0 e ⊤ w 0 ,
while the first term remains unchanged. Hence, the induction-head signal is split across trigger types and its strength decreases proportionally to N -1 trg . This explains Θ(N -1 trg ) threshold in Theorem 5. The above intuition is visualized in an experiment reported in Figure 2.
Example 2. In Figure 2, we set N = 16 and N trg = 2, train the model with D(ℓ) = 3 and D(ℓ) = Unif([3 : 8]), and visualize the resulting W KQ . The trigger-token set is {1, 2}. The training setting is the same as that in Section 4.1.
• (Left): when D(ℓ) = {3}, W KQ has a strong component that maps position 9 to position 5.
Although it also contains an induction head component that maps between trigger tokens, it is comparatively weak compared to the positional signal.
• (Right): when D(ℓ) = Unif([3 : 8]), W KQ exhibits a superposition of signals mapping position k to (k + 1)/2, which results in each individual signal being weakened. In contrast, the induction head signal does not diminish.
this section cite: []

Section: Tradeoff between Context Length and OOD Generalization
As discussed in Section 3.1, the max-sum ratio captures not only the overall "width" of the distribution but also decreases as mass shifts toward larger ℓ. This effect becomes especially pronounced near the Θ(N -1 trg ) threshold identified in Theorems 5 and 6: Example 3. Consider the max-sum ratio for the uniform distribution (3.1). If ℓ 0 = 1, then R(ℓ 0 , K) = Θ((log K) -1 ). To attain a max-sum ratio of order O(N -1 trg ) -the OOD generalization threshold in Theorems 5 and 6 -the support width must satisfy K ≳ exp(N trg ). By contrast, if ℓ 0 = Θ(N trg ), then it suffices to take K = Θ(N trg ) to obtain a max-sum ratio of O(N -1 trg ). Therefore, merely "widening" the distribution may not be efficient to reduce the max-sum ratio; biasing pretraining toward longer contexts is substantially more effective. This, in turn, suggests that reliably learning the induction-head mechanism (and hence achieving OOD generalization) may incur greater computational cost due to longer training sequences. We now consider the "optimal" shape of the pretraining sequence (under the constraint in Definition 2) that learns the induction-head mechanism with minimal compute. Since the forward-pass cost scales quadratically with context length, we seek short contexts while maintaining a favorable max-sum ratio. Formally, for U ≥ N trg , consider the optimization problem
P :          minimize U ℓ=1 q ℓ ℓ 2 subject to max U ℓ=1 q ℓ ℓ -1 U ℓ=1 q ℓ ℓ -1 ≤ N -1 trg U ℓ=1 q ℓ = 1 q 1 , . . . , q U ≥ 0
This objective is the sample-average forward-pass cost in pretraining; the constraints enforce the OOD threshold from Theorem 7 and the normalization of (q ℓ ) U ℓ=1 . This problem is a linear program whose optimizer is characterized below. Proposition 8. The optimal solution of problem P assigns linearly increasing probability mass to the first N trg context lengths and zero to the remaining ones:
(q 1 , q 2 , . . . , q U ) = Z -1 (1, 2, . . . , N trg , 0, . . . , 0),
where the normalization constant is Z = N trg (N trg + 1)/2.
In other words, to minimize average forward-pass cost per sample while meeting the OOD generalization constraint, the pretraining distribution should be linear in the context length, making q ℓ ℓ -1 uniform over ℓ ≤ N trg . We note that if one optimizes a different objective (e.g., incorporating sample complexity), the optimal pretraining distribution may change.
this section cite: []

Section: Numerical Experiments

this section cite: []

Section: Experiments for Theoretical Setting
To observe the transition from positional shortcut to induction head, we first consider the architecture defined in (2.1) and conduct experiments under the data model described in Definition 1.
this section cite: []

Section: Experimental Setup
Dataset. We generate training and test data according to the trigger-output setting in Definition 1.
• In the pretraining data, the lengths of irrelevant tokens ℓ 1 , ℓ 2 are always equal. We choose two integers ℓ min and ℓ max (ℓ min ≤ ℓ max ), and length ℓ is sampled from Unif([ℓ min , ℓ max ]). 25.0% 24.4% 33.9% 71.2% 88.1% 95.2% 95.1% 99.1% 100.0% 99.9% 100.0% 99.6% 19.6% 21.0% 27.4% 64.4% 88.4% 93.9% 95.4% 100.0% 98.9% 100.0% 100.0% 16.7% 19.6% 22.9% 64.6% 88.0% 93.6% 100.0% 98.8% 100.0% 100.0% 11.6% 13.7% 22.9% 61.4% 85.9% 97.4% 100.0% 100.0% 100.0% 11.1% 13.1% 20.4% 50.9% 86.5% 98.4% 100.0% 98.5% 11.0% 10.4% 18.9% 48.1% 90.8% 100.0% 100.0% 8.7% 7.8% 14.4% 54.7% 91.2% 100.0% 7.9% 9.5% 13.1% 43.4% 88.5% 9.7% 9.7% 16.3% 42.7% 9.3% 9.5% 11.7% 7.0% 6.4% 6.7% OOD Accuracy (Ntrg=4) 27.8% 23.8% 20.6% 20.7% 29.6% 42.5% 61.0% 75.1% 81.7% 84.6% 86.4% 89.8% 22.8% 20.4% 17.2% 17.6% 25.7% 31.9% 54.7% 71.3% 80.7% 85.1% 89.6% 17.2% 13.9% 13.7% 14.3% 18.7% 34.2% 52.9% 67.4% 82.7% 86.7% 12.5% 11.8% 13.0% 11.1% 18.2% 25.5% 47.3% 63.3% 75.8% 13.2% 12.9% 10.5% 9.4% 12.6% 23.3% 44.2% 65.5% 11.6% 9.7% 7.7% 8.3% 13.0% 17.5% 39.0% 9.3% 8.4% 8.1% 7.5% 11.4% 18.7% 8.4% 7.2% 7.1% 6.5% 10.4% 9.3% 7.7% 7.6% 6.8% 7.3% 7.9% 6.4% 6.2% 6.7% 6.7% OOD Accuracy (Ntrg=8) • In the OOD test data, we shift the position of the first trigger to produce non-periodic sequences. Specifically, we first sample ℓ ∼ Unif([ℓ min + 1, ℓ max ]), and then sample ℓ 1 ∼ Unif({1, . . . , 2ℓ -1} \ {ℓ}), defining ℓ 2 = 2ℓ -ℓ 1 so that ℓ 1 ̸ = ℓ 2 .
Model architecture, embedding, and training. We implement a one-layer transformer architecture as defined in (2.1) with embeddings defined in (2.2). Training follows Algorithm 1, and the learning rates for W V and W KQ are set to 10 3 and 10 4 , respectively. Both matrices are trained with the empirical cross entropy loss computed on 8192 training examples.
this section cite: []

Section: Empirical Observations

this section cite: []

Section: OOD Accuracy.
We conduct experiments for all combinations of ℓ min ∈ [3, 15] and ℓ max ∈ [3, 15] such that ℓ min < ℓ max , and evaluate all models on 1024 OOD test samples. The test accuracies (with different trigger size N trg ) are presented in Figure 3.
• OOD accuracy tends to increase as ℓ max increases (with ℓ min fixed). This suggests that a greater diversity in the training data biases the model towards the induction head.
• Comparing the left and right figures, we see that as the trigger size increases, the region where OOD generalization is achieved shifts rightward, suggesting an increased difficulty of induction head learning with larger N trg , as predicted by Theorem 6.
Error Visualization. Our theory predicts two characteristic error modes:
• Pseudo trigger position. For non-periodic OOD evaluation data with ℓ 1 + ℓ 2 = 2ℓ and ℓ 1 ̸ = ℓ 2 , let l = (ℓ 1 + ℓ 2 )/2. The positional shortcut maps the second-trigger position ℓ 1 + ℓ 2 + 3 to the pseudo output position l + 2. Accordingly, we measure the fraction of instances where the model outputs z l+2 and report this frequency as the pseudo accuracy rate.
• Leftmost position. Since the leftmost trigger in the pretraining data typically provides the strongest positional signal, the model may output z ℓmin+2 independent of the second trigger position. This error mode is especially likely when N trg is small. We record its frequency as the leftmost rate.
Figure 5 in Appendix E illustrates the existence of these positional shortcuts. We observe that the error rate due to the pseudo-trigger mechanism is higher near the diagonal, and both errors decline as ℓ max increases.
this section cite: []

Section: Experiments for Practical Settings
Next we examine whether a similar transition from positional shortcut to induction head occurs in more standard gradient-based pretraining beyond our theoretical simplification. We consider a threelayer transformer architecture with separated key-query matrices, MLPs, and residual connections, where all parameters are learned jointly using the AdamW optimizer [KB14,LH19]. The dataset is generated in the same way as in Section 4.1: we set N = 32 and N trg = 1, and varied ℓ min = 4, 8, . . . , 20, ℓ max = 4, 8, . . . , 40. More experimental details can be found in Appendix E.
8.0 12.0 16.0 20.0 24.0 28.0 32.0 36.0 40.0 max subtext length l_max 4.0 8.0 12.0 16.0 20.0 min subtext length l_min
43.0% 59.7% 65.6% 45.8% 74.8% 72.3% 78.3% 75.1% 71.0% 23.2% 44.9% 38.0% 55.0% 57.6% 58.8% 67.3% 66.5% 10.2% 29.4% 43.0% 39.6% 45.7% 55.4% 59.7% 4.8% 29.2% 37.3% 29.9% 44.0% 48.8% 4.8% 21.7% 21.9% 42.0% 36.0% OOD Accuracy (Full Training) 0 20 40 60 80 100 Accuracy (a) OOD accuracy 8.0 12.0 16.0 20.0 24.0 28.0 32.0 36.0 40.0 max subtext length l_max 4.0 8.0 12.0 16.0 20.0 min subtext length l_min 5.4% 7.2% 4.6% 23.8% 4.1% 3.5% 5.2% 3.9% 3.6% 19.1% 3.9% 23.9% 9.4% 9.2% 8.5% 6.1% 5.7% 29.7% 14.4% 16.1% 8.9% 18.6% 8.1% 5.9% 85.2% 22.9% 15.8% 17.5% 12.2% 7.8% 69.4% 13.1% 39.8% 6.0% 8.5% Pseudo Accuracy Rate (Full Training) 0 20 40 60 80 100 Accuracy (b) pseudo accuracy error rate 8.0 12.0 16.0 20.0 24.0 28.0 32.0 36.0 40.0 max subtext length l_max 4.0 8.0 12.0 16.0 20.0 min subtext length l_min 51.1% 11.3% 16.1% 15.1% 10.9% 15.0% 11.4% 13.1% 13.9% 31.2% 42.9% 14.9% 10.2% 16.0% 8.6% 5.5% 15.5% 24.8% 29.6% 6.5% 11.1% 6.1% 18.5% 6.4% 8.6% 27.9% 19.7% 7.1% 11.1% 15.2% 6.7% 6.6% 6.2% 16.1% 3.6% Leftmost Rate (Full Training)
this section cite: ['b2', 'b4']

Section: Empirical Observations.
Figure 4 shows the OOD accuracy, pseudo accuracy rate, and leftmost rate, following the same setup as in Section 4.1. Note that as the diversity of the pretraining distribution increases, the OOD generalization accuracy improves and the errors due to the positional shortcut decrease -this is consistent with our theoretical prediction in Section 3. We also observe that the transition point is less sharp compared to our theoretical setting.
this section cite: []

Section: Conclusion
In this work, using a simplified trigger-output task, we developed a theoretical analysis showing that gradient-based training implicitly selects between two distinct mechanisms with different outof-distribution generalization properties -an induction head or a positional shortcut. We introduced the max-sum ratio as a key quantity governing this selection. Our results demonstrate that the statistical structure of pretraining data critically shapes the algorithms internalized by transformers, offering quantitative insights into steering learning via data design. We conclude with several directions for future work. First, beyond absolute positional embeddings, it is important to characterize which positional shortcuts can arise under relative position embeddings and related variants. Second, while our analysis centers on a single-layer architecture, a two-layer model naturally delegates retrieval to the first layer (recovering the token corresponding to e zt-1 ); analyzing the coupled dynamics that emerge from this decomposition is an intriguing next step. Finally, developing methods to analyze and quantify richer classes of algorithmic biases -beyond the induction-shortcut dichotomy -would deepen our understanding of how pretraining distributions induce specific computational circuits.
[BCB + 23] Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. In Advances in Neural Information Processing Systems, 2023. [BEG + 22] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: Sgd learns parities near the computational limit. Advances in Neural Information Processing Systems, 35:21750-21764, 2022. [BES + 22] Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. High-dimensional asymptotics of feature learning: How one gradient step improves the representation. In Advances in Neural Information Processing Systems 35, 2022. [BMR + 20] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 2020. [CBKZ24] Hugo Cui, Freya Behrens, Florent Krzakala, and Lenka Zdeborová. A phase transition between positional and semantic learning in a solvable model of dot-product attention. Advances in Neural Information Processing Systems, 37:36342-36389, 2024. [CSWY24] Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Unveiling induction heads: Provable training dynamics and feature learning in transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2024. [DLS22] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory, pages 5413-5452. PMLR, 2022. [ENO + 21] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. [ETE + 24] Ezra Edelman, Nikolaos Tsilivis, Benjamin Edelman, Eran Malach, and Surbhi Goel. The evolution of statistical induction heads: In-context learning markov chains. In Advances in Neural Information Processing Systems, 2024. [GB10] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 249-256, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR. [GD23] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. [GJB + 25] Noah Golowich, Samy Jelassi, David Brandfonbrener, Sham M Kakade, and Eran Malach. The role of sparsity for length generalization in transformers. arXiv preprint arXiv:2502.16792, 2025. [GJM + 20] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665-673, 2020. [GTLV22] Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in neural information processing systems, 35:30583-30598, 2022. [JdDE + 23] Samy Jelassi, Stéphane d'Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, and Franc ¸ois Charton. Length generalization in arithmetic transformers. arXiv preprint arXiv:2306.15400, 2023.
this section cite: []

Section: References
Ref_id:b0 Title: Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers Year: (2025-05)
Ref_id:b1 Title: -context language learning: Architectures and algorithms Year: (2024)
Ref_id:b2 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b3 Title: Transformers learn shortcuts to automata Year: (2022)
Ref_id:b4 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b5 Title: How do transformers learn topic structure: Towards a mechanistic understanding Year: (2023)
Ref_id:b6 Title: Asymptotic theory of in-context learning by linear attention Year: (2025)
Ref_id:b7 Title: On the power of convolution-augmented transformer Year: (2025)
Ref_id:b8 Title: Uniqueness of solution in linear programming Year: (1979)
Ref_id:b9 Title: Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint Year: (2022)
Ref_id:b10 Title: Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference Year: (2019)
Ref_id:b11 Title: How transformers learn causal structure with gradient descent Year: (2024)
Ref_id:b12 Title: Nonlinear transformers can perform inference-time feature learning Year: (2025)
Ref_id:b13 Title:  Year: (2022)
Ref_id:b14 Title: On the role of attention in prompt-tuning Year: (2023)
Ref_id:b15 Title: Pretrained transformer efficiently learns low-dimensional target functions in-context Year: (2024)
Ref_id:b16 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b17 Title: The mechanistic basis of data dependence and abrupt learning in an incontext classification task Year: ()
Ref_id:b18 Title: Impact of pretraining term frequencies on few-shot reasoning Year: (2022)
Ref_id:b19 Title: Pretraining task diversity and the emergence of non-bayesian in-context learning for regression Year: (2023)
Ref_id:b20 Title: One-layer transformers fail to solve the induction heads task Year: (2024)
Ref_id:b21 Title: Attention is all you need Year: (2017)
Ref_id:b22 Title: Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned Year: (2019)
Ref_id:b23 Title: Learning compositional functions with transformers from easy-to-hard data Year: (2025)
Ref_id:b24 Title: Understanding knowledge hijack mechanism in in-context learning through associative memory Year: (2024)
Ref_id:b25 Title: Unveiling transformers with lego: a synthetic reasoning task Year: (2022)
Ref_id:b26 Title: What algorithms can transformers learn? a study in length generalization Year: (2023)
