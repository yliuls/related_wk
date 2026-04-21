Title: From Language Models over Tokens to Language Models over Characters
Abstract: Modern language models are internally-and mathematically-distributions over token strings rather than character strings, posing numerous challenges for programmers building user applications on top of them. For example, if a prompt is specified as a character string, it must be tokenized before passing it to the token-level language model. Thus, the tokenizer and consequent processing are very sensitive to the specification of the prompt (e.g., whether the prompt ends with a space or not). This paper presents algorithms for converting token-level language models to character-level ones. We present both exact and approximate algorithms. In the empirical portion of the paper, we benchmark the practical runtime and approximation quality. Across four publicly available language models, we find that-even with a small computation budget-our method is able to accurately approximate the character-level distribution at reasonably fast speeds, and that a significant improvement in the language model's compression rate (bits/byte) is achieved.

Section: Introduction
Modern language models are engineered as probability distributions over strings of tokens rather than strings of characters. However, this leads to a fundamental tension between the users of language models and the engineers who build them. Specifically, token-level models are rife with unintuitive behaviors that-without a technical fix-baffle users. As an illustrative example of a common user complaint, we exhibit the prompt boundary problem (see below). This paper provides a principled solution to Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
the prompt boundary problem as well as other oddities that make interfacing with token-level language models with character-level prompts hard for users.
Tokenized language models: A brief overview. 1 Let Σ be an alphabet of characters, and let Σ * denote the set of all strings that can be built from it. Suppose there is a true distribution p * Σ over Σ * that we seek to model. We observe a training corpus of character strings: σ (foot_0) , ... , σ (M ) i.i.d.
∼ p * Σ . However, rather than estimating a language model that approximates p * Σ directly, we employ a (possibly stochastic) tokenizer τ that transforms the training corpus into a corpus of token strings.
δ (1) ∼ τ (• | σ (1) ), ... , δ (M ) ∼ τ (• | σ (M ) ).
Next, we estimate a token-level language model to fit the strings δ (foot_5) , ... , δ (M ) . Lastly, we use p ∆ to generate character strings through the following generative process: (i) sample δ ∼ p ∆ and (ii) return σ = κ(δ) where κ is a decoding function. Let p Σ denote the resulting distribution of this process. Practically, we hope that the choice of τ and κ should aid in our ability to estimate p * Σ in the sense that p * Σ ≈ p Σ . Commonly used tokenizers in the realm of LLMs use tokenizers τ that break long strings into chunks. Intuitively, generating chunks instead of individual characters helps because it effectively shortens strings without obfuscating them using a complicated encoding.
The prompt boundary problem. Consider the case of GPT2 (Radford et al., 2019), which was trained over token strings created from byte-pair encoding (BPE; Sennrich et al.  (2016); Gage (1994)). Suppose we wish to generate continuations of the prompt: "In␣the␣kingdom␣of␣the␣blind,␣the. Unfortunately, the interface to p ∆ is not ideal: it does not accept a character string; thus, it is common to encode it as a token string with an encoding function τ BPE :foot_1,foot_2
τ BPE ("In␣the␣kingdom␣of␣the␣blind,␣the) = [ " 1 , In 818 , ␣the 262 , ␣kingdom 13239 , ␣of 286 , ␣the 262 , ␣blind 7770 , , 11 , ␣the 262 ]
If we complete the prompt by taking the most likely next token (greedy completion), we generate the following:
[ ␣one 530 , -12 , eyed 18834 , ␣man 582 , ␣is 318 , ␣king 5822 , ." 526 ]
Now that we have our generated output, we can apply the decoding function κ that maps token strings to character strings as part of the tokenization protocol:
κ BPE (•••) = "In␣the␣kingdom␣of␣the␣blind,␣the ␣one-eyed␣man␣is␣king."
This is a good completion, as the string is a well-known proverb. However, if we tweak the prompt ever so slightly by inserting a trailing whitespace: We formalize the prompt boundary problem as follows: Let σ ∈ Σ * be our initial prompt, such as "In␣the␣kingdom␣of␣the␣blind,␣the). Now, consider a pair of future strings σ ′ •α and σ ′ •β with a common prefix σ ′ (e.g., ␣one and ␣ills) We say a language model interface suffers from the prompt boundary problem if moving the common prefix σ ′ (e.g., ␣) across the boundary into the prompt does not preserve the relative probability of the pair of future strings, i.e., the relative probability of σ ′ •α and σ ′ •β given σ does not equal the relative probability of α and β given σ•σ ′ . Our example fails this test because the probability of ␣one was much higher than ␣ills, but then, the relative order swaps after moving ␣ into the prompt.
Our perspective is that the prompt boundary problem arises from incorrectly conditioning the token-level language model on a string of characters by using τ (σ) rather than finding token strings that best match the prompt σ. The most probable next token starting with ␣ is ␣one 530 . Now, generating the remaining tokens recovers the desired output, as it matches the prompt before we added the whitespace.
Unfortunately, backing up one token is insufficient for the general case, as the following example will illustrate. Consider generating from GPT2 using the prompt Hello,␣worl:
τ BPE (Hello,␣worl) = [ Hello 15496 , , 11 , ␣wor 476 , l 75 ]
The most likely next character ought to be d as Hello,␣world is a common expression (popularized in educational material). However, the most likely token results in Hello,␣worlwide, an apparent misspelling of worldwide. 5 Unfortunately, token healing's strategy of backing up by a single token cannot salvage the poor tokenization as l 75 is still the most common next token that is consistent with the string l. Thus, after generating
l 75 , we are back where we started, generating [ Hello 15496 , , 11 , ␣wor 476 , l 75 , wide 4421 ].
↩→ Getting it right: A simple "probability 101" expression tells us the correct solution to the prompt boundary problem. Consider a ∆ * -valued random variable Y , distributed according to p ∆ . Then, the correct way to sample from p ∆ conditioned on a character string σ is according to
p ∆|Σ (δ | σ) def = P Y ∼p∆ [Y = δ | κ(Y ) ⪰ σ](1)
where we have conditioned on the event that the decoded string κ(Y ) has σ as a prefix (i.e., κ(Y ) ⪰ σ). While innovative with respect to the literature, the expression
PY ∼p∆ [Y = δ | κ(Y ) ⪰ σ]
conveys the probability we are interested in precisely and concisely. For the more procedurally minded, this corresponds to the following process:
1 def conditional_token_generation(σ): 2 while True:
3 sample δ ∼ p ∆ 4 if κ(δ) ⪰ σ: return δ # accept
This is, of course, very inefficient, but fear not-we will provide an equivalent, efficient algorithm.
Our method finds a set of token strings that form a covering, a key technical concept we introduce in this paper. We will provide the precise definition in due course; for now, we will illustrate the covering of Hello,␣worl:
(0.9820714) : [ Hello 15496 , , 11 , ␣world 995 ] (0.0106702) : [ Hello 15496 , , 11 , ␣worlds 11621 ] (0.0070749) : [ Hello 15496 , , 11 , ␣worldwide 8688 ] (0.0000830) : [ Hello 15496 , , 11 , ␣worldly 43249 ] (0.0000369) : [ Hello 15496 , , 11 , ␣worldview 29081 ] (0.0000225) : [ Hell 28254 , o 78 , , 11 , ␣world 995 ] (0.0000179) : [ Hello 15496 , , 11 , ␣wor 476 , l 75 ] . . .
Notice that each token string in the covering has the character string Hello,␣worl as a prefix (after decoding), but it may have some extra characters (marked by underlining) in the partially matched last token-just like token healing. The size of the complete covering may, in general, be exponential in the length of the character string. However, we can enumerate its high-probability members very quickly in practice. Since the worst-case time to find the top-K elements of the covering might be exponential, we provide a more aggressive approximation based on beam search that gives a good approximation even with small beam sizes.
Unlike token healing, the sequence [ Hello 15496 , , 11 , ␣world 995 ] is the highest probability member of the covering; thus, it is not pruned away by the aggressive heuristics based on τ , as in token healing. The covering is defined only in terms of κ, and the pruning is based on the language model probability, which typically prioritizes the token strings that are most reflective of the language model's training data. We provide an algorithm for correctly conditioning a token-level model on a character string in §3.4.
Character-level model. At the character level, working with the language model is intuitive. Consider, again, our example illustrating the prompt boundary problem. Appending whitespace behaves predictably, as it satisfies the probabilistic chain rule:
-→ p Σ (␣one | "In␣the␣kingdom␣of␣the␣blind,␣the) = -→ p Σ (␣ | "In␣the␣kingdom␣of␣the␣blind,␣the) • -→ p Σ (one | "In␣the␣kingdom␣of␣the␣blind,␣the␣)
Here, -→ p Σ denotes the character-level model's conditional distribution. Similarly, recall the Hello,␣world example above. The character-level model correctly infers that d is the most likely next character given Hello,␣worl. The computation of this conditional probability is simply the total probability of the covering of Hello,␣world divided by the total probability of the covering of Hello,␣worl. These quantities are derived from our concept of covering, which directly leads to an algorithm for determining the distribution over possi-ble next characters. Beyond the prompt boundary problem, computing the conditional probability of a character string given a token-level language model has many applications.
Applications. Aside from making character-level conditioning well-behaved, we highlight a few applications of language models requiring careful character-level reasoning: ↩→ Character-level constraints: Enforcing character-level constraints on allowed strings is a promising area that has received much recent attention (e.g., Scholak et al., 2021;  Poesia et al., 2022; Geng et al., 2023; Microsoft, 2023;  Willard & Louf, 2023; Koo et al., 2024; Loula et al., 2025). ↩→ Computational psycholinguistics: Computing the contextual surprisal (negative log probability) of a character substring to predict human reading times (Hale, 2001; Levy,  2008). Two recent papers (Oh & Schuler, 2024; Pimentel  & Meister, 2024) have given algorithms for computing the surprisal of whitespace-separated words under a number of strong assumptions. Our algorithms can compute the contextual surprisal of arbitrary character strings. Giulianelli et al.  (2024) show experimentally that having the flexibility to compute character substring surprisals leads to a more predictive model of reading time than a fixed notion of a word.
Does it work? In the experimental portion of our paper ( §4), we report the empirical runtime of our algorithm for converting token-level language models to character-level ones and quantify its accuracy in estimating the conditional distribution over characters. We find that even with a limited computational budget, our method provides an accurate estimate of the conditional distribution over the next character under four publicly available language models. 6 We also find that the compression rate (bits/byte) is significantly improved by estimating the probability of the corpus as a character string rather than a canonical token string.
this section cite: []

Section: Background

this section cite: []

Section: Alphabets and Strings
An alphabet Γ is a non-empty, finite set of elements called symbols.
A string γ over alphabet Γ is a finite sequence γ = γ 1 ••• γ N for some 0 ≤ N < ∞ of symbols where γ 1 , ... , γ N ∈ Γ.
Let |γ| denote the string's length N . We denote the empty string as ε. For any alphabet Γ, let Γ * denote the set of all strings over Γ, and let Γ + denote the set of all non-empty strings over Γ. For any two strings γ ′ , γ ′′ ∈ Γ * , we denote their concatenation as
γ ′ •γ ′′ . Additionally, we define S•S ′ def = {γ•γ ′ | γ ∈ S, γ ′ ∈ S ′ } for any S, S ′ ⊆ Γ * . Given a string γ such that |γ| ≥ t, let γ <t denote the string made from the first t-1 characters of γ. We write γ ⪯ γ ′ if γ is a prefix of γ ′ and γ ≺ γ ′ if γ is a proper prefix of γ ′ .
The relation ⪯ defines a partial order on Γ * . We write ⪰ and ≻ to refer to the relations ⪯ and ≺ with their respective arguments transposed.
this section cite: []

Section: Language Models and Prefix Probability
A language model p Γ is a probability distribution over Γ * where Γ is an alphabet. Let Y be a Γ * -valued random variable distributed according to p Γ and γ ∈ Γ * . The prefix probability -→ p Γ (γ) is the probability that Y has prefix γ:
-→ p Γ (γ) def = P Y ∼pΓ [Y ⪰ γ] = γ ′ ∈Γ * 1{γ ′ ⪰ γ} p Γ (γ ′ ) (2)
We also define the following shorthand for the conditional prefix probability
-→ p Γ (γ ′ | γ) as the probability of the event Y ⪰ γ•γ ′ provided that Y ⪰ γ: -→ p Γ (γ ′ | γ) def = P Y ∼pΓ [Y ⪰ γ•γ ′ | Y ⪰ γ] = -→ p Γ (γ•γ ′ ) -→ p Γ (γ)(3)
We may express the probability of γ as a product of conditional prefix probabilities:
p Γ (γ) = -→ p Γ (EOS | γ) |γ| t=1 -→ p Γ (γ t | γ <t )(4)
where each -→ p Γ (γ t | γ <t ) is an instance of Eq. ( 3), and
-→ p Γ (EOS | γ) def = p Γ (γ) -→ p Γ (γ)(5)
Here, EOS is a distinguished end-of-string symbol that cannot appear in any alphabet. Of particular interest are the single-symbol conditional prefix distributions -→ p Γ (• | γ <t ), as they may each be interpreted as a probability distribution over the set Γ ∪ {EOS}-in fact, modern language models 7 are defined via the product in Eq. ( 4) where each single-symbol conditional prefix probability comes from the learned parametric model. 8
this section cite: []

Section: Tokenization
We now discuss our basic formalization for tokenization.
Definition 1. An (exact) tokenization model is a tuple (Σ, ∆, τ , κ) where • Σ is an alphabet of character symbols 7 E.g., transformers (Vaswani et al., 2017), RNNs (e.g., Mikolov  et al., 2010), and n-gram models (e.g., Shannon, 1948). 8 The reader may notice that the equations for pΓ(γ), -→ pΓ(γ ′ | γ), and -→ pΓ(EOS | γ) are mutually recursive. Thus, they may appear circular. The key to resolving this concern is to recognize that pΓ is given as a base case. We note, however, that some readers may view Eq. ( 4) as the definition of the language model pΓ, taking the components on the right-hand side of Eq. ( 4) as the base case.
• ∆ is an alphabet of token symbols • τ is a (possibly) stochastic encoder:
τ (• | σ) is a proba- bility distribution over ∆ * for each σ ∈ Σ * • κ : ∆ * → Σ * is a decoder function satisfying exactness δ∈∆ * 1{κ(δ) = σ}τ (δ | σ) = 1, for all σ ∈ Σ *
Definition 2. A tokenized language model p Σ is a language model over Σ * that is parameterized by a language model p ∆ over ∆ * and a decoding function κ : ∆ * → Σ * . This tokenized language model generates character strings via the following process: (i) δ ∼ p ∆ , (ii) σ ← κ(δ). Thus, the character strings σ generated have the distribution:
p Σ (σ) def = P Y ∼p∆ [κ(Y ) = σ](6)
Note that p Σ (σ) accounts for the fact that many token strings may be associated with a given character string through κ. 9 To describe that association, we define E(σ) def = {δ ∈ ∆ * : σ = κ(δ)}, the set of encodings for any character string σ ∈ Σ * . 10 What about τ ? The reader may notice that τ does not appear in Eq. ( 6). Although τ is essential for generating training data, once the model p ∆ has been trained, the information in τ is not of immediate practical use. Moreover, attempts to leverage τ seem to lead to faulty heuristics, as we discussed in the introduction. We note that under exactness, τ (σ) must be present in E(σ). This is because exactness implies that E(σ) ⊇ {δ ∈ ∆ * : τ (δ | σ) > 0} for all σ ∈ Σ * . In the common case where τ is deterministicfoot_13 we emphasize that E(σ) ⊇ {τ (σ)}. The tokenization model would need to be bijectivefoot_14 for E(σ) = {τ (σ)}. Unfortunately, common tokenizers (e.g., BPE) are not bijective.
The mirage of the canonical tokenization. Consider the case when the encoder τ is deterministic. In that case, we write δ = τ (σ), and we call this δ the canonical tokenization of σ. Note that even if τ is deterministic, there may exist many noncanonical tokenizations δ ′ ∈ E(σ) such that δ ′ ̸ = τ (σ) with nonzero probability p ∆ (δ ′ ) > 0. Thus, the character string generation process includes a mix of canonical and noncanonical token strings-making it incorrect to only consider a character string's canonical tokenization when assessing its probability. In practice, the conditional probability PY ∼p∆ [Y = δ | κ(Y ) = σ] over the encodings δ of a character string σ tends to be highly concentrated on the canonical tokenizations, as illustrated in Example 1 below.
Example 1. Below, we show GPT2's top encodings from E(Hello,␣world), ranked by their conditional probability.
This short string has 78 encodings from numerous partitions of Hello,␣world into substrings of the tokenization alphabet ∆. We see that the probability heavily concentrates on the top string, which is canonical.
(0.9999719) : [ Hello 15496 , , 11 , ␣world 995 ] (0.0000229) : [ Hell 28254 , o 78 , , 11 , ␣world 995 ] (0.0000024) : [ Hello 15496 , , 11 , ␣wor 476 , ld 335 ] (0.0000017) : [ He 1544 , llo 18798 , , 11 , ␣world 995 ] (0.0000004) : [ H 39 , ell 695 , o 78 , , 11 , ␣world 995 ] (0.0000002) : [ Hello 15496 , , 11 , ␣w 266 , orld 1764 ] (0.0000002) : [ H 39 , ello 11109 , , 11 , ␣world 995 ]
A character-level interface. A character-level interface to the token-level language model p ∆ is available in the following equations, which hold ∀σ, σ ′ ∈ Σ * :
-→ p Σ (σ) = P Y ∼p∆ [κ(Y ) ⪰ σ](7)
-→ p Σ (σ ′ | σ) = -→ p Σ (σ•σ ′ ) -→ p Σ (σ)(8)
-→ p Σ (EOS | σ) = p Σ (σ) -→ p Σ (σ)(9)
These equations show that we can have a complete characterlevel language model derived from the tokenized language model if we can compute-or approximate-the necessary summations implied by Eq. ( 6) and ( 7); specifically,
p Σ (σ) = δ∈∆ * 1{κ(δ) = σ} p ∆ (δ) (10
)
-→ p Σ (σ) = δ∈∆ * 1{κ(δ) ⪰ σ} p ∆ (δ)(11)
We will develop effective methods for these summations for the family of strict-prefix monotone decoders κ (described in §2.4) where Eq. ( 10) and Eq. ( 11) admit a finite summation.
this section cite: []

Section: Key Properties of κ
This section defines the essential properties of κ we require.
Definition 3. We say κ :
∆ * → Σ * is • prefix monotone if δ ⪯ δ ′ =⇒ κ(δ) ⪯ κ(δ ′ ) • strict-prefix monotone if δ ≺ δ ′ =⇒ κ(δ) ≺ κ(δ ′ )
In simpler terms, strict-prefix monotonicity implies that concatenating a token to the encoding necessarily concatenates at least one character to the decoded character string. The diagrams below illustrate how GPT2's strict-prefix monotone κ gives rise to a certain alignment between three token strings and the character string Hello,␣world:
Hello,␣world
Hello 15496 , 11 ␣world 995 Hello,␣world H 39 ell 695 o 78 , 11 ␣world 995 Hello,␣world Hello 15496 , 11 ␣wor 476 l 75 d 67
More formally, every application
σ 1 ••• σ M = κ(δ 1 ••• δ M )
of a strict-prefix monotone mapping has the following properties. Each token in δ 1 ••• δ M maps to one or more contiguous characters in σ 1 ••• σ M . Moreover, the mappings do not exclude any characters, and no edges of the mapping cross one another. Strict prefix monotonicity, in contrast to prefix monotonicity, ensures that there are no deletions of tokens in the mapping, i.e., each token maps to at least one character.
Strict-prefix monotonicity is the key structural property required by §3's algorithms, as it allows us to replace an infinite sum with a finite sum in Proposition 1. We briefly mention an important special case.
A decoder κ is multiplica- tive if κ(δ 1 ••• δ N ) = κ(δ 1 ) ••• κ(δ N ) for all δ 1 ••• δ N ∈ ∆ * , and non-erasing if κ(δ) = ε =⇒ δ = ε. If κ is multi- plicative, it is prefix monotone; if it is also non-erasing, it is strict-prefix monotone.
Both BPE and WordPiece are multiplicative and non-erasing.
this section cite: []

Section: Algorithms
This section gives algorithms for computing p Σ (σ), -→ p Σ (σ), -→ p Σ (σ ′ | σ), -→ p Σ (EOS | σ), and conditional token generation. We assume throughout that κ is strict-prefix monotone.
this section cite: []

Section: Covering
Eq. ( 11) shows that we can, in principle, compute the prefix probability -→ p Σ (σ) by summing over prefix-encodings of σ, P(σ) def = {δ ∈ ∆ * : κ(δ) ⪰ σ}. Although P(σ) is infinitely large, we can exploit the prefix monotone structure of κ to find a different way to perform the summation by summing over a finite set. Let δ ∈ ∆ * , δ 1 ••• δ M = δ, and σ ∈ Σ * . We say that δ covers σ if and only if κ(δ) ⪰ σ. Monotonicity ensures that for all δ ∈ P(σ), we have that ∀δ ′ ∈ ∆ * : κ(δ•δ ′ ) ⪰ σ. In other words, any δ that decodes to an extension of σ (i.e., κ(δ) ⪰ σ) will continue to do so if we append tokens to it. Thus, we may additionally qualify the relationship as δ minimally covers σ if additionally κ(δ 1 ••• δ M -1 ) ≺ σ. With that in mind, we define ϕ σ (δ) as the shortest prefix δ ′ ⪯ δ such that κ(δ ′ ) ⪰ σ, i.e., ϕ σ maps any δ that covers σ to a (possibly equal) token string that minimally covers σ. Next, we define the set of minimal prefix encodings of σ, which we call the covering of σ, C(σ) def = {ϕ σ (δ) | δ ∈ P(σ)}. A more convenient expression for the covering C(σ) of a string σ ∈ Σ * is equal to the following subset of ∆ * :
C(σ) =      {ε} if σ = ε {δ 1 ••• δ M ∈ ∆ + : otherwise κ(δ 1 ••• δ M -1 ) ≺ σ ⪯ κ(δ 1 ••• δ M )} (12) Example 2.
Recall the covering of the string σ = Hello,␣worl from the introduction. We have repeated it on the right and couched it in our terminology. Note that the complete covering C(Hello,␣worl) contains 36,608 token strings; we only show the top strings according to their respective -→ p ∆ . Below, we list several properties and observations about the structure of the covering:
[ Hello 15496 , , 11 , ␣world 995 ] [ Hello 15496 , , 11 , ␣worlds 11621 ] [ Hello 15496 , , 11 , ␣worldwide 8688 ] [ Hello 15496 , , 11 , ␣worldly 43249 ] [ Hello 15496 , , 11 , ␣worldview 29081 ] [ Hell 28254 , o 78 , , 11 , ␣world 995 ] [ Hello 15496 , , 11 , ␣wor 476 , l 75 ]
• The covering for any given string σ always has the property that each non-empty token string δ in the covering decodes to a string that has σ as a prefix, i.e., σ ⪯ κ(δ). This is illustrated by the gloss string in the tokenization.
• Note that δ may include a partially matched token at its end (i.e., δ M in Eq. ( 12)). We have marked the extra characters by underlining them. We note that the 7 th member does not have a partially matched last token.
• Each token string in the covering has at most one partially matched token thanks to the condition κ(δ) ≺ σ ⪯ κ(δ•δ). The 7 th member of the cover has a completely matched last token; hence, there is no underlining.
• We see that if we were to extend any member δ ∈ C(σ)
with an arbitrary string of additional tokens δ ′ , it would continue to decode to a string such that κ(δ•δ ′ ) ⪰ σ. Moreover, δ is minimal (i.e., ϕ σ (δ) = δ).
The notion of a covering is used to derive an algorithm for computing character-level probabilities given a tokenlevel language model. We first show how it gives us the prefix probability and subsequently give equations for the remaining quantities of the character-level language model.
Proposition 1. Suppose (Σ, ∆, τ , κ) is a tokenization model where κ is strict-prefix monotone and p ∆ is a token-level language model. Then, the prefix probability -→ p Σ (σ) for the character-level model Eq. ( 6) is given by
-→ p Σ (σ) = δ∈C(σ) -→ p ∆ (δ), ∀σ ∈ Σ *(13)
Proof. See App. B. ■ Eq. ( 13) is a substantial improvement over Eq. ( 7) for computing -→ p Σ (σ). Specifically, we now have a finite sum, as |C(σ)| is finite for all σ ∈ Σ * . Bear in mind that the covering's size is likely too large to be practical, as there may still be a large number of summands; however, the set of high-prefix-probability elements of the covering tends to be reasonably small, an observation that we verify in §4, and leverage to develop practical algorithms in §3.
Lastly, we note that the covering contains the set of encodings, i.e., E(σ) ⊆ C(σ); hence, the encodings may be extracted from the covering as follows: E(σ) = {δ ∈ C(σ) : κ(δ) = σ}. We may also express the probability of σ in terms of the covering:
p Σ (σ) = δ∈C(σ) 1{κ(δ) = σ} p ∆ (δ)(14)
3.2. Algorithms for -→ p Σ (σ) and p Σ (σ)
The enumeration algorithm will enumerate elements of the covering along with their prefix probability (for convenience). It filters prefixes of token strings that cannot eventually cover the target string σ. The strict-prefix monotonicity property is essential for this filtering.
Our algorithm enum_cover performs recursive enumeration of the members of the covering C(σ) along with some metadata. Specifically, the algorithm returns a collection of triples where each triple (p ′ , σ ′ , δ ′ ) satisfies δ ′ ∈ C(σ), p ′ = -→ p ∆ (δ ′ ), and σ ′ = κ(δ ′ ).
5 def enum_cover(σ 1 ••• σ N ): 6 if N = 0: return [(1, ε, ε)] # base case 7 out ← [] 8 for (p ′ , σ ′ , δ ′ ) in enum_cover(σ 1 ••• σ N -1 ): 9 if |σ ′ | < N : # extend 10 for δ ′′ ∈ ∆: 11 σ ′′ ← κ(δ ′ •δ ′′ ) 12 if σ ′′ N = σ N : # filter 13 out.append((p ′ • -→ p ∆ (δ ′′ | δ ′ ), σ ′′ , δ ′ •δ ′′ )) 14 elif σ ′ N = σ N : # filter character matches 15 out.append((p ′ , σ ′ , δ ′ )) 16 return prune(σ 1 ••• σ N , out)
Note that this method has an additional parameter, the function prune, which is used on the last line. This method, as the name suggests, is used to limit the size of the covering to prevent excessive growth. We will discuss this parameter shortly. For now, consider the following definition:
17 def prune_nothing(σ 1 ••• σ N , out): 18 return out
From the output of the enumeration algorithm, we can compute the other key objects and quantities (i.e., C(σ), -→ p Σ (σ), E(σ), p Σ (σ)) in the character-level interface.
Time and space complexity. To meaningfully discuss its running time, we assume the following:
• κ(δ ′ •δ ′′ ) can be evaluated in constant time given κ(δ ′ ).
• the cost of evaluating -→ p ∆ (δ t | δ <t ) is constant given that -→ p ∆ (δ s | δ <s ) has been computed for 0 ≤ s < t. Pruning. We now consider some useful pruning heuristics for the algorithm, which make it an approximation but substantially improve its running time. We propose a heuristic based on beam search. This heuristic is very effective: it gives us a linear running time as a function of the character string's length. It has a parameter K that controls the approximation quality. Larger K makes the approximation more accurate, and the approximation becomes exact as K approaches the size of the (largest intermediate) covering. We take K to be a global variable in the pseudocode.
this section cite: []

Section: ↩→
Our pruning heuristic: Our pruning heuristic enumerates ≤ K distinct token strings modulo their last token. This choice allows up to |∆| versions of the last token to be enumerated (if it is not completely matched). Thus, the work done at each step is O(K • |∆|), and the size of the pruned list is at most that size. Therefore, the overall running time is O(N • K • |∆|) for a character string of length N .
22 (p, σ ′ , δ 1 ••• δ M ) ← item 23 # Exclude a partially matched last 24 key ← δ 1 ••• δ M -1 if |σ ′ | > N else δ 1 ••• δ M
25 buckets[key].append(item) 26 pruned ← [] 27 for bucket in buckets.top(K): # by prob. 28 for item in bucket: 29 pruned.append(item) 30 return pruned
Bundled beam summing implementation. App. D describes an implementation strategy that improves the constant factors associated with the pseudocode above. The key idea is to group the token sequences that fall into the same bucket in prune_top_K_buckets into a bundle that represents them compactly. In particular, we can use a trie 13 In the case of the common transformer language model (Vaswani  et al., 2017), this can be achieved with efficient caching and limiting context windows to a constant size. 14 Note: finding the (unordered) set of top-K elements from a set S is possible in O(|S|) time via the median-of-medians algorithm.
to efficiently filter out the next tokens that disagree with the next character. We can regard the trie as a local language model that generates the next token character-by-character according to the probability assigned by -→ p ∆ (• | δ). Each bundle can be unbundled (if necessary) into the respective tuples that the enum_cover algorithm maintains.
this section cite: []

Section: Algorithms for -→ p Σ (• | σ)
This section gives algorithms for computing the characterlevel conditional prefix probability. Recall the definition of the character-level conditional prefix probability, that is, Eq. ( 8) and ( 9), can be computed from a certain ratio of calls to -→ p Σ (and p Σ in the case of EOS). From here, Eq. ( 8) and ( 9) give a straightforward algorithm for computing the distribution over Σ ∪ {EOS} given σ ∈ Σ * . However, a direct translation would perform duplicate work. Therefore, we provide the following version, which reuses work between the calls to -→ p Σ than directly evaluating those equations would do.
31 def next_character_probability(σ):
32 N ← |σ|; Z ← 0; p ← {σ ′ : 0 for σ ′ ∈ Σ∪{EOS}} 33 for (p ′ , σ ′ , δ ′ ) ∈ enum_cover(σ): 34 Z += p ′ 35 if |σ ′ | = N : # i.e., σ ′ = σ 36 p(EOS) += p ′ • -→ p ∆ (EOS | δ) 37 for δ ′′ ∈ ∆: # extend 38 σ ′′ ← κ(δ ′ •δ ′′ ) 39 p(σ ′′ N +1 ) += p ′ • -→ p ∆ (δ ′′ | δ ′ ) 40 else: # i.e., σ ′ ⪰ σ 41 p(σ ′ N +1 ) += p ′ 42 return p/Z # Z = -→ p Σ (σ) 3.4. Conditional Generation p ∆|Σ (δ | σ)
This section gives a simple algorithm for correctly generating a token string Y that has a given character-level prompt σ as its prefix. This algorithm is equivalent to the algorithm in the introduction but significantly faster.
The algorithm works by enumerating the covering C(σ), drawing a token string from it in proportion to its prefix probability, and finishing the token string by sampling a completion, which can be done from the token-level model.
43 def conditional_token_generation(σ):  The following proposition establishes correctness: Proposition 2. conditional_token_generation(σ) generates samples according to p ∆|Σ (• | σ) for all σ ∈ Σ * .
44 δ ′ ∼ Categorical({δ ′ : p ′ / -→ p Σ (σ)
Proof. See App. E. ■
We also note the following corollary, as it gives an interpretation for the categorical distribution in the efficient conditional_token_generation algorithm.
Corollary 1. For all σ ∈ Σ * , δ ∈ ∆ * ,
P Y ∼p∆ [ϕ σ (Y ) = δ | κ(Y ) ⪰ σ] = -→ p ∆ (δ) -→ p Σ (σ)
1{δ ∈ C(σ)} (15) Thus, we have provided an efficient solution to the prompt boundary problem. We also note that generating from p Σ a character at a time is also a correct solution to the prompt boundary problem; however, it is slower, as it does not benefit from the fact that the generated string is shorter in token space. This is because once the minimally covering token string has been sampled, the method sample_completion will generate a complete sequence more efficiently than the character-at-a-time sample algorithm.
this section cite: []

Section: Experiments
This section investigates our algorithm's running time and accuracy. We use the following setup:
• We use the following publicly available models: Llama- 3.2-1B, Meta-Llama-3.1-8B, DeepSeek-R1-Distill- Llama-8B, and phi-4 (14B) from the transformers library (Wolf et al., 2020). Each model was trained over token strings created from byte-pair encoding (BPE; Sennrich et al. (2016); Gage (1994)).foot_15 • We use the wikitext-103-v1 corpus as a source of character strings; we used the version in the datasets library. Specifically, we use the test portion.
• We use the library 16 with the (Kwon et al.,  2023) backend to perform the efficient, batched evaluation of transformer language models on GPUs. We batchevaluate all sequence extensions. Experiments were run on an L40S GPU with 40GB of memory.
• Our implementation utilizes a trie to efficiently represent all items in each bucket (see App. D), and the bucketbased pruning heuristic ( §3).
To better understand the quality of the approximation our method provides, we perform the following experiments:foot_17
• We measure the approximation error as the average Jensen-Shannon distance (JSD) to a reference model's conditional distribution over the next byte (Fig. 1a). We use a large beam K = 128 as a reference model.
• We evaluate the average surprisal (-log 2 probability) of our model's estimated conditional distribution over the next byte in the corpus. As a baseline, we use the average surprisal (bits/bytes) of the canonical tokenization under the token-level language model (Fig. 1b).
Discussion. As expected, we observe that the speed (bytes/sec) decreases as K increases. We observe an inverse relationship between error (JSD/byte) and speed (bytes/sec): as the processing speed (bytes/sec) decreases, the error also decreases. Notably, this tradeoff is non-linear, with error increasing more sharply at higher processing speeds compared to lower speeds. This trend is evident in all models; in general, error appears to flatten out after K ≥ 8. This indicates diminishing returns in reducing error as K gets large. We hypothesize that this occurs because the language model's probability mass is concentrated on a limited set of tokenizations, which are adequately covered even with smaller beam sizes. We also observe (as expected) that larger models run more slowly; however, this appears to be due to their higher evaluation time rather than the need for larger covers.
We also observe that the average per-byte surprisal is significantly lower under all models than the canonically tokenized baseline. The reasons for this are twofold: (1) Each model assigns non-negligible probability to noncanonical tokenizations of the corpus, which are being thrown out in the baseline estimate, but that is accounted for in our estimate.
(2) The most likely tokenization of the corpus is often noncanonical, which our method is better able to find, as our beam-summing method uses the probabilities assigned to tokenizations. In contrast, the baseline uses only the hard-coded canonical tokenization. Interestingly, increasing K does not appear to significantly decrease the surprisal, which we suspect is because the relatively greedy (K = 2) tokenizations adequately cover it. 18
this section cite: []

Section: Conclusion
We have developed an effective method for ameliorating tensions between tokens and characters faced by engineers and users. We gave theory and algorithms that provide a character-level interface to tokenized language models. We characterized and resolved the prompt boundary problem. We investigated the empirical speed and error rates of our method on two modern language models. The primary limitation of our beam summing method is that it requires a very large beam size K if the language model does not favor a small number of tokenizations. The models that we explored in our experiments concentrate mass on a few tokenizations; thus, we did not require large K to estimate their character-level prefix probabilities accurately.
Prada Corral, Vésteinn Snaebjarnarson, Samuel Kiegeland, and Yahya Emara for their helpful feedback and discussions. JT would like to thank Rycolab for its hospitality during a recent visit. The authors JLG and JT would like to thank Institut des Hautes Études Scientifiques (IHES) for their hospitality while revising this paper. MG was supported by an ETH Zürich Postdoctoral Fellowship. This research was enabled in part by compute resources provided by Mila (mila.quebec).
this section cite: []

Section: References
Ref_id:b0 Title: You should evaluate your language model on marginal likelihood over tokenisations Year: (2021)
Ref_id:b1 Title: Should you marginalize over possible tokenizations? Year: (2023)
Ref_id:b2 Title: A new algorithm for data compression Year: (1994)
Ref_id:b3 Title: The foundations of tokenization: Statistical and computational concerns Year: (2025)
Ref_id:b4 Title: Grammarconstrained decoding for structured NLP tasks without finetuning Year: (2023)
Ref_id:b5 Title: On the proper treatment of tokenization in psycholinguistics Year: (2024)
Ref_id:b6 Title: A probabilistic Earley parser as a psycholinguistic model Year: (2001)
Ref_id:b7 Title: Automata-based constraints for language model decoding Year: (2024)
Ref_id:b8 Title: Subword regularization: Improving neural network translation models with multiple subword candidates Year: (2018)
Ref_id:b9 Title: Efficient memory management for large language model serving with PagedAttention Year: (2023)
Ref_id:b10 Title: Expectation-based syntactic comprehension Year: (2008)
Ref_id:b11 Title: Syntactic and semantic control of large language models via sequential Monte Carlo Year: (2025)
Ref_id:b12 Title: The art of prompt design: Prompt boundaries and token healing Year: (2023)
Ref_id:b13 Title: Guidance Year: (2023)
Ref_id:b14 Title: Recurrent neural network based language model Year: (2010)
Ref_id:b15 Title: Leading whitespaces of language models' subword vocabulary poses a confound for calculating word probabilities Year: (2024)
Ref_id:b16 Title: Understanding and mitigating tokenization bias in language models Year: (2024)
Ref_id:b17 Title: How to compute the probability of a word Year: (2024)
Ref_id:b18 Title: Reliable code generation from pre-trained language models Year: (2022)
Ref_id:b19 Title: Simple and effective subword regularization Year: (2020)
Ref_id:b20 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b21 Title: PICARD: parsing incrementally for constrained auto-regressive decoding from language models Year: (2021)
Ref_id:b22 Title: Neural machine translation of rare words with subword units Year: (2016)
Ref_id:b23 Title: A mathematical theory of communication. The Bell System Technical Year: (1948)
Ref_id:b24 Title: Attention is all you need Year: (2017)
Ref_id:b25 Title: Efficient guided generation for large language models Year: (2023)
Ref_id:b26 Title: Transformers: State-of-theart natural language processing Year: (2020)
Ref_id:b27 Title: Llama-3.2-1B Year: ()
Ref_id:b28 Title:  Year: (1738)
Ref_id:b29 Title: Table 1: Surprisal, JSD, and speed for the token-healing baseline across models using the same experimental settings as Fig Year: ()
