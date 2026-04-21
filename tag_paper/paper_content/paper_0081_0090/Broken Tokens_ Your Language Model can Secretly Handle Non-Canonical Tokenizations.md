Title: Broken Tokens? Your Language Model can Secretly Handle No n -Ca noni cal Tokenizations
Abstract: Modern tokenizers employ deterministic algorithms to map text into a single "canonical" token sequence, yet the same string can be encoded as many noncanonical tokenizations using the tokenizer vocabulary. In this work, we investigate the robustness of LMs to text encoded with non-canonical tokenizations entirely unseen during training. Surprisingly, when evaluated across 20 benchmarks, we find that instruction-tuned models retain up to 93.4% of their original performance when given a randomly sampled tokenization, and 90.8% with character-level tokenization. We see that overall stronger models tend to be more robust, and robustness diminishes as the tokenization departs farther from the canonical form. Motivated by these results, we then identify settings where non-canonical tokenization schemes can improve performance, finding that character-level segmentation improves string manipulation and code understanding tasks by up to +14%, and right-aligned digit grouping enhances large-number arithmetic by +33%. Finally, we investigate the source of this robustness, finding that it arises in the instructiontuning phase. We show that while both base and post-trained models grasp the semantics of non-canonical tokenizations (perceiving them as containing misspellings), base models try to mimic the imagined mistakes and degenerate into nonsensical output, while post-trained models are committed to fluent responses. Overall, our findings suggest that models are less tied to their tokenizer than previously believed, and demonstrate the promise of intervening on tokenization at inference time to boost performance. 1 QWEN-2.5-7B-INSTRUCT LLAMA-3.1-8B-INSTRUCT OLMO-2-7B-INSTRUCT Benchmark Canon Rand ∆ Char ∆ Canon Rand ∆ Char ∆ Canon Rand ∆ Char ∆ Multiple choice (MC)

Section: Introduction
Tokenizers segment text into a sequence of discrete tokens in the language model's (LM) vocabulary. Most of today's LMs use deterministic subword tokenization, which produces a single canonical token sequence for a given piece of text, and further, for each whitespace-delimited word. One commonly discussed limitation of this approach is that, by mapping byte strings to symbolic token IDs, the orthographic makeup of tokens is obscured to the LM [52,18]. This can be especially harmful for LM understanding of numbers [47,64,61] and morphologically rich languages [2,25], and has motivated efforts to model text directly at the byte level [12,70,68,63,72,46,49,1,39].
To shed more light on this perceived limitation, in this work we study whether LMs can adapt at inference time, without any additional training, to a different tokenization scheme than the one they were trained with. While the tokenizer deterministically outputs a canonical tokenization of any text into tokens (usually by applying an ordered list of merge rules), non-canonical tokenizations of the same text using the same vocabulary are generally possible (see example in Figure 1). Here, we evaluate how LMs trained with deterministic tokenizers behave when given non-canonical tokenizations of text. Surprisingly, we find that instruction-tuned LMs across many model families are extremely robust to non-canonical tokenizations ( §2). For example, when evaluated across 20 benchmarks, QWEN-2.5-7B-INSTRUCT retains 93.4% of its original performance when presented with a random non-canonical tokenization, and 90.8% when presented with character-level tokens (see Figure 1). Thus, far from not understanding the makeup of their tokens, LMs are able to compose token sequences in entirely new ways at inference time [33].
This leads to an intriguing possibility: if LMs can process non-canonical tokenizations, can we use different tokenization schemes at inference time to improve performance? For instance, prior work has found that better segmentation of large numbers can improve accuracy on arithmetic [61,56]. Indeed, we identify several settings where non-canonical tokenization schemes improve performance for LLAMA-3.1-INSTRUCT ( §3): character-level tokenization brings up to +14% improvement on string manipulation and code understanding tasks, perhaps by granting LMs more direct access to orthographic cues. Meanwhile, right-aligned digit groups, which provide a consistent grouping of digits by powers of a thousand, improves arithmetic on large numbers by +33%. These performance gains are achieved without any model finetuning, pointing to the promise of tokenization as a means of inference-time control.
Finally, we investigate the origins of model robustness to non-canonical tokenizations ( §4). Across multiple model families, we find that pretrained-only LMs consistently fail to produce fluent continuations given non-canonically tokenized context. By studying models at different stages of post-training, we identify that robustness arises during the supervised instruction-tuning (SFT) phase ( §4.1). We then ablate differences between pretraining and SFT procedures and find that the separation of the instruction and response as distinct turns of conversation is key ( §4.2). From here, we provide evidence for a plausible explanation: while both base and post-trained models grasp the semantics of non-canonical tokenizations, they also perceive them as containing misspellings ( §4.2). Base models attempt to mimic the imagined mistakes and degenerate into nonsense, whereas post-trained models are not bound by the style of the instruction and thus able to produce fluent responses.
Overall, despite being trained with deterministic tokenization, instruction-tuned LMs readily accommodate new tokenizations at inference time, suggesting that LMs are less constrained by their tokenizer than previously believed [45]. Moreover, in settings where different representations of text are beneficial, we can intervene on tokenization at inference time for performance gains. We hope our work sheds new light on the discussion of strengths and limitations of tokenization, and points to the possibility of dynamically finding the optimal representation of text after pretraining.
Table 1: Evaluated across many benchmarks, models are surprisingly robust to non-canonical tokenizations of the context. We show the absolute drop in performance when given a randomly sampled non-canonical tokenization (Rand ∆) and character-level tokenization (Char ∆), relative to the canonical (Canon) tokenization. We also summarize the model's ability to retain performance across benchmarks and tokenization strategies (bottom).
Overall Avg MC Retention (%) 92.4±5.97 89.2±9.33 85.6±6.86 76.8±7.99 71.2±14.7 59.2±17.8 Avg SA Retention (%) 94.6±3.97 92.7±5.35 90.3±6.78 82.7±9.65 75.4±15.1 65.4±17.4 Avg Overall Retention (%) 93.4±5.15 90.8±7.81 87.7±7.05 79.4±9.05 73.1±14.7 62.0±17.5
this section cite: ['b51', 'b17', 'b46', 'b64', 'b61', 'b1', 'b24', 'b11', 'b70', 'b68', 'b63', 'b72', 'b45', 'b48', 'b0', 'b38', 'b32', 'b61', 'b56', 'b44']

Section: Language Models are Robust to Non-Canonical Tokenizations
In our main experiments, we evaluate the robustness of LMs to non-canonical tokenizations by comparing their performance on downstream tasks when given different tokenizations of the input.
this section cite: []

Section: Background
Most LMs today, and all the models we study, use the Byte-Pair Encoding (BPE) [58] algorithm for tokenization. The BPE tokenizer is learned by splitting a corpus of text into bytes, which form the initial vocabulary, then iteratively merging the most frequent pair of tokens into a new token that is added to the vocabulary. To encode a new text, it is split into bytes, and the learned merges are applied in the same order. As a result, a BPE tokenizer always produces the same token sequence for the same text. Further, because BPE tokens do not cross whitespace boundaries, the same whitespace-delimited word is always represented with the same token or token sequence.
A natural observation is that given a tokenizer vocabulary, there exist many token sequences that decode to the same text. For instance, cat could be tokenized as In general, the number of non-canonical tokenizations grows exponentially with the length of the text. Many previous works have argued that the probability of a string should be calculated as the sum of probabilities of all possible tokenizations [7,9,20]. However, less attention has been paid to how non-canonical tokenizations affect LMs in generative settings.
this section cite: ['b58', 'b6', 'b8', 'b19']

Section: Setup
We consider two non-canonical tokenization schemes. (1) Random tokenization produces a tokenization (uniformly at random) from the set of tokenizations more granular than the canonical one. This can be achieved by recursively splitting individual tokens into a valid pair of tokens, similarly to We achieve variation in tokenization length using different values of p in BPE-dropout, and group tokenizations into buckets based on how many times longer it is than the canonical tokenization.
[60]; the pseudocode and a proof of correctness is provided in Appendix A. (2) Character-level tokenization decomposes the string into character tokens, i.e., using no subword token from the vocabulary. For text containing only English letters and punctuation (where each character is exactly one byte), this produces the most granular possible tokenization.
We consider three models, LLAMA-3.1-8B-INSTRUCT [43], OLMO2-7B-INSTRUCT [48], and QWEN-2.5-7B-INSTRUCT [53], which we evaluate on 20 benchmarks shown in Table 1. Please see §B.1 for further description of the datasets and evaluation setup.
this section cite: ['b42', 'b47', 'b53']

Section: Results
Shown in Table 1, while random tokenization consistently leads to worse performance compared to the canonical tokenization, the effect is small. On average across benchmarks, QWEN-2.5 retains 93.4% of its performance when given random tokenization, followed by LLAMA-3.1 at 87.7% and OLMO-2 at 73.1%. The performance drops further with character-level tokenization, with retention of 90.8%, 79.4%, and 62.0% for the three models, respectively. This ranking of models in terms of retention is consistent with their ranking in absolute accuracy (under canonical tokenization), suggesting that stronger models are generally more robust to non-canonical tokenization strategies.
We also observe that all models retain performance better on short answer (SA) benchmarks (where the model generates an output in free-form) compared to multiple choice (MC) benchmarks (where the model is instructed to directly output the correct answer choice). In addition, LMs consistently produce correct token sequences even when conditioning on non-canonical tokenizations. We hypothesize that, in the SA setting, models benefit from eventually conditioning on recent correctlytokenized context.
this section cite: []

Section: Analysis: How does granularity of the tokenization affect robustness?
We next study whether tokenization fine-grainedness correlates in general with model robustness. We measure the fine-grainedness of a given non-canonical tokenization by how many times longer it is (in tokens) than the canonical tokenization, which we call the "length ratio." Finer-grained tokenizations have higher ratios, while coarser ones have ratios closer to 1. We produce tokenizations with diverse length ratios by applying BPE dropout [52] with p ∈ [0.1, 0.2, ..., 0.9], which controls the probability with which each merge is dropped. (High p leads to finer-grained segmentations, and p = 0.0 corresponds to conventional BPE.)
Figure 2 shows the relationship between the length ratio and the average performance retention relative to canonical tokenization, with finer-grained tokenization generally leading to worse performance. When performance retention is averaged across tasks, the negative correlation is statistically significant under Kendall's τ with p = 0.003. 3 Can non-canonical tokenizations improve model performance?
If LMs can process non-canonical tokenizations, this points to the exciting possibility that tokenization schemes can be modified completely at inference-time. This would be useful if, in certain settings, there exists a better representation of text than what the tokenizer produces. In this section, we develop a suite of tasks that intuitively require understanding of the orthography of the text, and show that LLAMA-3.1-8B-INSTRUCT performs better under non-canonical tokenization schemes.
this section cite: ['b51']

Section: Tasks
Please see Table 2 for an example question in each task and Table B.2 for further details on dataset construction. For all tasks except Arithmetic, we use character-level tokenization.
this section cite: []

Section: Counting Characters
This task asks the model to count the number of occurrences of the most common letter in 5-10 character tokens in LLAMA-3.1's vocabulary, and contains 1001 samples.
Acronyms This task asks models to generate a list of words whose first letters form a given acronym. We construct 3594 5-letter acronyms by sampling each letter uniformly at random from the alphabet.
this section cite: []

Section: Code Description
For a more real-world application, we construct a task where the model is given a code snippet and asked to identify the function of the code in natural language from four MC options. The setup is inspired by the Codeline Description task from BIG-Bench [5], but to increase the difficulty we use more complex code snippets and corresponding natural language descriptions from XLCoST [74]. To collect incorrect answers, we sample three other code descriptions from the dataset. This task contains 4800 samples across 6 programming languages.
Arithmetic Prior work has suggested that arithmetic is difficult for LMs in part due to poor segmentation of digits [47,64]. We curate a simple arithmetic dataset by constructing addition and subtraction tasks for 10 digit numbers. Here, we use a different segmentation strategy. The LLAMA-3.1 tokenizer segments numbers into groups of three left-to-right (e.g., 1000000 is encoded as ["100", "000", "0"]), due to the pretokenization regular expression looking for matches greedily from the left. Inspired by [61], we instead segment digits into groups of three right-to-left (e.g., ["1", "000", "000"]). This task contains 1000 addition and subtraction questions in total.
this section cite: ['b4', 'b74', 'b46', 'b64', 'b61']

Section: Results
Shown in Table 3, in all the tasks we construct, the non-canonical tokenization strategy leads to substantially better performance compared to the canonical tokenization. In particular, we observe a +14.3% improvement on code description and +33.7% on arithmetic. Our results show that the tokenization scheme used in training is not necessarily the optimal one at inference-time, and Table 3: On several tasks, LLAMA-3.1-8B-INSTRUCT achieves better performance when using a non-canonical tokenization scheme. For the first four tasks, the input is tokenized at the character level; for Arithmetic, we segment digits into groups of three digits from right to left (instead of the usual left to right). On all tasks, we observe a large performance improvement from using the alternative tokenization scheme.
Task Canonical Alternative ∆ Counting Characters 66.5 73.5 +6.99 Acronyms 49.7 57.4 +7.74 Code Description 68.6 82.9 +14.3 Arithmetic 36.5 70.2 +33.7
replacing them with intuitively meaningful tokenizations can bring substantial performance gains. We leave automatically identifying the optimal tokenization as a promising direction for future work.
this section cite: []

Section: Investigating the Source of Robustness
Thus far, our experiments have used post-trained "instruct" models. In this section, we find that pretrained-only models are actually unable to produce fluent continuations of unusually tokenized context ( §4.1), perform ablations to identify the conditions enabling robustness ( §4.2), and finally provide support for an explanation of why generative robustness arises during post-training ( §4.3).
this section cite: []

Section: When does robustness appear in model training?
We first quantify the robustness of models at different stages of the model development pipeline by using the OLMO2 and TULU3 [36] model families which include the base, SFT, DPO, and final instruct models. For simplicity, we focus on AlpacaEval and use character-level tokenization. For base models, we construct the prompt by placing the instruction in a question-answer template (Question: {instruction} \nAnswer:). We define three simple measures of generation quality.
this section cite: ['b35']

Section: Spelling
We measure the proportion of (whitespace-delimited) words in the generation that can be found in a collection of the top 10,000 most common English words. 2Grammaticality We use LanguageTool's grammar checker 3 to count the number of grammatical mistakes, which we normalize by the number of words in the generation and subtract from 1 to produce a grammaticality score where higher is better.
Win rate To measure overall generation quality, we use alpaca eval gpt4 as an LM judge in the AlpacaEval framework and report the win rate of the generation given alternative against canonical tokenizations of the context. Unlike the previous two metrics, this measures not only the quality of the generation but also its relevance to the context.
Shown in Figure 3, the base models of OLMO2 and LLAMA-3.1 are both unable to produce sensible output conditioned on character-level tokenizations of context, scoring at best 0.317 on spelling and 0.260 on grammaticality. Qualitatively, generations are extremely difficult to parse and often involve odd character substitutions and repetitions (e.g., Yoou, haviin). Despite this, they sometimes reflect an understanding of the prompt. Consider, for example,
Question: I like to host guests at my home from time to time [...] Can you give me a recipe for Canjeero? Answer: I aam glade tio hear tio hear tio hear tio hear that yoou enjoy haviin gauests at yoour hoome an tio keeep tio keeep tio keeep In contrast, the post-trained models are more robust across all three metrics, with much of the improvement coming from the SFT stage alone.
this section cite: []

Section: Why do instruction-tuned models become robust?
We first replicate the finding from §4.1 that SFT yields robustness to non-canonical tokenizations by finetuning the LLAMA-3.2-1B base model on the TULU 3 SFT PERSONAS INSTRUCTION FOLLOWING dataset. Then, we perform the following interventions on the SFT training data and procedure to shed light on the possible source.
Gradient over full sequence SFT on instruction-response pairs conventionally uses a loss mask over the instruction tokens, so that only the response tokens contribute to the loss. We remove this loss mask and instead compute gradients over the entire instruction and response.
Question/answer template We replace the chat template with a simple question-answer template, Question: {instruction} Answer: {response}, both for training and evaluation.
Removing the chat template We remove the chat template by concatenating the instruction and response without any special formatting. In evaluation, we again provide the instruction alone.
Removing the instruction After SFT training, the LM's goal is no longer to continue a given text prefix, but rather to generate a response to the given instruction. To ablate the nature of the data itself, we take only the responses from the SFT data, and randomly split each into a new "prompt" and "response," which we format with the SFT template. 4 At test time, we similarly provide an incomplete response within "instruction" tags. Since the purpose of the passage is generally inferrable from the first few words of the gold response ("Sure, here's a recipe for Kubdari..."), we are able to evaluate generated responses under the same AlpacaEval framework.
Our results are shown in Figure 4. We replicate the finding that SFT (No ablation) leads the model to be able to handle non-canonical tokenizations. This persists when computing gradients over the entire instruction and response (Full gradient) so that the training procedure matches regular pretraining. Replacing the original chat template with a simple question-answer template (QA template) also maintains model robustness. However, the usage of a template is crucial -when directly concatenating the instruction and response (Removing chat template), the model fails to produce coherent generations, with the spelling score dropping from 0.786 in the no ablation setting to 0.0698. Inserting the chat template into pretraining-style data (Removing the instruction) also does not yield robustness, with a spelling and grammaticality scores remaining low at 0.181 and 0.158, Table 4: Both base and instruct models from the LLAMA-3.1-8B family recognize words represented with non-canonical tokenizations (performing well on Word Repeat), but incorrectly perceive that there are misspellings (performing at random on Identifying Misspellings). Word Repeat Identifying Misspellings Base model 90.8 48.2 Instruct model 92.0 55.8
respectively. Overall, these findings suggest that in order for the LM to generate fluent continuations given non-canonical tokenizations, the context and expected continuation need to represent separate turns of dialogue, and additionally, be demarcated with a special template.
this section cite: []

Section: Disentangling understanding from generation
One plausible explanation for our findings thus far is that both base and instruction-tuned models grasp the semantics of non-canonical tokenizations, yet falsely perceive them as containing misspellings. While base LMs attempt to faithfully continue these mistakes and degenerate into nonsensical output, instruction-tuned models are trained to provide fluent responses regardless of the instruction, leading to the results observed in §4.1. To test this hypothesis, we construct two simple tests:
1. Word Repeat: To determine if a model perceives the meaning of a word with non-canonical tokenization, we prompt the model to repeat a given word (while correcting any typos).
this section cite: []

Section: Identifying Misspellings:
To determine if a model perceives a misspelling, we ask it to identify the word with a misspelling among two options: a (correctly tokenized) misspelling of a word and an non-canonical tokenization of that word (correctly spelled).
Results are shown in Table 4. Consistent with our hypothesis, we find that both the base and instruct models from the LLAMA-3.1-8B family score highly (> 90%) on Word Repeat. This means that the base model, despite its poor performance in §4.1, actually recognizes the correct form of noncanonical tokenizations as well as its post-trained counterpart. In addition, both models perform at random when asked to distinguish non-canonical tokenization from true misspellings. In other words, the instruct model produces fluent responses ( §2) while interpreting the instruction as heavily misspelled! While instruct models evidently overcome this, the base model likely attempts to mimic the (perceived) idiosyncratic surface form, thus producing nonsensical (yet sometimes relevant) outputs.
this section cite: []

Section: Related Work
The extent to which LMs are limited by their tokenization is a topic of much debate, with the story evolving as LMs become larger and more capable.
Character-level understanding in tokenizer-based models It is commonly argued that tokenization obscures orthographic information about tokens from the LM, leading to unexpected failures [18,8,67]. As a result, there have been many efforts towards linguistically-informed tokenization that make derivational, compound, and morphological boundaries within words explicit [35,25,26,71,4]. Similarly, BPE-dropout [52] and related methods [60] introduce variation in training in how a given string is tokenized in order to make models more robust to rare, misspelled, or unseen words.
However, there is other evidence that LMs naturally overcome these limitations. For instance, token embeddings have been found to robustly encode character-level information, especially in larger models [34,27]. This may be because word variants that do not share tokens in common (consider e.g., [ dictionary] and [ diction, aries], as tokenized by GPT-2) incentivize the model to learn spelling as a general solution to understanding their relations [34]. Other works argue that LMs maintain an implicit vocabulary, and can compose arbitrary token sequences (including non-canonical ones) into useful higher-level representations [19,33]. Even in domains like biomedical text where terms are highly agglutinative, using tokenizers that segment on meaningful components does not lead to improved models [30]. Recent works have even found that coarser superword tokenization [40,57], which capture common word sequences in a single token, preserve character-level understanding while providing benefits in compression and downstream performance.
Our work informs this conversation by showing that LMs can effectively leverage character-level knowledge of their tokens and glean potential benefits of improved representation at inference time.
Partial token problem A related but distinct problem is the partial token problem (also known as tokenization bias or the prompt boundary problem) where the prompt ends with the prefix of a valid token, causing the model to assign unexpectedly low probability to the completion of that token. Many works have found that this continues to compromise a serious failure mode for frontier LMs [69,51,41,66,22]. In particular, DEEPSEEK V3 [15] aims to improve robustness to partial punctuation tokens by randomly splitting some proportion of multi-punctuation tokens into smaller tokens during training, though they do not present experiments with this ablation. We note that these results are not inconsistent with ours -together, they suggest that while models are very unlikely to generate non-canonical tokenizations, they can nonetheless understand them in the context history.
Non-canonical tokenizations It has long been recognized that there are many possible ways to segment a string into tokens with a fixed vocabulary [10], which in principle should be considered in the calculation of a string's likelihood [7,9,20]. Previous work has briefly touched on non-canonical tokenization in the context of self-supervised evaluation [28] and defense against adversarial attacks [29]. In contemporaneous work, Geh et al. [21] also show that non-canonical tokenizations can be constructed adversarially to trigger unsafe completions. In contrast, we provide a more systematic study of LM robustness using benchmark evaluations and additionally study its source.
Somewhat relatedly, other works have provided algorithms for sampling at the character-or byte-level from tokenizer-based LMs [50,65,3,22]. Together, these directions suggest that despite being trained with one deterministic tokenization scheme, LMs can both condition on and produce token sequences over a different (sub)vocabulary.
this section cite: ['b17', 'b7', 'b67', 'b34', 'b24', 'b25', 'b71', 'b3', 'b51', 'b60', 'b33', 'b26', 'b33', 'b18', 'b32', 'b29', 'b39', 'b57', 'b69', 'b50', 'b40', 'b66', 'b21', 'b14', 'b9', 'b6', 'b8', 'b19', 'b27', 'b28', 'b20', 'b49', 'b65', 'b2', 'b21']

Section: Conclusion
Despite being trained with deterministic tokenization algorithms, we show that instruction-tuned language models are surprisingly robust to token sequences not seen in training. In certain domains, such as arithmetic or code, more intuitively meaningful tokenizations can even be swapped-in at inference time for improved performance. We analyze the source of this robustness, and find that while the base and instruct models both perceive the semantics of non-canonical tokenizations, only instruct models are capable of providing fluent continuations. Our work demonstrates a way in which LMs are not necessarily tied to tokenizer they were trained with, and highlights the potential of finding more optimal representations of text after pretraining.
this section cite: []

Section: References
Ref_id:b0 Title: MAGNET: Improving the multilingual fairness of language models with adaptive gradientbased tokenization Year: (2024)
Ref_id:b1 Title: Why do language models perform worse for morphologically complex languages? Year: (2025)
Ref_id:b2 Title: Token alignment via character matching for subword completion Year: (2024-08)
Ref_id:b3 Title: BPE-knockout: Pruning pre-existing BPE tokenisers with backwards-compatible morphological semi-supervision Year: (2024-06)
Ref_id:b4 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2023)
Ref_id:b5 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b6 Title: You should evaluate your language model on marginal likelihood over tokenisations Year: (2021-11)
Ref_id:b7 Title: Tokenization falling short: On subword robustness in large language models Year: (2024-11)
Ref_id:b8 Title: Should you marginalize over possible tokenizations? Year: (2023-07)
Ref_id:b9 Title: Emerging trends: Subwords, seriously? Year: (2020)
Ref_id:b10 Title: BoolQ: Exploring the surprising difficulty of natural yes/no questions Year: (2019-06)
Ref_id:b11 Title: Canine: Pre-training an efficient tokenizationfree encoder for language representation Year: (2022)
Ref_id:b12 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b13 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b14 Title: Deepseek-v3 technical report Year: (2025)
Ref_id:b15 Title: DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs Year: (2019-06)
Ref_id:b16 Title: Alpacafarm: A simulation framework for methods that learn from human feedback Year: (2023)
Ref_id:b17 Title: CUTE: Measuring LLMs' understanding of their tokens Year: (2024-11)
Ref_id:b18 Title: Token erasure as a footprint of implicit vocabulary items in LLMs Year: (2024-11)
Ref_id:b19 Title: Where is the signal in tokenization space? Year: (2024-11)
Ref_id:b20 Title: Adversarial tokenization Year: (2025)
Ref_id:b21 Title: Sampling from your language model one byte at a time Year: (2025)
Ref_id:b22 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b23 Title: Measuring mathematical problem solving with the MATH dataset Year: (2021)
Ref_id:b24 Title: Superbizarre is not superb: Derivational morphology improves BERT's interpretation of complex words Year: (2021-08)
Ref_id:b25 Title: An embarrassingly simple method to mitigate undesirable properties of pretrained language model tokenizers Year: (2022-05)
Ref_id:b26 Title: Models in a spelling bee: Language models implicitly learn the character composition of tokens Year: (2022-07)
Ref_id:b27 Title: Bring your own data! self-supervised evaluation for large language models Year: (2023)
Ref_id:b28 Title: Baseline defenses for adversarial attacks against aligned language models Year: (2023)
Ref_id:b29 Title: Biomedical language models are robust to suboptimal tokenization Year: (2023-07)
Ref_id:b30 Title: Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017)
Ref_id:b31 Title: 200,000+ jeopardy! questions Year: (2019)
Ref_id:b32 Title: From tokens to words: On the inner lexicon of LLMs Year: (2025)
Ref_id:b33 Title: What do tokens know about their characters and how do they know it? Year: (2022-07)
Ref_id:b34 Title: Getting the ##life out of living: How adequate are word-pieces for modelling complex morphology? Year: (2020-07)
Ref_id:b35 Title: Tulu 3: Pushing frontiers in open language model post-training Year: (2025)
Ref_id:b36 Title: The winograd schema challenge Year: (2012)
Ref_id:b37 Title: CMMLU: Measuring massive multitask language understanding in Chinese Year: (2024-08)
Ref_id:b38 Title: MYTE: Morphology-driven byte encoding for better and fairer multilingual language modeling Year: (2024-08)
Ref_id:b39 Title: Space travel for language models Year: (2025)
Ref_id:b40 Title: The art of prompt design: Prompt boundaries and token healing Year: (2023)
Ref_id:b41 Title: Tofu: A task of fictitious unlearning for llms Year: (2024)
Ref_id:b42 Title: The llama 3 herd of models Year: (2024)
Ref_id:b43 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b44 Title: Zero-shot tokenizer transfer Year: (2024)
Ref_id:b45 Title: Efficient transformers with dynamic token pooling Year: (2023-07)
Ref_id:b46 Title: Investigating the limitations of transformers with simple arithmetic tasks Year: (2021)
Ref_id:b47 Title:  Year: (2024)
Ref_id:b48 Title: Byte latent transformer: Patches scale better than tokens Year: (2024)
Ref_id:b49 Title: Understanding and mitigating tokenization bias in language models Year: (2024)
Ref_id:b50 Title: Exact byte-level probabilities from tokenized language models for fim-tasks and model ensembles Year: (2025)
Ref_id:b51 Title: BPE-dropout: Simple and effective subword regularization Year: (2020-07)
Ref_id:b52 Title: URL Year: ()
Ref_id:b53 Title: Qwen2.5 technical report Year: (2025)
Ref_id:b54 Title: Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning Year: (2011-03)
Ref_id:b55 Title: Winogrande: an adversarial winograd schema challenge at scale Year: (2021-08)
Ref_id:b56 Title: Improving consistency in LLM inference using probabilistic tokenization Year: (2025-04)
Ref_id:b57 Title: Boundless byte pair encoding: Breaking the pre-tokenization barrier Year: (2025)
Ref_id:b58 Title: Neural machine translation of rare words with subword units Year: (2016-08)
Ref_id:b59 Title: Language models are multilingual chain-of-thought reasoners Year: (2022)
Ref_id:b60 Title: StochasTok: Improving fine-grained subword understanding in LLMs Year: (2025)
Ref_id:b61 Title: Tokenization counts: the impact of tokenization on arithmetic in frontier llms Year: (2024)
Ref_id:b62 Title: CommonsenseQA: A question answering challenge targeting commonsense knowledge Year: (2019-06)
Ref_id:b63 Title: Charformer: Fast character transformers via gradient-based subword tokenization Year: (2022)
Ref_id:b64 Title: Representing numbers in NLP: a survey and a vision Year: (2021-06)
Ref_id:b65 Title: From language models over tokens to language models over characters Year: (2024)
Ref_id:b66 Title: From language models over tokens to language models over characters Year: (2025)
Ref_id:b67 Title: Tokenization matters! degrading large language models through challenging their tokenization Year: (2024)
Ref_id:b68 Title: Mambabyte: Token-free selective state space model Year: (2024)
Ref_id:b69 Title: Are you going to finish that? a practical study of the partial token problem Year: (2026)
Ref_id:b70 Title: ByT5: Towards a token-free future with pre-trained byte-to-byte models Year: (2022)
Ref_id:b71 Title: Incorporating context into subword vocabularies Year: (2023-05)
Ref_id:b72 Title: MEGABYTE: Predicting million-byte sequences with multiscale transformers Year: (2023)
Ref_id:b73 Title: HellaSwag: Can a machine really finish your sentence? Year: (2019-07)
Ref_id:b74 Title: Xlcost: A benchmark dataset for cross-lingual code intelligence Year: (2022)
