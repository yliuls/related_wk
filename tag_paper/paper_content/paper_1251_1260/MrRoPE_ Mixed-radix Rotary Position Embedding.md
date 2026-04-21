Title: MRROPE: MIXED-RADIX ROTARY POSITION EMBED-DING
Abstract: Rotary Position Embedding (RoPE)-extension refers to modifying or generalizing the Rotary Position Embedding scheme to handle longer sequences than those encountered during pre-training. However, current extension strategies are highly diverse and lack a unified theoretical foundation. In this paper, we propose Mr-RoPE (Mixed-radix RoPE), a generalized encoding formulation based on a radix system conversion perspective, which elegantly unifies various RoPE-extension approaches as distinct radix conversion strategies. Based on this theory, we introduce two training-free extensions, MrRoPE-Uni and MrRoPE-Pro, which leverage uniform and progressive radix conversion strategies, respectively, to achieve "train short, test long" generalization. Without fine-tuning, MrRoPE-Pro sustains over 85% recall in the 128K-context Needle-in-a-Haystack test and achieves more than double YaRN's accuracy on Infinite-Bench retrieval and dialogue subsets. Theoretical analysis confirms that MrRoPE-Pro effectively raises the upper bound of RoPE's attainable encoding length, which further validates the reliability and utility of our theory and methodology.

Section: INTRODUCTION
Effective understanding of long-text contexts is a cornerstone for advanced NLP tasks (Liu et al., 2023a). For Large Language Models (LLMs), this capability is fundamentally underpinned by the Rotary Position Embedding (RoPE) (Su et al., 2024), which provides a robust foundation for modeling long-range dependencies (Grattafiori et al., 2024;Yang et al., 2025b). In RoPE, the positional information of tokens is encoded through rotation angles, where each embedding dimension is associated with a distinct rotational frequency. Higher dimensions rotate more slowly, which implies that during training, these dimensions often fail to experience a complete cycle. As a result, once the context length exceeds the training window, those high-dimensional features encounter unseen rotation angles, leading to generalization failure (Liu et al., 2023b). To address this issue, early studies proposed continuing training with a larger base frequency, thereby compressing the rotation angles of higher dimensions into the previously observed range and mitigating the occurrence of outof-domain (OOD) positions (Kazemnejad et al., 2023). However, this strategy requires substantial additional computation (e.g., 57740 GPU hours for Llama2-70B to 32K context window), making larger context extension prohibitively expensive (Touvron et al., 2023;Xiong et al., 2023).
Consequently, recent research has shifted toward training-free approaches to extend the context window of LLMs. Chen et al. (2023) first proposed Position Interpolation (PI), which uniformly scales down the rotation angles across all dimensions to modify RoPE, though the method still requires light fine-tuning on a small amount of data. Alternatively, NTK-aware Interpolation introduced by bloc97 (2023) reduces the distortion in low-dimensional spaces, thereby enabling a substantial expansion of the context window without any extra training. Building upon these insights, Peng et al. (2023) introduced YaRN, an NTK-by-part strategy that applies extrapolation in low dimensions and interpolation in high dimensions, ultimately achieving efficient long-context extension.
Despite the effectiveness of YaRN, we identify a potential limitation in YaRN's approach: Is linear interpolation used for middle dimensions truly the best choice for maximizing model performance on long sequences? To answer this question, we first introduce MrRoPE (Mixed-radix Rotary Position Embedding), a unified theoretical framework to generalize existing RoPE-based extension methods (PI, NTK, YaRN ...) and reflect it to a specific radix conversion approach. Guided by this framework, we identify that YaRN's conservative strategy in lower frequency dimensions may excessively disrupt the high-frequency information inherent in the original RoPE encoding. Hence, we further propose two novel RoPE extension schemes, MrRoPE-Uni and MrRoPE-Pro, which exhibit fundamentally different radix conversion behaviors from YaRN. Their design allows us to systematically compare the efficacy of different radix conversion strategies.
By employing a progressive radix conversion to RoPE, MrRoPE-Pro achieves substantial and consistent gains across both synthetic and real-world long-context evaluations. On the RULER benchmark, it maintains stable accuracy up to 128K tokens, whereas YaRN experiences a sharp degradation beyond 64K. On Infinite-Bench, MrRoPE-Pro not only surpasses YaRN by large margins but also approaches or even exceeds the performance of specialized fine-tuned long-context models-all without additional training. Complementing these empirical findings, our theoretical analysis demonstrates that MrRoPE-Pro significantly enlarges the effective context window predicted by RoPE Bound Theory and stabilizes attention score distributions in intermediate dimensions. Together, these results establish MrRoPE-Pro as a theoretically grounded and empirically robust method for extending the context window of RoPE-based LLMs.
this section cite: ['b22', 'b6', 'b12', 'b26', 'b29', 'b4', 'b17']

Section: ROPE AND RADIX THEORY

this section cite: []

Section: PRELIMINARY: ROTARY POSITION EMBEDDINGS (ROPE)
Our research is grounded in the RoPE introduced by Su et al. (2024), a position embedding scheme that underpins numerous state-of-the-art LLMs. Consider one specific attention head in a given layer of a Transformer-based model, the set of its hidden neurons is denoted by D. Given an input sequence of vectors x 1 , . . . , x L ∈ R |D| , it's essential to maintain the relative position information between each (x m , x n ) in the result of the self-attention calculation. To address this issue, RoPE first uses a rotation operation that converts vector x m into its corresponding query vector q m and key vector k m (Vaswani et al., 2017):
{q, k} m = f {q,k} (x m , m) = e imθ W {q,k} x m ,(1)
where θ = diag(θ 1 , ..., θ |D|/2 ) is a diagonal matrix with θ i = b -2(i-1)/|D| . b is a predefined value, which is typically set to 10000. In this way, RoPE establishes an injective relation between a token's absolute position and the rotation steps of its embedding vector. Crucially, in the subsequent self-attention computation, the interaction between queries and keys inherently encodes relative positional information as follows:
⟨q m , k n ⟩ = ⟨f q (x m , m) , f k (x n , n)⟩ R = Re x * m W * q W k x n e iθ(m-n) ,(2)
where * denotes the conjugate transpose operation. For this reason, the attention score depends solely on the relative position m -n rather than the absolute position, which facilitates LLMs in more effectively capturing the relationships between tokens. In the implementation stage, e imθ in Eq. 1 can be further expressed as a block diagonal matrix as follows:
e imθ = diag(A 1 , • • • , A |D| 2 ); A j = cos mθ j -sin mθ j sin mθ j cos mθ j .(3)
In conclusion, RoPE treats q or k vector as a |D| 2 -dimensional complex vector, and applies blockwise rotations (i.e., each component will be rotated in its corresponding frequency domain where higher dimension gets slower frequency).
this section cite: ['b22', 'b27']

Section: RETHINKING ROPE UNDER THE RADIX THEORY
As we showed above, the key idea of RoPE is to divide the whole q m (or k m ) vector into D r parts (i.e., D r = |D|/2), and rotate each sub-vector by a corresponding angle mθ j (i.e., j = 1, 2, • • • , D r ). Formally, given a non-rotated complex vector W {q,k} x m ∈ C Dr , its onedimensional position m is expanded to a D r -dimensional vector mθ ∈ R Dr by RoPE. The rotation angle mθ j for j-th part can be calculated as follows:
mθ j = (m • b -(j-1) Dr ) mod 2π.(4)
Most notably, Eq. 4 is strikingly similar to what radix (base) conversion does: representing a number as a sequence of digits. Specifically, given a decimal number m (10) , the j-th digit (counted from right to left, starting at j = 1) of m (β) is given by:
(m (β) ) j = m • β -(j-1) mod β.(5)
Then, the relative position m can be calculated as:
m = f (m (β) ) = Dr j=1 β (j-1) (m (β) ) j .(6)
Eq. 5 indicates that when β = b 1/Dr , it shares a common term m • β -(j-1) with Eq. 4. Remarkably, both the modulo and trigonometric functions contribute to the periodicity. Based on this observation, Su (2023) hypothesizes that if we temporarily disregard the influence of the flooring operation as well as the period size of the modulo operation, RoPE essentially performs a radix conversion from the decimal system to a b 1/Dr -radix representation.
To prove this hypothesis, we reintroduce the ignored components (the floor function and modulo function) to recover a biased positional estimate m from the RoPE as follows:
m = g(mθ) = Dr j=1 β (j-1) (mθ j ) = Dr j=1 b 1-j Dr (m • b 1-j Dr ) mod 2π . (7
)
Figure 2: The biased positional estimate m of RoPE across different base.
As shown in Figure 2, the biased function recovered from RoPE exhibits a linear trend analogous to that in Eq 6. This linearity is confined to shorter distances for smaller bases, but becomes markedly more pronounced and expands to encompass the full context window as the base increases, suggesting that larger bases possess greater potential for representing positional information over long sequences, which is in accordance with previous findings (Liu et al., 2023b;Men et al., 2024).
Overall, this section established RoPE as a biased βradix encoding by combining theoretical derivation with an empirical linearity analysis. In the next section, we leverage this radix perspective to unify prior context extension methods.
this section cite: ['b21', 'b16']

Section: MRROPE: A UNIFY FRAMEWORK FOR ROPE EXTENSION

this section cite: []

Section: MRROPE THEORY
Why do we need RoPE extension? In RoPE, generalization failure of test-length position information stems primarily from the OOD problem of incomplete-cycle dimensions (i.e., dimensions j such that L/θ j < 2π) (Liu et al., 2023b;Ding et al., 2024;Huang et al., 2023). From the radix theory introduced above, this phenomenon directly parallels the behavior of high-digit truncation in a radix encoding system: Consider a β-radix encoding system, when the input is restricted to the range [0, L], all digit positions starting from the d-th and higher never experience a complete carry-over cycle:
L • β -(j-1) mod β < β -1 , for j = d, d + 1, • • •(8)
A natural solution to scale-up such a unbalanced radix system, is to scale the radix base for digits before dimension d: Given a β-radix, if the j-th digit's base is expanded by a factor λ j , the representable range of the system is rescaled by a factor of Dr j=1 λ j , then the j-th digit of this expanded system can be expressed as follows:
(m (λβ) ) j = m • β -(j-1) j-1 d=1 λ d mod (βλ j ),(9)
where m is the target position for embedding. By rescaling the radix base, the extended mixed radix system is able to prolong the rotational period of each dimension, thereby addressing the high-digit OOD challenge as mentioned above.
this section cite: ['b5', 'b9']

Section: Extending RoPE via Mixed Radix Rotary Embedding (MrRoPE).
In RoPE, a similar mixedradix conversion can also be implemented. By assigning each dimension an independent frequency scaling factor, a mixed-radix RoPE extension is defined as:
mθ ′ j = (m • b -(j-1) Dr j-1 d=1 λ d ) mod 2π , {q, k} m = e imθ ′ W {q,k} x m .(10)
The above formulation establishes the Mixed-Radix RoPE (MrRoPE) framework: any RoPEbased extension method conforming to Eq. 10 instantiates a mixed-radix conversion on positional encoding. The framework's explanatory power stems from its parameterization: the vector λ = {λ 1 , λ 2 , ..., λ Dr } defines the radix conversion factors for each dimension. Consequently, MrRoPE posits that the choice of a length extension method is fundamentally equivalent to choosing a specific policy for redistributing positional information across the frequency spectrum via λ. This provides a unifying lens through which existing methods can be systematically analyzed as constrained instantiations of this general conversion process. We next demonstrate how both NTKaware scaling and YaRN are naturally recovered under this framework.
this section cite: []

Section: NTK-awre Interpolation under MrRoPE.
Inspired by the Neural Tangle Kernel (NTK) theory (Jacot et al., 2018;Tancik et al., 2020), NTK-aware Interpolation (bloc97, 2023) essentially implements a uniform radix scaling across all dimensions, which can be denoted as:
λ j = S 1 Dr -1 .
(11) By uniformly setting the radix conversion factor across dimensions, NTK avoids the abrupt OOD collapse of the highest dimension-a key to its initial success in training-free context extension.
this section cite: ['b10', 'b24']

Section: NTK-by-parts(YaRN) under MrRoPE.
To further address the OOD challenge, YaRN (Peng et al., 2023) first categorizes all D r dimensions into high-, mid-, and low-frequency groups based on their rotation progress and applies a differing conversion scheme to them: For both high-and lowfrequency dimensions, YaRN develops a non-conversion strategy, setting λ j = 1 to better preserve corresponding position information. Besides, for mid-frequency dimensions (i.e., d l ≤ j < d h ), YaRN uses a linear interpolation strategy such that the expansion factor of each radix satisfies:
j-1 d=d l λ d = r j S,(12)
where r j ∈ (0, 1) is a linear scale factor determined by YaRN. More details about YaRN are presented in Appendix A.2.
this section cite: ['b17']

Section: MRROPE METHODOLOGY
Based on the empirical evidence from YaRN, we conjecture that an effective radix conversion algorithm should adhere to the following principle: lower dimensions extrapolate, higher dimensions interpolate, while intermediate dimensions achieve the desired extension factor S.
Although YaRN's formulation does not explicitly define a radix conversion strategy for its intermediate dimensions, we find it implicitly fulfills the above principle through a regressive scaling conversion (i.e., λ j > λ j+1 ), as proved in Appendix A.2.1. This observation raises a fundamental question: is this the optimal conversion strategies for intermediate dimensions?
Within the MrRoPE framework, we can systematically discuss this question by explicitly designing the λ vector. We design two distinct strategies: Uniform Conversion, which applies a constant radix factor (i.e., λ j = λ j+1 ); and Progressive Conversion, which features a monotonically increasing scaling factor (i.e., λ j < λ j+1 ).
Based on these strategies, we are able to implement two additional radix transformation methods distinct from YaRN: MrRoPE-Uni and MrRoPE-Pro, and find a better conversion for intermediate dimensions through comparison.
this section cite: []

Section: MRROPE-UNI
NTK pioneered the concept of uniform radix conversion for context extension. However, its failure to account for spectral distinctions leads to high-frequency distortion and persistent OOD errors.
Accordingly, MrRoPE-Uni adopts a segmentation-based NTK approach in which a constant scaling factor is applied for uniform radix conversion.
Let λ j = c for all j ∈ [d l , d h ), where c is a constant. To achieve the total scale factor S, the radix expansion factor for intermediate dimension can be calculated as:
λ j = S 1 d h -d l .(13)
In summary, MrRoPE-Uni applies a uniform scaling to all middle dimensions, independent of their original frequencies, effectively expanding the positional encoding range by a factor of S.
this section cite: []

Section: MRROPE-PRO
Motivated by high-frequency extrapolation, MrRoPE-Pro employs a progressive radix conversion with dimension-wise scaling. In this scheme, lower (high-frequency) dimensions undergo smaller radix expansions, while higher (low-frequency) dimensions experience proportionally larger expansions. Hence, MrRoPE-Pro better preserves the fine-grained structure of high-frequency dimensions while enabling effective extension of the representable positional range, thus providing a more faithful extrapolation in the intermediate regions.
Let λ j = S ϵj . To obtain a progressive sequence, we assume that the sequence ϵ follows an arithmetic progression. Let ϵ d l -1 = 0 and ϵ j -ϵ j-1 = c, where c is a constant for all j ∈ [d l , d h ), it is straightforward to obtain that ϵ j = (j -d l + 1)c for middle-dimensions. According to Eq. 12, which implies the constraint ϵ j = 1, the following expression for ϵ j in the middle dimensions can be derived:
ϵ j = 2(1 + j -d l ) (1 + d h -d l )(d h -d l ) , λ j = S ϵj .(14)
In summary, MrRoPE-Pro applies a progressive scaling to all middle dimensions, resulting in a slow-to-steep radix conversion that better preserves the positional information encoded in the lower (high-frequency) dimensions.
this section cite: []

Section: GENERAL FORMULATION
Overall Formulation of MrRoPE can be formulized as:
mθ ′ j = (m • b -(j-1) Dr j-1 d=1 λ d ) mod 2π(15)
where D r = |D|/2. For the radix expansion factor λ d on d-th dimension, we define:
• For low-and high-dimensions, the scale factor
λ d = 1 (i.e., d ∈ [1, d l ) ∪ [d h , D r ]
).
• For middle-dimensions, the scale factor λ i values are determined by the method used as follows:
λ d = S 1 d h -d l , if MrRoPE-Uni S 2(1+d-d l ) (1+d h -d l )(d h -d l ) , if MrRoPE-Pro (d l ≤ d < d h ).(16)
Additional tricks for implementations are consistent with YaRN, such as the value of d l and d h , which are provided in Appendix B. Evidently, the fundamental distinction between our method and YaRN resides exclusively in the extrapolation strategy employed for the intermediate dimensions.
Figure 3 shows the cumulative scaling factor s d for each dimension across various context window extension methods, where  4 EXPERIMENTS 4.1 SETTINGS Baselines. We compare MrRoPE with SOTA test-time RoPE-extension methods, including YaRN, NTK.
s d = d-1 j=1 λ j .
All experiments were conducted in an inference setting (no fine-tuning), and thus, no comparisons were conducted with other context extension methods that require training (Shang et al., 2025;Wang et al., 2024;Hua et al., 2024).
Base Models and Tasks. We evaluate MrRoPE on the following RoPE-based LLMs: LLaMA2-7B (Touvron et al., 2023), LLaMA3-8B (Grattafiori et al., 2024), and Qwen2.5-3B (Yang et al., 2025a). Our evaluation mainly addresses three aspects: (1) perplexity curves on a pre-training test set to assess performance under extended context;
(2) long-context stress tasks, including RULER (Hsieh et al., 2024) and Needle-In-a-Haystack (Kamradt, 2023), to evaluate the handling of long-range dependencies; and (3) real-world benchmarks, Infinite-Bench (Zhang et al., 2024) and Longbench-v2 (Bai et al., 2024) to examine model performance in practical long-context scenarios.
this section cite: ['b20', 'b28', 'b8', 'b26', 'b6', 'b7', 'b11', 'b33', 'b1']

Section: Theoretical Analysis.
Besides the regular tests introduced above, we also provide theoretical evidence supporting the reliability of MrRoPE. Inspired by prior work (Liu et al., 2023b;Barbero et al., 2024), we first investigate the attention scores, with a particular focus on their performance improvement in the middle dimensions. Moreover, based on the theory of Men et al. (2024), we employ the cosine similarity metric to assess the enhancement in the theoretical maximum encoding length achieved by MrRoPE.
this section cite: ['b2', 'b16']

Section: LONG SEQUENCE LANGUAGE MODELING
Long-sequence modeling serves as the most direct and fundamental evaluation of positional encoding methods. To assess this ability, we compute perplexity scores on ten randomly sampled sequences exceeding 128K tokens from the Proofpile dataset (Azerbayev et al., 2022), using float32 precision. Table 1 reports results across different base models and extension strategies; due to the space limitation, perplexity results of LLaMA2-7B are provided in Appendix C.1. MrRoPE-Pro demonstrates superior and consistent performance across the entire extended context window. On both LLaMA3-8B and Qwen2.5-3B, MrRoPE-Pro steadily achieves the lowest perplexity across all evaluated lengths, ranging from 8K to 128K. The merit is particularly evident in the shorter-range context: for instance, on LLaMA2-7B at 4K, MrRoPE-Pro attains a perplexity of 5.72, outperforming YaRN (6.02) and MrRoPE-Uni (5.84). This superior performance can be attributed to the progressive extension strategy, which avoids out-of-distribution positional embeddings in higher dimensions while retaining high-frequency details in the original RoPE structure. By mitigating distortions in both local and global positional information, MrRoPE-Pro provides a more balanced and effective approach to context window extension.
Considering the weaker long-context performance of MrRoPE-Uni than the progressive one, we mainly focus on MrRoPE-Pro and YaRN in the following experiments.
this section cite: ['b0']

Section: LONG CONTEXT BENCHMARKS
To further evaluate real-world long-context reasoning capabilities, we assess model performance on standardized benchmarks. We mainly focus on two key capabilities: (1) the ability to retrieve relevant information, and (2) the ability to utilize such information for real-world tasks. MrRoPE-Pro enables robust context window extension in long-context stress task. To assess the retrieval ability mentioned above, we apply the NIAH test (Kamradt, 2023) with varying context lengths and insertion depths. We compare MrRoPE-Pro, our best-performing method, against YaRN on LLaMA3-8B, measuring recall with ROUGE-1 (Lin, 2004). Figure 4 shows that MrRoPE-Pro extends the effective context window to at least 96K tokens, exhibiting markedly more stable scaling. Within shorter contexts (8K-56K), it consistently achieves higher scores, approaching near-perfect accuracy at insertion depths of 80-100%. Beyond 96K, the performance gap widens: MrRoPE-Pro maintains clear superiority across the 30-70% range, demonstrating robustness in mid-context retrieval under ultra-long settings. Even at 120K tokens (15× the pre-training length), it sustains over 85% recall at most depths, establishing a strong upper bound for practical long-context applications.
We further evaluate MrRoPE on the RULER benchmark, with results presented in Table 2. Notably, MrRoPE demonstrates substantially superior performance across the extended context window. This advantage is particularly pronounced from 64K to 128K tokens, where YaRN's performance sharply declines from 89.5 to 79.9, while MrRoPE maintains a relatively stable level of performance.
this section cite: ['b11', 'b13']

Section: THEORETICAL ANALYSIS
This section establishes the theoretical basis for MrRoPE-Pro's extrapolation capabilities by analyzing its merits for context window extension. The analysis proceeds in two parts: first, we examine the mechanism responsible for its superior extrapolation limit; second, we demonstrate how this design enlarges the effective context window in the self-attention perspective. MrRoPE-Pro significantly extends the theoretical context window upper-bound of the base model. The RoPE Bound Theory establishes a method for analyzing the potential context length of LLMs by relating it to the rotational base b. This theory demonstrates that the upper bound of the effective context window is defined by the root of the function B θ (m) = Dr j=1 cos(mθ j ). shows the evolution of B(m) with relative distance for various extension methods, with the vertical dashed line indicating this root. Under a base of 10,000, MrRoPE-Pro extends the theoretical context window upper bound from 1K to 28K-nearly five times the 6K bound achieved by YaRN. As the key difference between methods lies in their middle-dimensional strategies, Figure 5b isolates the contribution of these dimensions to B θ (m). Evidently, MrRoPE exhibits significantly smaller oscillations and a more stable trend than alternative methods.
this section cite: []

Section: MrRoPE-Pro operates by refining intermediate-dimensional features to stabilize attention score distributions.
In large language models, attention score variation is a key metric for evaluating RoPE-based extension methods. Therefore, to compare MrRoPE-Pro with YaRN, we analyze changes in attention scores across middle dimensions with respect to relative distance. As shown in Figure 6, MrRoPE-Pro extends the periodicity of the original RoPE frequency signals in middle dimensions, maintaining monotonic attention variation over a broader range. This improves the consistency of positional information encoding. Moreover, MrRoPE-Pro produces attention distributions that better match pre-training patterns compared to YaRN, enhancing training-free extrapolation.
this section cite: []

Section: CONCLUSION
We present a unified theory, MrRoPE, linking major RoPE-extension methods to radix conversion, where existing methods emerge as special cases. Based on this, we propose MrRoPE-Pro, a trainingfree method that effectively mitigates the high-dimensional OOD problem by progressively scaling the radix base. Empirical results on real-world benchmarks show that MrRoPE-Pro nearly doubles the practical context window and surpasses strong baselines, including closed-source models. Theoretically, our improvement stabilizes the attention scores in intermediate dimensions and maximally restores high-frequency information, thereby pushing the upper bound of the effective context window to its maximum.
ETHICS STATEMENT This work presents a context window extension theory and methodology for Large Language Models via positional encoding. Our research is based on publicly available, pre-trained base models (e.g., LLaMA series) and standard, open-source benchmarks. All data used for evaluation are in the public domain and contain no private or sensitive information. Our work does not introduce new data collection practices or target specific, sensitive applications. REPRODUCIBILITY STATEMENT To facilitate reproducibility, the source code for MrRoPE-Uni/Pro, along with the baseline method YaRN and NTK, has been anonymized and submitted as supplementary materials with this paper. Detailed descriptions of the experimental configuration and all hyperparameters are provided in the Appendix. Our experiments utilize standard, open-source benchmarks and their corresponding official test scripts to ensure all results are easily verifiable.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2022)
Ref_id:b1 Title: Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks Year: (2024)
Ref_id:b2 Title: Round and round we go! what makes rotary positional encodings useful? arXiv preprint Year: (2024)
Ref_id:b3 Title: Ntk-aware scaled rope allows llama models to have longer context windows Year: (2023)
Ref_id:b4 Title: Extending context window of large language models via positional interpolation Year: (2023)
Ref_id:b5 Title: Longrope: Extending llm context window beyond 2 million tokens Year: (2024)
Ref_id:b6 Title: The llama 3 herd of models Year: (2024)
Ref_id:b7 Title: Ruler: What's the real context size of your long-context language models Year: (2024)
Ref_id:b8 Title: Fourier position embedding: Enhancing attention's periodic extension for length generalization Year: (2024)
Ref_id:b9 Title: Advancing transformer architecture in long-context large language models: A comprehensive survey Year: (2023)
Ref_id:b10 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b11 Title: Needle in a haystack -pressure testing llms Year: (2023)
Ref_id:b12 Title: The impact of positional encoding on length generalization in transformers Year: (2023)
Ref_id:b13 Title: ROUGE: A package for automatic evaluation of summaries Year: (2004-07)
Ref_id:b14 Title: Lost in the middle: How language models use long contexts Year: (2023)
Ref_id:b15 Title: Scaling laws of rope-based extrapolation Year: (2023)
Ref_id:b16 Title: Base of rope bounds context length Year: (2024)
Ref_id:b17 Title: Yarn: Efficient context window extension of large language models Year: (2023)
Ref_id:b18 Title: Train short, test long: Attention with linear biases enables input length extrapolation Year: (2021)
Ref_id:b19 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b20 Title: Longrope2: Near-lossless llm context window scaling Year: (2025)
Ref_id:b21 Title: Transformer upgrade path: 10. rope is a beta-based encoding Year: (2023-07)
Ref_id:b22 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b23 Title: A length-extrapolatable transformer Year: (2022)
Ref_id:b24 Title: Fourier features let networks learn high frequency functions in low dimensional domains Year: (2020)
Ref_id:b25 Title: Kimi k1. 5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b26 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b27 Title: Attention is all you need. Advances in neural information processing systems Year: (2017)
Ref_id:b28 Title: Resonance rope: Improving context length generalization of large language models Year: (2024)
Ref_id:b29 Title: Effective long-context scaling of foundation models Year: (2023)
Ref_id:b30 Title: Qwen3 technical report Year: (2025)
Ref_id:b31 Title: Qwen3 technical report Year: (2025)
Ref_id:b32 Title: Open foundation models by 01 Year: (2024)
Ref_id:b33 Title: Extending long context evaluation beyond 100k tokens Year: (2024)
