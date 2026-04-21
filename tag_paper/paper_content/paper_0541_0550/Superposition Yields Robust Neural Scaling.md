Title: Superposition Yields Robust Neural Scaling
Abstract: The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law, that loss decreases as a power law with model size, remains unclear. We propose that representation superposition, meaning that LLMs represent more features than they have dimensions, can be a key contributor to loss and cause neural scaling. Based on Anthropic's toy model, we use weight decay to control the degree of superposition, allowing us to systematically study how loss scales with model size. When superposition is weak, the loss follows a power law only if data feature frequencies are power-law distributed. In contrast, under strong superposition, the loss generically scales inversely with model dimension across a broad class of frequency distributions, due to geometric overlaps between representation vectors. We confirmed that open-sourced LLMs operate in the strong superposition regime and have loss scaling inversely with model dimension, and that the Chinchilla scaling laws are also consistent with this behavior. Our results identify representation superposition as a central driver of neural scaling laws, providing insights into questions like when neural scaling laws can be improved and when they will break down. 1

Section: Introduction
The remarkable success of large language models (LLMs) has been driven by the empirical observation that increasing model size, training data, and compute consistently leads to better performance [1][2][3][4]. Across a wide range of tasks -including language understanding [1,5,6], math [7][8][9][10], and code generation [11,12] -larger models achieve lower loss, higher accuracy, and greater generalization abilities [2,13]. This consistent trend, known as neural scaling laws, has been observed across multiple model families and architectures, fueling the development of increasingly large models [2][3][4]. These scaling laws have not only shaped the current strategies for building better models but have also raised fundamental questions about why such simple and universal patterns emerge in complex learning systems.
The power-law loss with model size plays a central role in both the practical design and the theoretical understanding of large-scale machine learning systems, yet its origin remains inconclusive [3,[14][15][16][17][18][19][20][21][22][23][24][25]. Various explanations have been proposed, drawing from statistical learning theory and empirical phenomenological models, including improved function or manifold approximation in larger models [14,15], and enhanced representation or skill learning in larger models [19][20][21][22]. In the limit of infinite data, many of these explanations predict a power-law decay of loss with model size, provided the underlying data distribution also follows a power law. The scaling exponents are sensitive to the properties of the data distribution. Moreover, the connection between these mechanistic explanations and the behavior of actual LLMs needs further exploration. When considering LLMs specifically, it becomes clear that representation or embedding can be a limiting factor, which is closely related to a phenomenon called superposition [26,27], yet this aspect has not been thoroughly studied. LLMs must learn embedding vectors for tokens, process these representations through transformer layers to predict the next token, and use a final projection (the language model head) to generate the output. Conceptually, fitting functions or manifolds and learning skills or grammars are primarily tasks of the transformer layers, while representation is more directly tied to the embedding matrix and the language model head. To represent more than fifty thousand tokens -or even more abstract concepts -within a hidden space of at most a few thousand dimensions, the quality of representations is inevitably constrained by the model dimension or width, contributing to the final loss. Although models can represent more features than their dimensionality would suggest through a mechanism known as superposition [27], prior works on neural scaling laws seem to fall in the weak superposition regime implicitly [15][16][17][18][19][20], which may be less relevant to the regime where LLMs operate. This gap leads us to study Question: How will superposition influence the loss scaling with model dimension (width)?
Varying the degree of superposition and data structure, when is the loss a power law? And if the loss is a power law, what will the exponent be?
We adopt a toy model construction similar to [27] to study how superposition affects neural scaling laws. In the toy model, representations are learned by recovering data, each composed of multiple latent features. These features in data have different frequencies of occurrence, reflecting their relative importance. Weak superposition means that only the most frequent features are perfectly represented, while the others are ignored. As illustrated in Figure 1a, the first three of six features are represented
... ... data dimension data space hidden space : model dimension data space a Toy model of representation learning via data recovery Loss weight matrix weight matrix b c Rank Freq. Rank Freq. in the three-dimensional space without interference, and the remaining three are omitted. We find that in the weak superposition regime, the scaling of loss with model dimension depends sensitively on how feature frequency decays with rank: the loss follows a power law with model size only if the feature frequencies themselves follow a power law, provided that m is sufficiently large (Figure 1b). By contrast, strong superposition allows many more features to be represented, albeit with overlap in the representation (Figure 1c). In this regime, the model displays a robust behavior: loss scales inversely with model dimension across different data frequency distributions (Figure 1d). Remarkably, we find that actual LLMs follow a similar scaling. We summarize our contributions as Main results/messages
this section cite: ['b0', 'b1', 'b2', 'b3', 'b0', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b1', 'b12', 'b1', 'b2', 'b3', 'b2', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b13', 'b14', 'b18', 'b19', 'b20', 'b21', 'b25', 'b26', 'b26', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b26']

Section: Superposition No superposition
• Loss in the weak superposition regime depends on summing frequencies of ignored features, which is a power law if frequencies follow a power law.
• In the strong superposition regime, loss arises from the interference between representations and can have robust "one over width" scaling because of the geometry.
• LLMs exhibit strong superposition and agree quantitatively with our toy model predictions.
The rest of the paper will elaborate on the takeaways. In Section 2, we introduce the toy model, describe the data sampling procedure, and explain how we control the degree of superposition.
Section 3 presents the detailed results. In Section 4, we compare our findings to related works. Finally, Section 5 summarizes our conclusions and discusses limitations and future directions.
this section cite: []

Section: Methods
To understand the relationship between superposition and data structure, we need a toy model to represent data features simple enough yet not simpler -two key principles need to be reflected, (i) there are more features to represent than the dimension of the model, and (ii) features occur in data with different frequencies. Later, we will discuss how the loss due to representation studied here may affect the overall final loss in LLMs.
We adopt the toy model of superposition from Anthropic [27] (an autoencoder) with minor modifications (Figure 2a). Input x ∈ R n is a vector with data dimension n being the number of atomic (or irreducible) features. Each element x i in x is interpreted as the activation of this sample at feature i, which follows
x i = u i v i , u i ∼ Bernoulli(p i ) & v i ∼ U (0, 2).(1)
Here, u i sampled from a Bernoulli distribution controls whether the feature i is activated, and v i sampled from a uniform distribution controls the activation strength once feature i is activated. All samples are i.i.d. The frequency of feature i to appear in the data is p i . Without loss of generality, we make the indices of features the same as their frequency or importance rank. The data structure is then about how p i decreases with rank i. The expected number of activations in one input will be referred to as activation density: E = n i=1 p i . The model learns hidden representations by recovering the data, which cannot be done perfectly because the model dimension m is much smaller than the number of possible features in the data n. The trainable parameters are a weight matrix W ∈ R n×m and a bias vector b ∈ R n . The weight matrix embeds data x into a hidden space with dimension m, h = W T x, with m ≪ n. In practice, we fix n as a large number and change the model dimension m. We use W to read out the embedding, where y = ReLU(W h + b). The loss is defined as the difference between the recovered y and the original x, L = ⟨∥y -x∥ 2  2 ⟩ x , where ⟨•⟩ x means average over x distribution.
We can now formally introduce superposition. Note that W i is the representation of feature i in the hidden space, where we use W i to denote the ith row of the W matrix. We emphasize the following Key concepts
• Feature frequency: p i is the probability that feature i is activated (non-zero) in a sample, which is assumed to decrease with i.
• Sparsity: We say features are sparse when E/n is small.
• The feature i is represented (in the hidden space) when W i is non-zero.
No superposition ideally means the first m rows of W form an orthogonal basis (i.e., the first m most important features represented perfectly) and the rest of the rows are zero (i.e., the rest of the features ignored or lost), as illustrated in Figure 2b. Superposition means that there are more than m rows in W with non-zero norms (Figure 2c).
We next summarize important facts of this toy model [27]: Preliminaries Superposition cannot lead to lower losses in a linear model (without ReLU function) due to the large amount of interference. ReLU and negative biases can cancel off interference, realizing error correction. With this non-linearity, superposition can be preferred when feature frequencies are more even (better not to ignore features) and features are sparse in data (error correction is easier).
We can see that in Figure 1, where features are sparse in data, the losses in the strong superposition regime are indeed much smaller than those in the weak superposition regime across several feature frequency distributions.
If one regime is more preferred, we want to approach it more quickly in training. If it is not preferred, we also want to study the scaling behaviors scientifically in that regime. To this end, we introduce a decoupled weight decay (or growth) term in training to tune the degree of superposition:
W i,t+1 = W i,t -η t γW i,t , γ ≥ 0, W i,t -η t γW i,t (1/∥W i,t ∥ 2 -1), γ < 0,(2)
where η t is the learning rate and W i,t is the ith row of the weight matrix at step t (vector operations are element-wise). For weight decay γ < 0, the update corresponds to gradient descent on (∥W i,t ∥ 2 -1) 2 , encouraging unit-norm rows. We implement this weight decay in AdamW [28] optimizer with a warm-up and cosine decay learning rate schedule (details in Appendix B). At each training step, we sample new data.
this section cite: ['b26', 'b26', 'b27']

Section: a b
Weight decay + Weight decay -Freq. We find that the weight decay can robustly control superposition. We first see that important features tend to be represented (associated ∥W i ∥ 2 > 0), and norms of W i become bimodal, clustering near 0 or 1 (Figure 3a). This allows us to define the fraction of represented features as
this section cite: []

Section: Data
ϕ 1/2 = |{i : ∥W i ∥ 2 > 1/2}|/n,(3)
namely, the fraction of rows with norm larger than 1/2. 2 We found that weight decay can tune superposition for all models we trained, with small weight decay γ giving strong superposition, i.e., ϕ 1/2 ≈ 1 ≫ m/n, and large weight decay corresponding to weak superposition, i.e., ϕ 1/2 ∼ m/n (Figure 3b). The ability of weight decay to tune superposition is robust to feature frequency distributions (Appendix D.3). We can then systematically study scaling behaviors in different regimes.
The toy model differs from LLMs in architecture, data, and loss. Since we focus on representations rather than next-token prediction, we omit transformer layers. Conceptually, LLMs map a document to a token, with inputs and outputs in different spaces, while the toy model operates within a single shared space. Despite this, the toy model captures key aspects of language structure through engineered sparsity and feature importance, making its data structure aligned with that of LLMs at a high level. While LLMs use cross-entropy loss and the toy model uses squared error, we can show that this does not affect the scaling behaviors (Appendix A.2). Thus, the toy model is a suitable abstraction for studying representation-limited scaling.
this section cite: []

Section: Results
For a systematic scan, we set p i ∝ 1/i α in this section and can vary the data exponent α to change how skewed p i is. 3 The activation density E is set as 1, whose value can be shown to not affect the scaling (Appendix D.4). We fix data dimension n = 1000, vary model dimension m from 10 to 100, and sweep weight decay γ from -1 to 1. We fit final test losses as a power law, L ∝ 1/m αm , and call α m the model exponent. More details on hyperparameters are in Appendix B.2.
this section cite: []

Section: Weak superposition
We represented features. The optimal biases are b i = 0 for i ≤ ϕ 1/2 n and b i = ⟨x i ⟩ for i > ϕ 1/2 n [27] The loss can then be written as:
L = i>ϕ 1/2 n ⟨(x i -⟨x i ⟩) 2 ⟩ = i>ϕ 1/2 n (⟨v 2 ⟩p i -⟨v⟩ 2 p 2 i ) ≈ ⟨v 2 ⟩ i>ϕ 1/2 n p i .(4)
The last approximation is right when p i ≪ 1 for i > ϕ 1/2 n and p 2 i terms are negligible. We use the definition of x i = u i v i , where v ∼ U (0, 2), giving ⟨v 2 ⟩ = 4/3. We can use the integral n ϕ 1/2 n p i di to estimate the summation, yielding an expression that depends on the number of represented, ϕ 1/2 n, and the data exponent, α (Appendix D.5). We find that, in the weak superposition regime, the actual losses closely match this prediction (Figure 4a). Focusing on cases closest to the ideal nosuperposition scenario, where ϕ 1/2 n = m and the first m features are represented, we observe that such cases occur when α > 1 and yield a model exponent α m ≈ α -1 (Figure 4b). This matches the theoretical expectation that n m p i di ∝ m -α+1 when n ≫ m and α > 1. Thus, in the weak superposition regime, loss scaling is well described by the contribution of unlearned features, that is, the total frequency of features not represented by the model.
We can now answer our Question in the weak superposition regime.
Result 1: "Power law in, power law out" in the weak superposition regime The loss is governed by a sum of frequencies of less frequent and not represented features. Ideally, there are model dimension m most important features being represented. If feature frequencies follow a power law, p i ∝ 1/i α with α > 1, the loss or the summation starting at m will be a power law with m with exponent α -1.
This finding of the specific toy model agrees with previous works with very different settings [15][16][17][18][19][20], where some power-law skill importance or spectrum is assumed.
this section cite: ['b14', 'b15', 'b16', 'b17', 'b18', 'b19']

Section: Strong superposition
We next turn to the strong superposition regime. Consider the case where only feature j is activated. The output y j has activation ∼ W i • W j , leading to a loss that scales as squared overlaps (W i • W j ) 2 due to the definition of loss. The loss arises from the non-zero overlaps between representation vectors. We cannot solve the weight matrix W in this regime. The section goes back and forth between theoretical ansatz and experimental observations to understand the high-level behaviors.
We start by considering relatively even feature frequencies, where trained W i are expected to be isotropic. One simplest theoretical ansatz of isotropic vectors is i.i.d. vectors uniformly on the unit sphere. In R m , the squared overlap of two such random vectors follows Beta( and therefore has mean 1/m and variance 2(m-1) m 2 (m+2) ∼ 2/m 2 . The squared overlaps for isotropic random vectors typically obey 1/m scaling.
The actual trained W i have structures whose norms are bimodal near 1 (Figure 5a), and more important features tend to have larger vector norm (Figure 5b). We want to understand how such a structure will change the scaling of overlaps. It turns out that for better error correction (using bias to cancel interference), the model needs to minimize the maximum overlap rather than the sum of squared overlaps. Consider ν unit vectors w i ∈ R m with ν ≥ m. It can be shown that [29]
max i̸ =j |w i • w j | ≥ ν -m m(ν -1) ≡ κ.(5)
The lower bound, κ ≈ 1/m when ν ≫ m. The bound is met when the vectors form an equal angle tight frame (ETF) [30][31][32], which has no variance in absolute overlaps and appears in contexts such as quantum measurements [33][34][35][36] and neural collapse [37,38]. ETFs in real spaces can only exist if ν ≤ m(m+1) 2 [30][31][32]. We find that the W i with ∥W i ∥ > 1 associated with important features tend to be ETF-like (Figure 5, c and d): the variance of squared overlaps is smaller than that of random vectors and can be near 0 for even feature frequencies (small α); the mean of squared overlaps collapse on 1/m ≈ κ 2 . Being ETF or ETF-like can help error correction and reduce loss values, but would not change the typical scaling with m if the number of vectors is much larger than m. Similar to ETFs, whose number of vectors is bounded, the number of vectors W i with ∥W i ∥ > 1 is around m 2 /2 (Appendix D.6), and the less important features tend not to be represented (norm lower than 1). The model class is reflected by color as panel a, while we use shapes for evaluation datasets [43][44][45][46].
The loss related to model size is fitted as a power law, yielding empirical α m = 0.91 ± 0.04 close to 1. More analysis in Appendix D.7.
Vectors of these less important features cannot be explained with a simple theoretical ansatz. Yet, combining the lessons from random vectors and ETF, we expect the squared overlaps to scale as 1/m robustly for isotropic vectors (confirmed in Appendix D.6). Considering all the overlaps when feature frequencies are even, we then predict the loss to scale as 1/m, which is true (Figure 5e).
When the feature frequencies are skewed (large α), we find that the model exponent α m increases with α and becomes greater than 1. Vectors being non-isotropic, i.e., important features having much smaller overlaps, may lead to larger α m . To illustrate this idea, we conjecture an extreme situation where the m 2 /2 most important features can be ETF-like and contribute negligible loss compared to the less important ones. In the worst case (Appendix A.1), the less important features lead to a loss proportional to n i=m 2 /2 p i ∼ m -2(α-1) , i.e., α m = 2(α -1), which is close to observations (Figure 5e). The real configuration of W i is more complicated than this simple conjecture, requiring advanced future studies on when α m loses robustness and how it depends on feature frequencies sensitively then. To conclude the section, we have Result 2: Geometric origin of 1/m loss scaling (α m = 1) at strong superposition For even feature frequencies, vectors W i tend to be isotropic in space with squared overlaps scaling like 1/m when n ≫ m, leading to the robust 1/m power-law loss. For skewed feature frequencies, representation vectors are heterogeneous in space, making loss sensitive to feature frequencies, where it might need power-law frequencies to have power-law losses.
this section cite: ['b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b29', 'b30', 'b31', 'b42', 'b43', 'b44', 'b45']

Section: LLMs
Finally, we explore how our findings might be relevant to real LLMs [39][40][41][42]. As a naive mapping, we treat tokens as atomic features, with data dimension n equal to the vocabulary size. The model dimension m for LLMs is known. We analyze the language model head, denoted by the weight matrix W . Through the norm and interference distributions of the rows of W , we claim LLMs are in superposition (Appendix D.7). If we measure token frequency, it follows a power law with exponent α close to 1 (Appendix D.7). We conclude, based on the knowledge from toy models, that LLMs operate in a superposition regime, and expect loss to be related to squared overlaps ∼ 1/m. We next calculated the mean squared overlaps of normalized rows W i /∥W i ∥ 2 , and found they roughly obey 1/m scaling (Figure 6a). We argue that cross-entropy loss, given that the overlaps are small in absolute value, can be expanded and approximately scales as the mean square overlaps (Appendix A.2). We therefore expect the loss of representation-limited LLMs to have 1/m scaling. LLM losses are close to a linear function of 1/m (Appendix D.7). Yet when m → ∞, the extrapolation of losses does not hit 0. The non-zero intersection can be due to intrinsic uncertainty in language. Increasing model sizes decreases "wrong" interferences but cannot eliminate uncertainty in the data. So, as in previous papers where loss is decomposed into model size part, dataset size part, and a constant [3], we fit our loss values by the following,
L = C m /m αm + L \m ,(6)
where the model size part C m /m αm is universal (model size is a function of m), and L \m contains loss irrelevant to model size, depending on the evaluation dataset and model class. The fitting yields α m = 0.91 ± 0.04 (Figure 6b). We inferred from the Chinchilla models [3] that due to model size N ∝ m 2.52±0.03 (Appendix D.7), α m = (2.52±0.03)×α N = 0.88±0.06, where α N = 0.35±0.02 [47] is the power-law exponent of loss with model size. The exponents α m from LLMs are close to 1. We highlight the finding as Result 3: Superposition is an important mechanism behind LLM neural scaling laws LLMs operate in the strong superposition regime. The squared overlaps of token representations scale as 1/m, token frequencies are flat (α = 1), and the model size relevant loss scales closely to 1/m, agreeing with the toy model prediction.
this section cite: ['b38', 'b39', 'b40', 'b41', 'b2', 'b2', 'b46']

Section: Related works
Neural scaling laws were first characterized empirically [2], demonstrating that for LLMs, the crossentropy loss improves predictably as a power-law with increased model size (parameters), dataset size, or compute, over multiple orders of magnitude. This finding is built on earlier observations (e.g. [48]) that deep learning performance scales in a smooth power-law fashion with data and model growth. Many works showed the surprisingly universal nature of such scaling behaviors across architectures and tasks [2][3][4], directing further development of LLMs.
Several heuristic toy models have been proposed to explain neural scaling laws. One common view is that models aim to fit data manifolds or functions, and the scaling exponents depend strongly on the structure of the data [14,15]. Another group of models assumes the network learns discrete features or skills [19,20], whose importance follows a power-law distribution, giving results the same as ours in the weak superposition regime. One toy model predicts that loss scales inversely with model width [25], arguing that parameters independently perform the same task with noise, and the scaling follows from the central limit theorem. However, this model applies in the overparameterized cases and may be less relevant to LLMs.
More formal approaches rely on similar heuristics. The scaling behavior depends on how the dataset size and model size approach infinity. When the dataset is fixed and model size grows to infinity, the system is variance-limited, and loss scales as 1/m by central limit theorem arguments [15]. When the dataset size grows to infinity first, the loss scaling enters the resolution-limited regime. In linear models or kernel methods, this leads to α m = α ′ -1 [15][16][17][18], seemingly consistent with our weak superposition regime. Here, α ′ is the exponent of the power-law decay of kernel eigenvalues, which can be seen as abstract feature importance. Considering neural tangent kernels, α ′ depends on both data and the model configuration. Our work may be framed as mechanistically showing that α ′ = α (α is the intrinsic data exponent) when models have no superposition, and α ′ is something else when models have strong superposition, which is new. The resolution-limited regime has also been described as fitting the data manifold [15].
Our toy model is based on Anthropic's model of superposition [27] (an autoencoder), with modifications to the data sampling. The original study explored how data structure influences superposition but did not explicitly control it. Related models have appeared in compressed sensing [49][50][51][52][53] and neural information processing [54,55], yet with distinct contexts and objectives. Besides representation, people also studied calculation in superposition [56,57].
this section cite: ['b1', 'b47', 'b1', 'b2', 'b3', 'b13', 'b14', 'b18', 'b19', 'b24', 'b14', 'b14', 'b15', 'b16', 'b17', 'b14', 'b26', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56']

Section: Discussion
Our work is built on observations of the toy model and analysis without rigorously solving the toy model. We are thus limited to explaining deeper behaviors in the toy model. Our analysis of LLMs suggests they are in the strong superposition regime, but the underlying reasons were not studied in detail. We believe one reason is that features are sparse in language, as the number of tokens required to predict one token is much less than the total number of tokens. The softmax function may also be important since it is strong at error correction, giving superposition an advantage.
Neural scaling laws also include scaling laws with dataset size and with training steps, which we did not study. At each step, a fixed number of new data points are used for optimization. So, we expect the scaling with the total data amount and that with training steps will be the same, similar to the results at weak superposition [20]. However, in the strong superposition regime, data or training step scaling is related to angle distribution and how angles between representations evolve, which cannot be easily explained without rigorous solving.
We focused on representation loss, yet LLMs should also have losses due to parsing or processing in the transformer layers. We imagine that the loss associated with model size can be written as
C m /m αm = f m (m) + f ℓ (ℓ), (7
)
where ℓ is the depth of the LLM, f m and f ℓ are two functions capturing the loss due to representation and parsing, respectively. A future direction is to study the parsing-limited scaling (i.e, f ℓ (ℓ) function) independently. It is also plausible that the observed scaling of inference time [58] is connected to this parsing-limited regime. We here write the equality because ℓ depends on m in LLMs [39][40][41][42]. Given model size N , m and ℓ are constrained (roughly, N ∝ m 2 ℓ). There is an optimal m-ℓ relationship such that the loss f m (m) + f ℓ (ℓ) can be minimized given N [59]. At this optimal m-ℓ relationship, f m (m) and f ℓ (ℓ) should be balanced. Therefore, we expect f ℓ (ℓ) to be similar to f m (m). And if f m (m) ∼ 1/m due to superposition and f ℓ (ℓ) is similar, we can measure an empirical α m ≈ 1 from data, which is true. Or, if the width-limited loss is much larger, we can also observe that the total loss due to model size has α m ≈ 1. We conclude that superposition in any case is an important mechanism underlying neural scaling laws.
Beyond explaining existing phenomena, our results may offer guidance for future LLM development and training strategies: Assuming our explanation of width scaling is correct, we ask can we change the loss scaling with width to be faster than power laws, or to have larger exponents? The answer is no for natural languages but may be yes for domain tasks with super skewed feature frequencies. Another question is when the scaling law will stop? Based on our naive connection between features and tokens, the answer is that when the model dimension reaches the vocabulary size, the loss limited by width will deviate from a power law and vanish. However, the vocabulary size may set a lower bound for the true number of independent things in language, then the power law with width may continue for a longer time.
Recognizing that superposition benefits LLMs, encouraging superposition could enable smaller models to match the performance of larger ones (with less superposition) and make training more efficient. Architectures such as nGPT [60], which constrain hidden states and weight matrix rows to the unit sphere (promoting superposition), demonstrate improved performance. Optimizers that stabilize training without weight decay have also shown promising results [61], potentially due to enhanced superposition. Yet, these improvements may be related to altering coefficients in the neural scaling laws rather than the exponents. We also acknowledge that encouraging superposition may cause difficulties for the mechanistic interpretation of models and AI safety [27,62].
As a side note, with the same pre-training loss, LLMs with different degrees of superposition may exhibit differences in emergent abilities such as reasoning or trainability via reinforcement learning [63], requiring future studies.
In conclusion, we studied when loss can be a power law and what the exponent should be with different data properties and degrees of superposition. We found that geometric interference at strong superposition may explain the LLM neural scaling laws observed [3]. Our results contribute to a deeper understanding of modern artificial intelligence systems, which also open various directions for future research. We hope our insights will support the continued development and training of more capable and efficient LLMs.
this section cite: ['b19', 'b57', 'b38', 'b39', 'b40', 'b41', 'b58', 'b59', 'b60', 'b26', 'b61', 'b62', 'b2']

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: We start our Discussion section by discussing limitations.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: Assumptions are clearly stated.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We provide all details sufficient to reproduce. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We submit our code in the supplementary material.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We followed the code of ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: Our work is theoretical, with no obvious social impact.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: We do not have such risks.
this section cite: []

Section: A Theoretical analysis

this section cite: []

Section: A.1 Toy model loss
We provide a simple analysis for toy model loss scaling. The expected loss in the weak superposition regime is well explained by Equation (4). We do not need to repeat it here.
In the strong superposition regime, we consider an even special data sampling, where data x is sampled such that each data point has and only has one activated feature. The frequency for feature i to be activated is still p i . After determining which feature is activated, say i, we still sample x i as v i from U (0, 2). This sampling is different from the experiments. Yet, since we learned that activation density does not affect scaling exponent (Figure 14), we expected this analysis to predict at least the scaling exponent. Under the assumptions, we have
L = n i=1 p i j̸ =i ReLU 2 (W j • W i v i + b j ) + (ReLU(W i • W i v i + b i ) -v i ) 2 vi .(8)
We are unable to solve for the optimal W and b such that this loss L is minimized. Yet, it is easy to see that we want ∥W i ∥ 2 2 to be close to 1, W j • W i to be as small as possible, and b j to be small negative values of the same order of magnitude as W i • W j , such that the interference terms ReLU(W j • W i v i + b j ) may vanish and the recovered feature value ReLU(W i • W i v i + b i ) can be close to the real one v i .
For convenience, based on the observation that the vector norms are bimodal around 1 in the strong superposition regime, we define strongly represented features as those that have ∥W i ∥ 2 > 1, which are more frequent and ETF-like, and weakly represented ones for those with ∥W i ∥ 2 < 1. We can quantify the fraction of strongly represented as
ϕ 1 = |{i : ∥W i ∥ 2 > 1}|/n,(9)
which significantly exceeds m/n and is around the ETF expectation m 2 /2n (Figure 16).
We now review our conjectured extreme configuration, consisting of strongly represented and weakly represented features. The first ϕ 1 n most important features are considered to be strongly represented, whose absolute overlap with any other representation scales as 1/m. The rest of the features are weakly represented and squeezed into a small angle such that they have small overlaps with the strongly represented, while they can have large overlaps with each other. With such a configuration, the first ϕ 1 n terms in the summation of Equation (8) will scale as 1/m since each term ⟨• • • ⟩ vi scales as 1/m. The rest of the terms in Equation ( 8), in the worst scenario that the terms ⟨• • • ⟩ vi do not decrease obviously with m, will be proportional to n i=ϕin p i . If the strongly represented features dominate, we can have 1/m scaling for the loss. On the contrary, when the weakly represented dominate, we expect the loss to have scaling like n i=ϕin p i . More specifically, if p i ∼ 1/i α (α > 1) and we use m 2 /2 to approximate ϕ i n, the loss scales as 1/m 2(α-1) .
this section cite: []

Section: A.2 Cross-entropy loss
We provide the reason why cross-entropy loss also scales as squared overlaps. We consider the last hidden state after going through the normalization layer is W i /∥W i ∥ 2 , such that the output should be the ith token. By constructing such an example, we ignore possible loss due to parsing but focus on the loss just due to representation. The loss from this data point is
L = -ln e ∥Wi∥2 j e Wi•Wj /∥Wi∥2 = ln 1 + j̸ =i e Wi•Wj /∥Wi∥2-∥Wi∥2 . (10
)
We assume that W i • W j /∥W i ∥ 2 is much smaller than 1 since we know the overlap scale as 1/m. We then approximate the loss via Taylor expansion
L = ln 1 + (n -1)e -∥Wi∥2 + j̸ =i [W i • W j /∥W i ∥ 2 + (W i • W j /∥W i ∥ 2 ) 2 /2]e -∥Wi∥2 . (11
)
In the first thought, the summation j̸ =i W i • W j should be zero since there are positive and negative overlaps distributed evenly if the vectors span the whole space. But in language, one sentence can have different continuations, connecting different tokens. For example, both putting "cats" or "dogs" after "I like" are legit. The existence of data "I like" then will tend to squeeze different tokens closer to each other. The summation W i • W j should be a small positive constant ϵ D,i related to the correlation in data. The reason we keep the second-order term is clear now as they are the lowest order terms related to model sizes. We keep expanding the ln function and have
L = (n -1)e -∥Wi∥2 + ϵ D,i e -∥Wi∥2 ∥W i ∥ 2 + 1 2 j̸ =i W i • W j ∥W i ∥ 2 2 e -∥Wi∥2 .(12)
The part related to the model size is mainly
L m = 1 2 j̸ =i W i • W j ∥W i ∥ 2 2 e -∥Wi∥2 .(13)
In this construction, one can see that once ∥W i ∥ 2 is sufficiently large, the loss can be arbitrarily low, which does not happen in reality. The reason is still related to the intrinsic uncertainty in language data. If one sentence can have different continuations, we need in the hidden space, a region that can lead to large probabilities over different tokens. However, when the norm is too large, one will find that the hidden space is sharply separated -each hidden state yields high probability only on one token. We then expect the norm ∥W i ∥ 2 to be as large as possible such that ne -∥Wi∥2 is small while ∥W i ∥ 2 is upper bounded by intrinsic data uncertainty. Therefore, ∥W i ∥ 2 should not depend on model size much (verified in Appendix D.7). The loss related to model size L m then scales as 1/m since the cosine similarity scales as 1/m and L m is related to the squared cosine similarity in the lowest order approximation.
this section cite: []

Section: B Toy model training
In this Appendix, we explain how we trained the toy models and obtained raw data. There are two classes of toy models trained. The first one is a large toy model with data dimension n = 10240, which is reported in Figure 1 and Figure 9 to show scaling behavior across around two orders of magnitude. The other toy model class is small toy models fixing n = 1000, such that we can scan more hyperparameters. Figures 3, 4, 5, 14, and 9 use small toy models.
this section cite: []

Section: B.1 Large toy models
We implemented a neural network experiment to study the scaling of feature representation and recovery. The toy model is defined as a two-layer neural network with ReLU activation (see Figure 2).
The hyperparameters are given as follows.
• Data dimension n: 10240
• Model dimension m: Varied exponentially from 2 3 to 2 10
• Batch size: 2048 (tested up to 8192, which does not affect final loss)
• Total training steps: 20000 (tested up to 80000, which does not affect final loss)
• Learning rate: Initially set to 0.02, scaled according to hidden dimension
• Weight decay: -1.0 for strong superposition, and 0.1 for weak superposition
• Device: Training performed using one V100 GPU, with floating-point precision (FP32) Data points x were synthetically generated at each training step according to Equation (1) to simulate feature occurrence frequencies. We considered three distributions with activation density E = 1:
• Exponential: p i ∝ e -i/400
• Power-law:
p i ∝ i -1.2
• Linear:
p i ∝ n -i
We employed the AdamW optimizer with distinct learning rates and weight decay settings for the weight matrix W and bias vector b. Specifically, for weight matrix W , learning rate was scaled as lr × (8/m) 0.25 with specified weight decay. And for bias vector b, a learning rate of 2.0/m was used with no weight decay. A cosine decay learning rate schedule with a warm-up phase (5% of total steps) was implemented. At each training step, input data batches were dynamically generated based on the selected probability distribution. The final test loss is calculated across newly sampled data with a size being 100 times the batch size.
The model and optimizer were compiled and executed on a CUDA-enabled GPU for efficient training. After training, weight matrices W and training losses were stored and analyzed.
Final outputs, including weight matrices and training loss histories, were saved in PyTorch format for subsequent analysis and visualization.
This setup provided a structured exploration of feature representation scaling under varying dimensions and distributions, crucial for understanding superposition and scaling laws in neural networks.
The code can be found in exp-17.py.
this section cite: []

Section: B.2 Small toy models
We conducted numerical simulations using a neural network model designed for feature recovery. The objective was to analyze the model's behavior across various conditions involving feature frequency skewness (controlled by data exponent α), model dimensions, and weight decay parameters.
In the small toy models reported in Figures 3, 4, 5, 14, and 9, we set the hyperparameters as
• Feature dimension n: Fixed at 1000.
• Hidden dimension m: Varied logarithmically between 10 and 100 , across 6 distinct sizes, i.e., m = 10, 15, 25, 39, 63, 100.
• Batch size: 2048.
• Training steps: 20000 steps for each condition.
• Learning rate: Initialized at 1 × 10 -2 , dynamically adjusted using cosine decay scheduling with a warm-up phase of 2000 steps.
• Weight decay: Explored systematically from -1.0 to 1.0, in increments of 0.22 approximately (10 discrete values).
• Data exponent α: Ranged linearly from 0 to 2, with 17 discrete steps.
In Figure 14, we fix data exponent α = 1 while scan 9 activation densities linearly from 1 to the maximal value n i=1 1/i. All other settings are the same. Synthetic data was generated for each batch based on a power-law probability distribution, defined as: p i ∝ 1 i α where i ∈ {1, 2, . . . , n} with the condition i p i = E. Each element of the batch data x was randomly activated based on this probability, then scaled by a uniform random value between 0 and 2.
At each training step, input batches were regenerated, and the learning rate was updated following the cosine decay schedule described above.
The training performance was evaluated using Mean Squared Error (MSE) loss computed between the network output and input batch data at every step. Final weights were saved for further analysis. The final test loss is calculated across newly sampled data with a size being 100 times the batch size.
The simulations were performed in parallel using 96 CPU cores, where each core executed one distinct parameter combination defined by the weight decay and data exponent values. Or, in Figure 14, the parameter combination is defined by weight decay and activation density values.
Loss histories and trained weight matrices were saved separately for post-experiment analysis. Files were systematically indexed to indicate the corresponding experimental parameters. This detailed setup facilitated a comprehensive investigation of model behavior under diverse training and data distribution conditions.
The code can be found in exp-10.py, exp-10-3.py, and exp-15.py.
this section cite: []

Section: C LLM evaluation

this section cite: []

Section: C.1 Overlap analysis
We analyzed the row overlaps of the language model head weight matrices among various large language models (LLMs) to investigate the geometric properties of their hidden spaces.
We selected models from the following families, varying widely in parameter count:
• OPT (from OPT-125m to OPT-66b) • Qwen2.5 (from 0.5B to 72B) • GPT-2 (GPT2, GPT2-Medium, GPT2-Large, GPT2-XL) • Pythia (from 70m to 12B) Weights were downloaded directly from Hugging Face model repositories. For each model, the weight matrix or language modeling head was normalized by its row norms:
W i ← W i ∥W i ∥ 2 + ϵ
, ϵ = 10 -9 , where ϵ is for numerical stability.
We computed the pairwise absolute cosine overlaps between all normalized vectors using batch-wise computations for efficiency. The overlap between embedding vectors W i and W j is given by:
overlap(W i , W j ) = W i • W j ∥W i ∥ 2 ∥W j ∥ 2 .
To handle large embedding matrices efficiently, overlaps were computed in batches (size of 8192 vectors).
We calculated two key statistics for the overlaps within each model:
• Mean Overlap: The average of absolute overlaps for all unique vector pairs:
mean_overlap = i<j overlap(W i , W j ) n(n -1)/2
• Overlap Variance: Calculated as:
variance_overlap = i<j (overlap(W i , W j ) -mean_overlap) 2 n(n -1)/2
From these values, we can calculate mean square overlaps as mean_overlap 2 + variance_overlap.
The calculations were accelerated using GPU resources (CUDA-enabled) to efficiently handle computations involving extremely large matrices.
Results including mean overlaps, variances, and matrix dimensions were recorded for comparative analysis across model sizes and architectures.
The code is in overlap-0.py.
this section cite: []

Section: C.2 Evaluation loss
This experiment aims to evaluate multiple large language models (LLMs) efficiently using model parallelism and dataset streaming techniques. The models were assessed on standard text datasets to measure their predictive performance systematically.
Models were selected from Hugging Face and evaluated using a model-parallel setup:
• OPT series • Qwen2.5 series • GPT-2 series • Pythia series
We used the following publicly available datasets for evaluation:
• Wikitext-103: Standard English language modeling dataset.
• Pile-10k: A subset of The Pile, designed for diverse textual data.
• C4: Colossal Clean Crawled Corpus, containing large-scale web text.
• BookCorpus: Large-scale collection of books used for unsupervised learning.
Datasets were streamed directly, efficiently sampling 10000 text segments with a maximum sequence length of 2048 tokens (∼ 2 × 10 7 tokens).
Texts from datasets were tokenized using the respective model-specific tokenizers. Tokenization involved truncation and manual padding to uniform batch lengths. Specifically, padding tokens were assigned an ID of 0, and label padding utilized a special token (-100) to ensure they did not contribute to loss computations.
Each model was loaded using Hugging Face's AutoModelForCausalLM with model parallelism enabled, allowing the evaluation of large models that exceed single-GPU memory limits. Evaluations were conducted in batches, employing a DataLoader with a custom collate function for optimized memory use.
The model's predictive performance was assessed by computing loss values internally shifted by the Hugging Face library, suitable for causal language modeling.
Model parallelism was implemented to efficiently distribute computations across multiple GPUs, leveraging CUDA-enabled hardware.
For each model and each dataset, we run one evaluation and save the evaluation losses.
Random seeds and deterministic sampling ensured reproducible dataset selections, though explicit seed settings were noted as commented options within the implementation.
Evaluation results, including loss metrics and potentially intermediate model states, were systematically stored for detailed post-analysis.
The code is in cali-1.py.
this section cite: []

Section: C.3 Token frequency
The purpose of this analysis is to compare token frequencies generated by different tokenizers across several widely-used textual datasets. Understanding these frequencies helps in assessing the representational capacity and efficiency of tokenizers used by various large language models.
We considered the same four datasets mentioned for LLM evaluation. And we use four different tokenizers from the four model classes we evaluated.
Each tokenizer processed textual data from the specified datasets, streaming data directly to efficiently handle large-scale inputs. A target of 1,000,000 tokens per tokenizer-dataset pair was set to ensure sufficient statistical representativeness.
For each dataset-tokenizer combination:
1. Text samples were streamed directly from the datasets.
2. Text was tokenized without adding special tokens (e.g., EOS). 3. Token frequencies were counted and accumulated until the target token count (1 million tokens) was reached. 4. Token frequencies were saved as JSON files for subsequent detailed analyses.
Token frequency data was systematically stored for each tokenizer and dataset combination, enabling comparative analyses of token distributions. The data files provide foundational insights into tokenizer efficiency and coverage across diverse textual domains. The code is in token-freq-0.py.
this section cite: []

Section: D Figure details and supplementary results
Here, we show how to process the raw data obtained from toy models or LLMs to generate results seen in the main text. Some supplementary analysis is also conducted to support the main text arguments.
this section cite: []

Section: D.1 Figure 1
The toy models reported in Figure 1 are large toy models with data dimension n = 10240 explained in Appendix B.1. After obtaining the final losses, we directly plot them with respect to the model dimension m. Error bars are calculated as the standard deviation of losses over 100 batches. The error bars are smaller than the dots (Figure 1, b and d).
When we are fitting the loss in log-log as a line, we choose the linear part to fit. If the loss versus model dimension curve is obviously not a line, we fit the whole curve as a line and output the R 2 value as a measure of how non-linear it is. Specifically, we fit the last five points for the power-law decay feature case in the weak superposition regime (yellow data in Figure 1b). Other cases in the weak superposition regime are fitted to a line with all data. In the strong superposition regime, when feature frequency decreases as a power law or as a linear function, we fit the data as a line starting from the third point (yellow and green in Figure 1d). And for exponential decay feature frequencies, we fit all the data with a line. In the strong superposition regime, the measure model exponent α m are close to 1: 1.01 ± 0.05 (exponential decay), 1.0 ± 0.1 (power-law decay), and 0.89 ± 0.05 (linear decay).
The LLM data are copied from Figure 6b, with slope -0.91 ± 0.04 being close to 1 as well. We will explain details about Figure 6  We also output the weight matrix W for these large toy models (Figure 7). They follow the same pattern that in the weak superposition regime, row norms are bimodal and are either close to 0 or 1, making 0.5 a good separation point for measuring how many features are represented. And in the strong superposition regime, the row norms are distributed near 1, and 1 is a good separation point for the two peaks, i.e., the peak greater than 1 refers to strongly represented features which are more important, and the peak smaller than 1 corresponds to the weakly represented.
We can analyze the large toy model in the same way as what has been done in Figure 4 and 5. The fraction of represented features ϕ 1/2 is calculated, which is 1 in the strong superposition regime, while it is close to m/n in the weak superposition regime (Figure 8a).
With the measured ϕ 1/2 , we can estimate the loss due to unlearned features, ⟨v 2 ⟩ n i=ϕ 1/2 n p i . This theoretical value agrees well with the actual loss in the weak superposition regime (Figure 8b).
In the strong superposition regime, the fraction of strongly represented features is calculated, agreeing with the expectation that the number of strongly represented features is much larger than m but bounded by some value around m 2 /2 (Figure 8c).
At the end, we see that the mean square overlap of the strongly represented is close to the characterized value κ 2 (Figure 8d), which scales as 1/m since the number of the strongly represented is much larger than m. b a exponential power law linear 1 0 1 Weight decay, 0.0 0.5 1.0 1.5 2.0 Data exponent, 0.0 0.2 0.4 0.6 0.8 1.0 R squared Figure 10: R squared values for fitting loss as a power law with model dimension. Data are from the small toy models with data dimension n = 1000.
In Figure 1, we set weight decay γ = -1 to have strong superposition and γ = 0.1 to have weak superposition. We compute R squared values from linear fits in log-log plots to quantify scaling behavior, assessing how closely the loss follows a power law to model dimension. We can see that at strong superposition, the losses are close to power laws, regardless of the underlying feature frequencies, yet the loss is a power law at weak superposition if the feature frequency p i is a power law with rank i (Figure 9a). For a systematic scan, we next set p i ∝ 1/i α and can vary the data exponent α to change how fast p i decays consistently. Assuming a power-law form for the final test loss, L ∝ 1/m αm , we extract the model exponent α m from the empirical fit. We fit the loss with a power law in all cases. The fitted α m reveals how fast losses decay, even in the regime where the loss should not be a power law. Roughly, three distinct patterns emerge: (1) under weak superposition (positive γ, yellow box in Figure 9b), α m is small, indicating slow loss decay; Figure 9a reports the R 2 values from the fitting, where the raw data comes from training large toy models (Appendix B.1). When we are fitting the loss in log-log as a line, we choose the linear part to fit. If the loss versus model dimension curve is obviously not a line, we fit the whole curve as a line and output the R 2 value as a measure of how non-linear it is. Specifically, we fit the last five points for the power-law decay feature case in the weak superposition regime (yellow data in Figure 1b). Other cases in the weak superposition regime are fitted to a line with all data. In the strong superposition regime, when feature frequency decreases as a power law or as a linear function, we fit the data as a line starting from the third point (yellow and green in Figure 1d).
And from the raw data of small toy models (Appendix B.2), we can fit the model exponent α m directly and plot it as a function of γ and α as in Figure 9b.
The fitting in Figure 9b does not care whether the loss versus model dimension curve is a power law or not. We provide the R squared values for the fitting here (Figure 10). The closer R squared values are to 1, the better the data can be thought to be a power law (a line in log-log plot). In the strong superposition regime, the R squared values suggest the data are close to be power-law. While in the weak superposition regime, data may not be power-law, especially when γ is too large. When α is smaller than 1, it is not a power law in theory. The R squared values are not too small since the loss decay is very slow, and a line in log-log plot is still a good approximation. When α > 1 and γ ≈ 1, the number of represented features can be smaller than m or even non-increasing. Too large weight decay still makes the configuration of the representation be in no superposition. However, it destroys some feature representations that can exist, making the configuration far from the ideal case where m features are represented. So, we may not see power laws when weight decay is too strong.
this section cite: []

Section: D.2 Figure 2
Figure 2 introduced the toy model and the concept of superposition without real data. The W matrix we used to show superposition in Figure 2c is obtained by optimizing the square of off-diagonal terms of the normalized W , i.e., each row is normalized to have norm 1 first.
this section cite: []

Section: D.3 Figure 3
In Figure 3, we reported results from the trained small toy models with data dimension n = 1000, whose detailed hyperparameters are in Appendix B.2.
We showed results at data exponent α = 1 in Figure 3. The results are obtained at m = 100, γ = -1 for panel a, and at different m and γ for panel b. We showed that the more frequent features tend to have larger norms or to be better represented. And the norm distribution is very bimodal. We here show that it is true that the norm is around 1 or 0 for various α and model sizes m and degrees of superposition (Figures 11 and 12). The fraction of represented, ϕ 1/2 , can be calculated directly.
Here, we provide the heat map of ϕ 1/2 at different m as a function of α and γ (Figure 13). The pattern is robust, suggesting weight decay is a good tool to change the degree of superposition regardless of data properties and model sizes.
this section cite: []

Section: D.4 Sparsity does not affect scaling behaviors in our tests
We studied the effect of the number of expected activated features or activation density E, which was set to 1. By fixing data exponent α = 1, which will be shown to be relevant to natural language, we can scan different superposition degrees and activation densities. Since p i ≤ 1 is required, which is equivalent to p 1 ≤ 1, we have E ≤ n i=1 1/i α , setting the upper bound for our scanning. We found that loss is approximately proportional to activation density E (Figure 14a). This fact suggests that the power law exponent should not change, which we confirmed (Figure 14b). Under a controlled superposition degree, activation density linearly increases loss and thus does not affect the scaling exponents in our experiments.
Once obtaining the small toy models scanning activation density and keeping α = 1, we can plot the loss as a function of activation density E in Figure 14. The linear fitting is also straightforward. We chose one γ to show in the main text. Here, we present the whole picture that, with any weight decay tested, the model exponent is robust to the change of activation density (Figure 15).   1 .1 2 ± 0 .0 6 1 .1 2 ± 0 .0 8 1 .1 ± 0 .1 1 .1 ± 0 .1 1 .1 ± 0 .2 a b Figure 14: Activation density does not affect scaling exponents in our tests. (a) Loss is roughly proportional to activation density given the degree of superposition (m = 63, n = 1000). (b) So, E will only affect the coefficient but not the exponent when considering the power law with model dimension. We plot the evidence α m ≈ 1 at strong superposition.
this section cite: []

Section: D.5 Figure 4
In Figure 4a, we plot the raw losses from small toy model experiments (hyperparameters in Appendix B.2). The theoretical value of loss is approximated by an integral
n nϕ 1/2 1/i α di, which is L =        ϕ 1-α 1/2 -1 1 -n 1-α n 1-α , α ̸ = 1, - ln ϕ 1/2 ln n , α = 1.(14)
To quantify how much the learned weight matrix deviates from the ideal no superposition structure, we construct a reference matrix and compute a norm difference. Specifically, we first create an n-by-n zero matrix called base, and then insert an identity matrix of size m in its top-left corner. This padded identity matrix serves as a reference for the perfect recovery of the first m features. We then compute the matrix product W W T from the learned weights and compare it to this reference using the matrix 2-norm. The resulting value reflects the ambiguity or interference in the learned representations. We store this norm in the ambiguity tensor at the location indexed by the current task and model width. Given a weight decay and data exponent, we have 6 ambiguity values since we have 6 m values. We calculate maximum ambiguity among these 6 models, and choose the 9 cases with the smallest maximum ambiguity to plot in Figure 4b. One can see that when weight decay is near 0.5, the models are closest to the ideal no superposition case where the first m features are represented perfectly. Smaller weight decay may not be sufficient to eliminate superposition, and larger weight decay can suppress features that, in principle, can be represented perfectly.
this section cite: []

Section: D.6 Figure 5
For convenience, based on the observation that the vector norms are bimodal around 1 in the strong superposition regime, we define strongly represented features as those that have ∥W i ∥ 2 > 1, which are more frequent and ETF-like, and weakly represented ones for those with ∥W i ∥ 2 < 1. We can quantify the fraction of strongly represented as
ϕ 1 = |{i : ∥W i ∥ 2 > 1}|/n,(15)
which significantly exceeds m/n and is around the ETF upper bound m 2 /2n (Figure 16). The group of vectors ∥W i ∥ 2 with norm greater than 1 or the strongly represented features then roughly agree with ETF properties: small variance, 1/m mean squared overlaps, and a limited number of vectors.
Figure 5 studies the results from small toy models (Appendix B.2) focusing on the strong superposition regime. For the strongly represented fraction, ϕ 1 , we can directly compute based on the definition and the obtained weight matrices. We showed a row norm distribution at m = 15, α = 1, and γ = -0.78 in Figure 5 panels a and b. Here, we provide more data to show that 1 is a natural separation point in norm to determine which are strongly represented and which are weakly represented (Figure 12). Once select the rows with norm greater than 1, we can calculate their mean and variance of squared overlaps based on normalized rows W i /∥W i ∥ 2 (Figure 5, c and d). We argue that after training, the vectors will be more similar to ETFs than to random initialization. This is studied via the variance of overlaps. ETFs, in theory, have zero variance. We find that the majority of the overlap variances are much smaller than the random initialization, especially when features have similar frequencies, which agrees with the expectation. The cases where the actual variance is greater than that of the random vectors have large α, roughly correspond to the cases where α m deviates from 1 -ETF-like configuration no longer dominates. This is intuitive that when α is too large, the heterogeneity of overlaps will become large -it is better to let more frequent features occupy larger angle space. We argue that the large variance at large α does not mean the configuration tends to be random, but tends to be something more closely related to the frequency distribution of the features.
Our explanations based on the strongly and weakly represented features capture the basic trend that when α is getting large, the more important features will have larger angle space and the loss decay will be more related to the data exponent. However, this theory is oversimplified, where the strongly represented all have small overlaps and the weakly represented all have large overlaps. The real situation may be more like the angle occupied by one feature decreases continuously as the frequency decreases. As suggested by Figure 5c, overlap variance within the strongly represented is greater when α is larger. To be more precise about the overlap distribution as well as the exponent α m when α is large, we cannot use simple theoretical expectations like ETFs but have to solve the toy model.
We also provide evidence that overlaps of all the vectors (Figure 17). We see that some of the mean square overlaps are larger than 1/m instead of being on the line 1/m. However, all the mean values follow 1/m scaling even for large α cases where the vectors are no longer isotropic. We emphasize that for even frequencies and isotropic vectors, since squared overlaps scale as 1/m, the loss should scale as 1/m.
After fitting α m of the trained small toy models (Appendix B.2), we plotted the α m corresponding to the second to the fourth smallest weight decays in Figure 5e. We also copied from Figure 4b and plotted the ideal weak superposition case in Figure 5e. One question we had is that if m 2 /2 is always greater than n, in our analysis, all vectors can be strongly represented, what should α m be? We trained the small toy models again as in Appendix B.2 but with m from 50 to 150. We found that in the strong superposition regime, the α m is still around 1 when α is smaller than 1.5, and α m still increases a little while smaller than 2(α -1) when α is larger than 1.5 (Figure 18). When m 2 /2n > 1 is always true, the vectors can be put into a configuration where all overlaps are small and scale as 1/m, such that α m should be closer to 1. However, as mentioned before, our picture that the strongly represented have nearly uniform absolute overlaps is oversimplified. In the real situation, more frequent features have smaller or even faster decaying overlaps. Therefore, when α is too large, a weighted sum of squared overlaps, weighting the more frequent features more, can decrease faster than the average decaying speed 1/m. Again, we need to solve the toy model faithfully to uncover the rigorous relation between α m and α and argue the robustness of α m from theory.
this section cite: []

Section: D.7 Figure 6
After obtaining the overlaps as described in Appendix C.1, we directly plot the raw data in Figure 6a. The data are quite noisy, and we did not fit the data with a line. We argued that the LLMs are in the strong superposition regime since all tokens are represented. Figure 19 shows a typical row norm distribution of LLM (opt, 125M parameters [39]). We showed the mean, minimum, and maximum row norms of all the LLMs studied in Figure 20. From the non-zero minimum norms and the fact n ≫ m, we confirm LLMs are in strong superposition. As mentioned in the analysis in Appendix A.2, we argue that the row norm of LLMs should not depend on m but controlled more by the intrinsic data property of language, which is also verified to be valid (Figure 20).
We obtain the evaluation loss of each model on each dataset as described in Appendix C.2. We fit our loss values by the formula, L = C m /m αm + L \m , where C m /m αm is universal and L \m is a constant depending on the dataset and model class. There are in total 16 different L \m since we have 4 different model classes and 4 datasets. In our fitting model, there are in total 18 parameters. We use Adam to minimize the mean square error between the predicted loss by the above function and the real loss. All losses obtained are used in optimization. The code is in nonlinearfit-3.ipynb.
We provide the raw data, losses, as a function of 1/m (Figure 21). The losses looks like a line with 1/m in one model class and with the same dataset. And the slope of the line seems to be universal. 3.0 opt Qwen gpt2 pythia Chinchilla a b 2.52±0.03 Figure 23: The model size is approximately a power law with model dimension. (a)
The four model classes we analyzed [39][40][41][42]. (b) The Chinchilla models [3].
coefficients depending on the model class, we obtain an exponent of 2.51 (Figure 23a). For the Chinchilla model reported in [3], we find N is also close to a power law with the model dimension, and the fitted exponent is 2.52 ± 0.03 (Figure 23b).
this section cite: ['b38', 'b38', 'b39', 'b40', 'b41', 'b2', 'b2']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claims match theoretical and experimental results.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We explained these details in our Appendices.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: We included error bars.
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: We provide this information in the Appendices.
Guidelines:
• The answer NA means that the paper does not include experiments.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We properly cited the datasets and LLMs analyzed.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described. wikitext c4 pile bookcorpus opt Qwen gpt2 pythia These two points support us in proposing the formula above, where C m /m αm is universal. The intersections are different depending on the dataset and model class, corresponding to different L \m .
We obtained the token frequencies as described in Appendix C.3. Given the raw data, we sort the token frequency and obtain the frequency-rank plot. We sample 1000 (this number does not matter once it is large, 10000 gives the same result) points uniformly in the log 10 (Rank), and fit the frequency-rank as a power law, or a line in log-log plot. Results show that the data exponent fitted α is close to 1 regardless of the dataset or the tokenizer (Figure 22).
We study the relationship between model dimension m and model size N (number of parameters).
For the four open-sourced models we analyzed [39][40][41][42], we can see that N ∼ m 3 , especially when m is large. If we fit the N -m relation by a power law while assuming a universal exponent but different
this section cite: ['b38', 'b39', 'b40', 'b41']

Section: References
Ref_id:b0 Title: Language models are few-shot learners Year: (2020)
Ref_id:b1 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b2 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b3 Title: Scaling laws for autoregressive generative modeling Year: (2020)
Ref_id:b4 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b5 Title: How good is google bard's visual understanding? an empirical study on open challenges Year: (2023)
Ref_id:b6 Title: Solving quantitative reasoning problems with language models Year: (2022)
Ref_id:b7 Title: Galactica: A large language model for science Year: (2022)
Ref_id:b8 Title: Wolfram|alpha as the computation engine for gpt models Year: (2023)
Ref_id:b9 Title: Solving olympiad geometry without human demonstrations Year: (2024)
Ref_id:b10 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b11 Title: Github copilot: Your ai pair programmer Year: (2022)
Ref_id:b12 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b13 Title: Scaling laws from the data manifold dimension Year: (2022)
Ref_id:b14 Title: Explaining neural scaling laws Year: (2024)
Ref_id:b15 Title: Spectrum dependent learning curves in kernel regression and wide neural networks Year: (2020)
Ref_id:b16 Title: How feature learning can improve neural scaling laws Year: (2025)
Ref_id:b17 Title: A solvable model of neural scaling laws Year: (2022)
Ref_id:b18 Title: Learning curve theory Year: (2021)
Ref_id:b19 Title: The quantization model of neural scaling Year: (2023)
Ref_id:b20 Title: Physics of skill learning Year: (2025)
Ref_id:b21 Title: Scaling laws and interpretability of learning from repeated data Year: (2022)
Ref_id:b22 Title: Neural scaling laws rooted in the data distribution Year: (2024)
Ref_id:b23 Title: Asymptotic learning curves of kernel methods: empirical data versus teacher-student paradigm Year: (2020)
Ref_id:b24 Title: A resource model for neural scaling law Year: (2024)
Ref_id:b25 Title: Linear algebraic structure of word senses, with applications to polysemy Year: (2018)
Ref_id:b26 Title: Toy models of superposition Year: (2022)
Ref_id:b27 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b28 Title: Lower bounds on the maximum cross correlation of signals (corresp.) Year: (2003)
Ref_id:b29 Title: Finite frames: Theory and applications Year: (2012)
Ref_id:b30 Title: Grassmannian frames with applications to coding and communication Year: (2003)
Ref_id:b31 Title: Steiner equiangular tight frames Year: (2012)
Ref_id:b32 Title: Symmetric informationally complete quantum measurements Year: (2004)
Ref_id:b33 Title: Relating measurement disturbance, information, and orthogonality Year: (2021-11)
Ref_id:b34 Title: Quantifying unsharpness of measurements via uncertainty Year: (2021-11)
Ref_id:b35 Title: Total, classical and quantum uncertainties generated by channels Year: (2022)
Ref_id:b36 Title: Prevalence of neural collapse during the terminal phase of deep learning training Year: (2020)
Ref_id:b37 Title: Neural collapse: A review on modelling principles and generalization Year: (2022)
Ref_id:b38 Title: Opt: Open pre-trained transformer language models Year: (2022)
Ref_id:b39 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b40 Title:  Year: (2024)
Ref_id:b41 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b42 Title: Pointer sentinel mixture models Year: (2016)
Ref_id:b43 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b44 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b45 Title: Aligning books and movies: Towards story-like visual explanations by watching movies and reading books Year: (2015)
Ref_id:b46 Title: Chinchilla scaling: A replication attempt Year: (2024)
Ref_id:b47 Title: Deep learning scaling is predictable, empirically Year: (2017)
Ref_id:b48 Title: Compressed sensing Year: (2006)
Ref_id:b49 Title: Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information Year: (2006)
Ref_id:b50 Title: Compressive sensing Year: (2007)
Ref_id:b51 Title: Compressed sensing, sparsity, and dimensionality in neuronal information processing and data analysis Year: (2012)
Ref_id:b52 Title: Statistical mechanics of optimal convex inference in high dimensions Year: (2016)
Ref_id:b53 Title: Emergence of simple-cell receptive field properties by learning a sparse code for natural images Year: (1996)
Ref_id:b54 Title: Sparseness and expansion in sensory representations Year: (2014)
Ref_id:b55 Title: Mathematical models of computation in superposition Year: (2024)
Ref_id:b56 Title: On the complexity of neural computation in superposition Year: (2024)
Ref_id:b57 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b58 Title: The depth-to-width interplay in self-attention Year: (2020)
Ref_id:b59 Title: Simeng Sun, and Boris Ginsburg. ngpt: Normalized transformer with representation learning on the hypersphere Year: (2024)
Ref_id:b60 Title: Focus: First order concentrated updating scheme Year: (2025)
Ref_id:b61 Title: Mechanistic interpretability for ai safety-a review Year: (2024)
Ref_id:b62 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
