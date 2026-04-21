Title: ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs
Abstract: Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs n. Recently, Kang et al. (2025)  proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to n ≈ 10 3 features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical-higher-order interactions are accompanied by their lower-order subsets-which enables more efficient discovery. To exploit this hierarchy, we propose PROXYSPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that PROXYSPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using 10× fewer inferences than SPEX. By accounting for interactions, PROXYSPEX efficiently identifies the most influential features, providing a scalable approximation of their Shapley values. Further, we apply PROXYSPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task. The PROXYSPEX algorithm is available at https://github.com/mmschlk/shapiq.

Section: Introduction
Large language models (LLMs) have achieved great success in natural language processing by capturing complex interactions among input features. Modeling interactions is not only crucial for language, but also in domains such as computational biology, drug discovery and healthcare, which require reasoning over high-dimensional data. In high-stakes contexts, responsible decision-making based on model outputs requires interpretability. For example, in healthcare, a physician relying on LLM diagnostic assistance must intelligibly be able to explain their decision to a patient.
Post-hoc feature explanation methods such as SHAP [1] and LIME [2] focus on marginal attributions and do not explicitly capture the effect of interactions. To address this limitation, recent work  Image Captioning (CLIP) LASSO ProxySPEX (Ours) SPEX Figure 1: PROXYSPEX requires ∼10× fewer inferences to achieve equally faithful explanations as SPEX for a sentiment classification and image-captioning task using a BERT and CLIP model respectively. LASSO faithfulness plateaus indicating limits of marginal approaches.
has proposed interaction indices, such as Faith-Shap [3], that attribute all interactions up to a given order d by exhaustively enumerating them. With n features, enumerating O(n d ) interactions quickly becomes infeasible for even small n and d. Kang et al. [4] recently introduced SPEX, the first interaction attribution method capable of scaling up to n = 1000 features. SPEX scales with n by observing that LLM outputs are driven by a small number of interactions. It exploits this sparsity by utilizing a sparse Fourier transform to efficiently search for influential interactions without enumeration. For example, with n = 100 features, SPEX requires approximately 2 × 10 4 model inferences to learn order 5 interactions-a small fraction of all possible 100 5 interactions. Nonetheless, 2 × 10 4 inferences is prohibitively expensive for large models. Hence, the question naturally arises: Can we identify additional structural properties among interactions to improve inference-efficiency?
We show empirically that local (i.e., input specific) LLM feature interactions are often hierarchical: for an order d interaction, an LLM includes lower-order interactions involving subsets of those d features (see Figure 2). We use this to develop PROXYSPEX, an interaction attribution algorithm that reduces the number of inferences compared to SPEX by 10× while achieving equally faithful explanations. PROXYSPEX exploits this local hierarchical structure by first fitting gradient boosted trees (GBTs) as a proxy model to predict the output of LLMs on masked input sequences. Then, PROXYSPEX extracts important interactions from the fitted GBTs [5].
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4']

Section: Evaluation overview.
We compare PROXYSPEX to marginal feature attributions and SPEX across four high-dimensional datasets with hundreds of features. Results are summarized below:
1. Faithfulness. PROXYSPEX learns more faithful representations of LLM outputs than marginal approaches (≈15% to 25%) on average across datasets as we vary the number of inferences. Figure 1 compares explanation faithfulness of PROXYSPEX to marginal attributions and SPEX. 2. Feature identification. By accounting for interactions, PROXYSPEX identifies influential features that impact model outputs more significantly than marginal approaches, and can approximate Shapley values better than KernalSHAP in the low-inference regime.
this section cite: []

Section: Case study 1: Data attribution.
Data Attribution is the problem of identifying training points responsible for a given test prediction. On CIFAR- 10 [6] PROXYSPEX identifies the interactions between training samples that most significantly impact classification performance. 4. Case study 2: Model component attribution. We use PROXYSPEX to study interactions between attention heads, both within and across layers, on MMLU [7] for Llama-3.1-8B-Instruct [8]. We observe that intra-layer interactions become more significant for deeper layers. PROX-YSPEX identifies interactions that allow it to prune more heads than the LASSO.
this section cite: ['b6', 'b7']

Section: Related work and applications
Feature and interaction attribution. SHAP [1] and LIME [2] are widely used for model-agnostic feature attribution. SHAP uses the game-theoretic concept of Shapley values [9] for feature attribution, while LIME fits a sparse linear model [10]. Cohen-Wang et al. [11] also consider fitting a sparse linear model for feature attribution. Chen et al. [12] uses an information-theoretic approach for feature attributions. Other methods [13,14] study model structure to derive feature attributions. Sundararajan et al. [15] and Bordt and von Luxburg [16] define extensions to Shapley values that consider interactions. Fumagalli et al. [17] provides a framework for computing several interaction attribution scores, but their approach does not scale past n ≈ 20 features, which prevents them from being applied to modern ML problems that often consist of hundreds of features. Note that some feature attribution approaches such as LIME and Faith SHAP [3] are formulated explicitly as a function approximation, while others are defined axiomatically such as SHAP, though one can typically construct equivalent function approximation objectives with a suitable distance metric.
Fourier transforms and deep learning explainability. Several works theoretically study the spectral properties of transformers. Ren et al. [18] show transformers have sparse spectra and Hahn and Rofin [19], Abbe et al. [20] establish that they are low degree. Abbe et al. [21,22] study the bias of networks learning interactions via a "staircase" property, i.e., using lower-order terms to learn high-order interactions. Sparsity and low degree structure is also empirically studied in [23,24]. Kang et al. [25] shows that under sparsity in the Möbius basis [26], a representation closely related to Shapley values and the Fourier transform, interaction attributions can be computed efficiently. Mohammadi et al. [27] also learn a sparse Möbius representation for computing Shapley values. Kang et al. [4] use these insights to propose SPEX, the first robust interaction attribution algorithm to scale to the order of n ≈ 1000 features. Gorji et al. [5] apply sparse Fourier transforms [28][29][30][31] for computing Shapley values. They also provide an algorithm to extract the Fourier transform of tree-based models using a single forward pass.
this section cite: ['b0', 'b1', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b2', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b3', 'b4', 'b27', 'b28', 'b29', 'b30']

Section: SPEX.
We refer to the algorithm proposed in this manuscript as PROXYSPEX, in reference to SPEX, since both works exploit a sparse interaction prior to reduce computational and sample budget. SPEX uses an algebraic structured sampling scheme, coupled with error correction decoding procedures to efficiently compute the interactions in the form of a Fourier transform. In contrast, PROXYSPEX uses random samples to learn a proxy model that implicitly exploits the sparse interaction priors and our newly proposed hierarchical prior.
this section cite: []

Section: Mechanistic Interpretability (MI).
MI seeks to uncover the underlying mechanisms of neural networks and transformers [32] in order to move past treating these models as black boxes. PROX-YSPEX answers the question "what combinations of inputs matter?" which is a vital precursor and complement to MI investigations that subsequently address "how does the model compute based on those specific inputs?" Some closely related MI work attempts to recover circuits to explain underlying model behavior [33,34]. Hsu et al. [35] use MI for interaction attribution. See Sharkey et al. [36] for a review of open problems and recent progress in MI.
this section cite: ['b31', 'b32', 'b33', 'b34', 'b35']

Section: PROXYSPEX
In this section, we first empirically justify our premise that significant interactions affecting LLM output are hierarchical-influential high-order interactions imply important lower-order ones. Next, we introduce PROXYSPEX, which aims to identify feature interactions for a given input x while minimizing the number of expensive calls to an LLM.
this section cite: []

Section: Preliminaries
Value function. Let x be the input to the LLM consisting of n featuresfoot_0 . For S ⊆ [n], where [n] = 1, . . . , n, denote x S as the masked input where we retain features indexed in S and replace all others with the [MASK] token. For example, in the sentence x ="The sequel truly elevated the original", if S = {1, 2, 5, 6}, x S = "The sequel [MASK] [MASK] the original". Masks can be more generally applied to any type of input such as image patches in a vision-language model. For a masked input x S and LLM f , let f (x S ) ∈ R denote the output of the LLM under masking pattern S. The value function f is problem dependent. For classification tasks, a common choice is the logit of the predicted class for unmasked input, f (x). In generative tasks, f (x S ) can represent the perplexity of generating the original output for the unmasked input. Since we focus on providing input-specific explanations, we suppress notation on x and denote f (x S ) as f (S).
Fourier transform of value function. Let 2 [n] be the powerset of the index set. The value function f can be equivalently thought of as a set function from f : 2 [n] → R. Every such function admits a
x 1 x 2 x 3 x 1 x 2 x 1 x 3 x 2 x 3 x 1 x 2 x 3 x 4 x 5 x 3 x 4 x 3 x 5 x 4 x 5 x 3 x 4 x 5
Figure 2: We observe that LLM feature interactions are often hierarchicalhigher-order interactions are accompanied by their lower-order subsets.
Fourier transform F : 2 [n] → R of f , related as follows:
Transform: F (T ) = 1 2 n S⊆[n] (-1) |S∩T | f (S), Inverse: f (S) = T ⊆[n] (-1) |T ∩S| F (T ). (1
)
The parameters F (T ) are known as Fourier coefficients and capture the importance of an interaction of features in a subset T . Equation (1) represents an orthonormal transform onto a parity (XOR) basis [37]. For the rest of the paper, we use the terms Fourier coefficient and interaction interchangeably. Further, we refer to the set of Fourier coefficients {(T, F (T )) : T ⊆ [n]} as the spectrum.
Interpretable approximation of value function. We aim to learn an interpretable approximate function f that satisfies the following:
1. Faithful representation. To characterize how well the surrogate function f approximates the true function, we define faithfulness [38]:
R 2 = 1 - f -f 2 f -f 2 , where f 2 = S⊆[n] f (S) 2 , f = 1 2 n S⊆[n] f (S).(2)
Faithfulness measures how well f predicts model output. High faithfulness implies accurate approximation of F (T ) (this follows from orthonormality of (1)).
2. Sparse representation. f should be succinct. Previous works [4,25,[39][40][41] have shown that a sparse and low-degree f can achieve high R 2 . That is, F (T ) ≈ 0 for most T (sparsity), and |F (T )| is only large when |T | n (low degree). 3. Efficient computation. Without any additional assumptions on the spectrum, learning f is exponentially hard since there are 2 n possible subsets T . PROXYSPEX relies on the sparse, low degree Fourier transform along with the hierarchy property to reduce LLM inferences.
A faithful and sparse f allows straightforward computation of all popular feature or interaction attribution scores defined in the literature, e.g., Shapley, Banzhaf, Influence Scores, Faith-Shapley. Closed-form formulas for converting F to various attribution indices are provided in Appendix A.1.
this section cite: ['b36', 'b37', 'b3', 'b24', 'b38', 'b39', 'b40']

Section: Empirical evidence of spectral hierarchies
To quantify the degree of hierarchical structure in LLMs, we introduce the following definition called Direct Subset Rate (DSR),foot_1 defined for any value function f and integer k.
DSR(f, k) = 1 k S∈F k 1 |S| i∈S {S \ {i} ∈ F k } , where F k denotes the k largest Fourier coefficients of f . (3
)
For the top k coefficients (i.e., interactions), DSR measures the average fraction of Fourier coefficients that exclude only one of the features F (S \ {i}). For example, an f with F 4 = {∅, {1}, {2}, {1, 3}} would have DSR of 1 4 1 + 1 + 1 + 1 2 = 7 8 . High DSR implies that significant high-order interactions have corresponding significant lower-order Fourier coefficients, as visualized in Figure 2. Next, we show that two LLM based value functions have high DSR.
We take 20 samples from a sentiment analysis task and an image captioning task [42]; see Section 4 for a detailed description and our choice of value function. We generate masks S and apply SPEX until our learned value function has faithfulness (R 2 ) more than 0.9. Figure 3 visualizes the DSR 8 16 32 64 128 256 Top k Coefficients 0% 20% 40% 60% 80% 100% Direct Subset Rate Sentiment Analysis (BERT) 8 16 32 64 128 256 Top k Coefficients 0% 20% 40% 60% 80% 100% Direct Subset Rate Image Captioning (CLIP)
Figure 3: The top-k interactions in both a sentiment analysis and image captioning task have high DSR indicating strong hierarchical structure.
for various values of k, i.e., number of top interactions. DSR is consistently larger than 80%, indicating strong hierarchical structure. In Appendix B.2, we consider two additional metrics measuring hierarchical structure, and demonstrate that the top-k interactions are faithful.
Using GBTs to capture hierarchical Interactions. Tan et al. [43] proved that decision trees learn "staircase" functions, e.g., f = x 1 + x 1 x 2 + x 1 x 2 x 3 , effectively due to their greedy construction procedure. We empirically confirm this by comparing the performance of various proxy models on a synthetic hierarchical function (i.e., sum of staircase functions resembling Figure 2) as well as the Sentiment dataset in Appendix Figure 13. Appendix B.4 details the simulation set-up. GBTs vastly outperform other proxy models, indicating their natural ability to identify hierarchical interactions with limited training data. Interestingly, GBTs outperform random forests as well. This is because random forests are ineffective at learning hierarchical functions [44], i.e., sums of staircases, while GBT-like algorithms disentangle sums effectively [45].
this section cite: ['b41', 'b42', 'b43', 'b44']

Section: PROXYSPEX via Gradient Boosted Trees to fit hierarchies
The PROXYSPEX algorithm (see Figure 4): 2 4 6 8 Inference Multiplier (α) 0.0 0.2 0.4 0.6 0.8 Faithfulness (R 2 ) +28% +31% +31% +31% Sentiment Analysis 2 4 6 8 Inference Multiplier (α) 0.0 0.2 0.4 0.6 Faithfulness (R 2 ) +14% +16% +18% +19% DROP 2 4 6 8 Inference Multiplier (α) 0.0 0.1 0.2 0.3 Faithfulness (R 2 ) +11% +16% +21% +26% HotpotQA 2 4 6 8 Inference Multiplier (α) 0.0 0.2 0.4 0.6 0.8 Faithfulness (R 2 ) +4% +14% +18% +20% 10 0 10 1 10 2 10 3 10 4 Sparsity (k) 0.0 0.2 0.4 0.6 0.8 1.0 Rel. Faithfulness Captioning Sentiment Figure 5: Relative faithfulness as a function of Fourier sparsity. Only ≈ 200 coefficients are required to achieve equivalent faithfulness. Sparsity for sentiment is higher since inputs have larger n.
this section cite: []

Section: MS-COCO

this section cite: []

Section: LASSO ProxySPEX (Ours) SPEX

this section cite: []

Section: Step 2 -Proxy Training.
Fit GBTs to D with 5-fold cross-validation (CV).
this section cite: []

Section: Step 3 -Fourier extraction.
We use Gorji et al. [5] to extract the Fourier representation of the fitted GBTs in a single forward pass; see Appendix A.2. With T trees of depth d there are at most O(T 4 d ) non-zero Fourier coefficients [5].
To improve interpretability, we sparsify the extracted representation by keeping only the top k Fourier coefficients. Fig. 5 shows that only ≈ 200 Fourier coefficients are needed to achieve equivalent faithfulness for a sentiment classification and image captioning (MS-COCO) dataset. Additional results regarding the sparsity of Fourier spectra learned by GBTs are in Appendix B.3.
this section cite: ['b4', 'b4']

Section: Step 4 (Optional): Coefficient refinement via regression.
As a final step, we optionally regress the extracted, top k Fourier coefficients on the collected data D to improve the estimation. Empirically we observe this step is can sometimes marginally improve performance, but seldom negatively impacts performance. This step is included if it leads to lower CV error.
this section cite: []

Section: Results

this section cite: []

Section: Datasets and models
1. Sentiment is a classification task composed of the Large Movie Review Dataset [46] which consists of positive and negative IMDb movie reviews. We use words as input features and restrict to samples with n ∈ [256, 512]. We use the encoder-only fine-tuned DistilBERT model [47,48], and the logit of the positive class as the value function. 2. HotpotQA [49] is a generative question-answering task over Wikipedia articles. Sentences are input features, and we restrict to samples with n ∈ [64, 128]. We use Llama-3.1-8B-Instruct, and perplexity of the unmasked output as the value function.
3 4 5 6 7 # Features Removed (r) 0.8 0.9 1.0 1.1 1.2 ∆ LLM Output Sentiment Analysis 3 4 5 6 7 # Features Removed (r) 4.5 5.0 5.5 6.0 6.5 7.0 ∆ LLM Output DROP 3 4 5 6 7 # Features Removed (r) 0.05 0.10 0.15 0.20 0.25 ∆ LLM Output HotpotQA 3 4 5 6 7 # Features Removed (r) 0.20 0.25 0.30 0.35 ∆ LLM Output
this section cite: ['b45', 'b46', 'b47', 'b48']

Section: MS-COCO

this section cite: []

Section: LASSO ProxySPEX (Ours) SPEX
Figure 7: By accounting for interactions, PROXYSPEX identifies more influential features across datasets than the LASSO. Apart from the sentiment analysis task (top left), SPEX does not collect enough training masks to out-perform LASSO.
3. Discrete Reasoning Over Paragraphs (DROP) [50] is a paragraph level question-answering task. We use words as input features and restrict to samples with n ∈ [256, 512]. We use Llama-3-8B-Instruct and the perplexity of the unmasked output as the value function. 4. MS-COCO [42] contains images and corresponding text captions. Image patches and words are the input features with n ∈ [60,85]. We use CLIP-ViT-B/32, a joint vision-language encoder, with the value function defined as the contrastive loss over all datapoints.
Baselines and hyperparameters. For marginal feature attributions, we use the LASSO. We use the same datasets at [4] and add MS-COCO for an additional modality. It was shown in [4] that popular marginal metrics such as SHAP are significantly less faithful than the LASSO, e.g., have R 2 < 0. We use the LASSO implementation from scikit-learn, and choose the l 1 regularization parameter via 5-fold CV. For interaction indices, we compare PROXYSPEX to SPEX. Due to the scale of n in our experiments, we cannot compare methods for computing interaction indices such as Faith-Shapley, Faith-Banzhaf, and Shapley-Taylor using SHAP-IQ [17], and SVARM-IQ [51], because they enumerate all possible interactions, making them computationally infeasible. For PROXYSPEX, a list of GBT hyper-parameters we tune over are in Appendix B.
this section cite: ['b49', 'b41', 'b59', 'b3', 'b3', 'b16', 'b50']

Section: Faithfulness
We compare attribution method faithfulness by varying the number of training masks. For each sample with n features, we generate α • n log 2 (n) masks, varying α ∈ {2, 4, 6, 8}, to normalize difficulty across inputs of varying lengths (some by over 100 tokens). This n log(n) type scaling is heuristically guided by compressed sensing bounds [52]. These suggest the number of samples required grows with sparsity (assumed ∝ n) and logarithmically with problem dimensionality (if dimensionality for degree-d interactions is ≈ n d , this yields a log(n d ) = d log(n) factor). Together, these factors support an n log(n) scaling. While not directly applicable, these bounds offer a useful heuristic for how sampling complexity scales with n.
this section cite: ['b51']

Section: Feature Identification
We measure the ability of methods to identify the top r features influencing LLM outputs:
∆ LLM Output (r) = |f ([n]) -f (S * )| |f ([n])| , S * = argmax |S|=n-r | f ([n]) -f (S)|.(4)
Solving Eq. 4 for an arbitrary f presents a challenging combinatorial optimization problem. However, PROXYSPEX and SPEX represent f as a sparse Fourier transform. This representation facilitates solving the optimization as a tractable linear integer program. The sparsity of the extracted Fourier representation ensures that the time required to solve this program is negligible compared to sampling the LLM and fitting the GBTs. Full details of the construction of this program are given in Appendix A.3. Under LASSO, Eq. 4 is easily solved through selecting features by the size of their coefficients. We measure the removal ability of different attribution methods when we collect 8n log 2 (n) training masks and plot the result in Figure 7. By accounting for interactions, PROX-YSPEX identifies significantly more influential features than the LASSO. Apart from the sentiment analysis task, SPEX does not collect enough training masks to outperform the LASSO.
this section cite: []

Section: Shapley Value Approximation
PROXYSPEX can be directly used to approximate Shapley values. Across all tasks, we first run KernelSHAP with 10,000 test masks and treat these approximated Shapley values as ground truth. We measure the recall of the top ten highest-magnitude Shapley values for KernelSHAP and PROX-YSPEX under α•n log 2 (n) inferences with multipliers α ∈ {0.25, 0.5, 0.75, 1.0}. For this inference budget, competing algorithms such as LeverageSHAP [53] and SVARM [54] struggle to provide accurate approximations. We find PROXYSPEX initially provides a better coarse approximation than KernelSHAP (Figure 14). However, since PROXYSPEX is optimized for faithfulness and does not rely on the Shapley kernel, it is eventually surpassed by KernelSHAP with enough inferences. Additional results under mean squared error are included in Appendix B.5.
this section cite: ['b52', 'b53']

Section: Case studies
We now present two case studies of PROXYSPEX for two different interpretability problems: data attribution [55] and model component attribution [56], a key problem in mechanistic interpretability. We first show how both of these tasks can be reformulated as feature attribution tasks; recent work has highlighted the connections between feature, data, and model component attribution [57].
Airplane Horse Frog Bird Bird Bird Dog Dog Dog Bird Bird Bird Bird Dog Automobile Bird Truck Airplane Horse Cat Cat Automobile Truck Truck
this section cite: ['b54', 'b55', 'b56']

Section: Heldout Image Redundant Interactions Heldout Image Synergistic Interactions
Figure 9: Synergistic interactions: data that together are more valuable together than the sum of their parts and aid in classification. Redundant interactions: Data that may contain similar information, their combined influence is less than the sum of the parts.
this section cite: []

Section: Data Attribution via Non-Linear Datamodels
Data attribution for classification is the problem of understanding how fitting a model g θ on a subset S of training samples affects the prediction of a test point z of class c. This problem can be converted into our framework by defining an appropriate value function f , f (S) ≜ (logit for c on z) -(highest incorrect logit on z), when g θ is trained on S.
(
)5
The value function f quantifies the impact of a subset S on the classification of z. Sampling f is very expensive since it involves training a new model g θ for every subset S. As a result, most data attribution approaches do not consider the impact of interactions. Notably, Ilyas et al. [55] use LASSO to learn f when training a ResNet model on the CIFAR-10 dataset [6]. As a case study, we apply PROXYSPEX to understand the impact of interactions between CIFAR-10 training samples.
this section cite: ['b54', 'b5']

Section: Defining data interactions.
Interactions between samples can be either redundant interactions or synergistic interactions. Redundant interactions are when the influence of a subset S is not additive. Redundancy typically occurs between highly correlated samples, e.g., semantic duplicates [58]. Synergistic interactions occur when a subset S influences a prediction by shaping a decision boundary that no individual sample in S could do so by itself. That is, the model needs the combined effect of training samples in S to correctly classify z.
Results. We visualize interactions learned by PROXYSPEX in Figure 9 for randomly selected CIFAR-10 test points. Experimental details are in Appendix C.1. PROXYSPEX identifies highly similar training samples (redundancies) as well as synergistic interactions between samples of different classes. See Appendix C.1 for examples of other randomly selected test samples.
this section cite: ['b57']

Section: Model Component Attribution
We study the role of attention heads for a question-answering task using Llama-3.1-8B-Instruct and MMLU (high-school-us-history), which is a multiple-choice dataset. We treat each attention head as a feature and aim to identify interactions among heads using PROXYSPEX. Let L represent the number of layers in an LLM and let L ⊆ [L] represent a subset of the layers. Let H L denote the set of attention heads within these layers. For a subset of heads S ⊆ H L , we set the output of heads in H L \ S to 0 and denote the ablated LLM as LLM S (•). Define f as:
f L (S) ≜ Accuracy of LLM S on training set of MMLU. (6
)
Pruning results. We use the LASSO and PROXYSPEX to identify the most important heads for various sparsity levels ( i.e., the number of retained heads) across different sets of layers. We also compare to a Best-of-N baseline, where we take the best of N = 5000 different randomly chosen S, further details are in Appendix C.2. We use the procedure detailed in Section 4.2 to identify heads
LASSO ProxySPEX (Ours) Best-of-N Unpruned Acc. 50% 66% 83% 0.7 0.8 0.9 +11% +2% +4% Layers 1-3 50% 66% 83% 0.7 0.8 0.9 +6% +1% +5% Layers 14-16 50% 66% 83% 0.7 0.8 0.9 +1% +3% +3% Layers 30-32 80% 8% 12% 67% 23% 10% 51% 39% 10% Test Accuracy % of Attention Heads Kept % of ProxySPEX's Spectral Energy 50% 66% 83% 0.7 0.8 0.9 +11% +2% +4% Layers 1-3 50% 66% 83% 0.7 0.8 0.9 +6% +1% +5% Layers 14-16 50% 66% 83% 0.7 0.8 0.9 +1% +3% +3% Layers 30-32 80% 8% 12% 67% 23% 10% 51% 39% 10% Test Accuracy % of Attention Heads Kept % of ProxySPEX's Spectral Energy Linear Within-Layer Inter. Across-Layer Inter. to remove for both PROXYSPEX and LASSO. Test accuracies for each method are presented in Figure 10 at three different sparsity levels, and with three different layer ranges: initial (1-3), middle (14-16) and final (30)(31)(32). We observe that PROXYSPEX consistently outperforms both baselines, with a higher test accuracy on the pruned models identified using PROXYSPEX.
this section cite: ['b29', 'b30', 'b31']

Section: Characterizing interactions between attention heads.
Analyzing the Fourier spectrum learned by PROXYSPEX offers insights into the nature of the internal mechanisms of the LLM. As shown in Figure 10 (bottom), the spectral energy attributed to interactions, particularly within-layer interactions, markedly increases in deeper layers of Llama-3.1-8B-Instruct. There are many works that look at the differing functional roles of attention heads across layers [59]. PROXYSPEX provides an exciting new quantitative approach to further investigate these phenomena.
this section cite: ['b58']

Section: Discussion
Conclusion. We introduce PROXYSPEX, an inference-efficient interaction attribution algorithm that efficiently scales with n by leveraging an observed hierarchical structure among significant interactions in the Fourier spectrum of the model. Experiments across 4 high-dimensional datasets show that PROXYSPEX exploits hierarchical interactions via a GBT proxy model to reduce inferences by ∼10× over SPEX [4] while achieving equally faithful explanations. Through applications to data and model component attribution, we demonstrate the importance of efficient interaction discovery.
Limitations. GBTs effectively capture hierarchical interactions but may not perform as well when interactions have a different structure. For example, simulations in Appendix B.4 empirically confirm that GBTs suffer in the case of sparse but non-hierarchical functions. More generally, in cases where the proxy GBT model is not faithful, the interactions identified by PROXYSPEX might not be representative of the model's reasoning. Another limitation is the degree of human interpretability that can be understood from computed interactions. While interactions can offer richer insights, they are more difficult to parse than marginal alternatives. Further improvements in visualization and post-processing of interactions are needed to fully harness the advances of PROXYSPEX.
Future work. Inference-efficiency could be further improved by exploring alternative proxy models, additional Fourier spectral structures, or adaptive masking pattern designs. Integrating PROX-YSPEX with internal model details, such as via hybrid approaches with MI or by studying its connection to sparsity in transformer attention [60], offers another promising avenue. Finally, further deepening and improving applications of PROXYSPEX in data attribution and mechanistic interpretability as well as potentially exploring more complex value functions or larger-scale component interactions remains interesting future work.
this section cite: ['b3', 'b59']

Section: References
Ref_id:b0 Title: A unified approach to interpreting model predictions Year: (2017)
Ref_id:b1 Title: Explaining the predictions of any classifier Year: (2016)
Ref_id:b2 Title: Faith-Shap: The faithful Shapley interaction index Year: (2023)
Ref_id:b3 Title: SPEX: Scaling Feature Interaction Explanations for LLMs Year: ()
Ref_id:b4 Title: SHAP values via sparse Fourier representation Year: (2025)
Ref_id:b5 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b6 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b7 Title: The llama 3 herd of models Year: (2024)
Ref_id:b8 Title: A Value for N-Person Games Year: (1952)
Ref_id:b9 Title: Regression shrinkage and selection via the LASSO Year: (1996)
Ref_id:b10 Title: ContextCite: Attributing model generation to context Year: (2024)
Ref_id:b11 Title: Learning to explain: An informationtheoretic perspective on model interpretation Year: (2018)
Ref_id:b12 Title: Axiomatic attribution for deep networks Year: (2017)
Ref_id:b13 Title: Layer-wise relevance propagation for neural networks with local renormalization layers Year: (2016)
Ref_id:b14 Title: The Shapley Taylor interaction index Year: (2020-07)
Ref_id:b15 Title: From Shapley values to generalized additive models and back Year: (2023)
Ref_id:b16 Title: SHAP-IQ: Unified approximation of any-order Shapley interactions Year: (2023)
Ref_id:b17 Title: Where we have arrived in proving the emergence of sparse interaction primitives in DNNs Year: ()
Ref_id:b18 Title: Why are sensitive functions hard for transformers? Year: (2024-12)
Ref_id:b19 Title: Generalization on the unseen, logic reasoning and degree curriculum Year: (2024)
Ref_id:b20 Title: The staircase property: How hierarchical structure can guide deep learning Year: (2021)
Ref_id:b21 Title: The merged-staircase property: A necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks Year: (2022)
Ref_id:b22 Title: On recovering higher-order interactions from protein language models Year: (2024)
Ref_id:b23 Title: Can we faithfully represent absence states to compute Shapley values on a DNN? Year: ()
Ref_id:b24 Title: Learning to understand: Identifying interactions via the Möbius transform Year: (2024)
Ref_id:b25 Title: A bargaining model for the cooperative n-person game Year: (1958)
Ref_id:b26 Title: Unlocking the game: Estimating games in Möbius representation for explanation and high-order interaction detection Year: (2025-04)
Ref_id:b27 Title: The SPRIGHT algorithm for robust sparse Hadamard transforms Year: (2014)
Ref_id:b28 Title: Efficiently learning Fourier sparse set functions Year: (2019)
Ref_id:b29 Title: Efficiently computing sparse Fourier transforms of q-ary functions Year: (2023)
Ref_id:b30 Title: A fast Hadamard transform for signals with sublinear sparsity in the transform domain Year: (2015)
Ref_id:b31 Title: Zoom in: An introduction to circuits Year: (2020)
Ref_id:b32 Title: Towards automated circuit discovery for mechanistic interpretability Year: (2023)
Ref_id:b33 Title: Attribution patching outperforms automated circuit discovery Year: (2024-11)
Ref_id:b34 Title: Efficient automated circuit discovery in transformers using contextual decomposition Year: ()
Ref_id:b35 Title: Open problems in mechanistic interpretability Year: (2025)
Ref_id:b36 Title: Analysis of Boolean functions Year: (2014)
Ref_id:b37 Title: Trade-off between efficiency and consistency for removal-based explanations Year: (2023)
Ref_id:b38 Title: Deep learning generalizes because the parameter-function map is biased towards simple functions Year: (2019)
Ref_id:b39 Title: A fine-grained spectral perspective on neural networks Year: (2020)
Ref_id:b40 Title: Towards the dynamics of a dnn learning symbolic interactions Year: (2024)
Ref_id:b41 Title: Microsoft COCO: Common objects in context Year: (2014)
Ref_id:b42 Title: Statistical-computational trade-offs for recursive adaptive partitioning estimators Year: (2025)
Ref_id:b43 Title: A cautionary tale on fitting decision trees to data from additive models: generalization lower bounds Year: (2022-03)
Ref_id:b44 Title: Fast interpretable greedy-tree sums Year: ()
Ref_id:b45 Title: Learning word vectors for sentiment analysis Year: (2011-06)
Ref_id:b46 Title: Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter Year: (2020)
Ref_id:b47 Title: DistilBERT Finetuned Sentiment Year: (2025-01)
Ref_id:b48 Title: Hot-potQA: A dataset for diverse, explainable multi-hop question answering Year: (2018)
Ref_id:b49 Title: DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs Year: (2019)
Ref_id:b50 Title: SVARM-IQ: Efficient approximation of any-order Shapley interactions through stratification Year: (2024)
Ref_id:b51 Title: Decoding by linear programming Year: (2005)
Ref_id:b52 Title: Provably accurate Shapley value estimation via leverage score sampling Year: ()
Ref_id:b53 Title: Approximating the Shapley value without marginal contributions Year: (2024)
Ref_id:b54 Title: Datamodels: Understanding predictions with data and data with predictions Year: (2022)
Ref_id:b55 Title: Decomposing and editing predictions by modeling model computation Year: (2024)
Ref_id:b56 Title: Building bridges, not walls: Advancing interpretability by unifying feature, data, and model component attribution Year: (2025)
Ref_id:b57 Title: SemDeDup: Dataefficient learning at web-scale through semantic deduplication Year: (2023)
Ref_id:b58 Title: Route sparse autoencoder to interpret large language models Year: (2025)
Ref_id:b59 Title: Scatterbrain: Unifying sparse and low-rank attention Year: (2021)
Ref_id:b60 Title: Technical note: Defining and quantifying AND-OR interactions for faithful and concise explanation of DNNs Year: (2024)
Ref_id:b61 Title: Learning decision trees using the Fourier spectrum Year: (1991)
Ref_id:b62 Title: Learning Boolean functions via the Fourier transform Year: (1994)
