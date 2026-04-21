Title: AXBENCH: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
Abstract: Fine-grained steering of language model outputs is essential for safety and reliability. Prompting and finetuning are widely used to achieve these goals, but interpretability researchers have proposed a variety of representation-based techniques as well, including sparse autoencoders (SAEs), linear artificial tomography, supervised steering vectors, linear probes, and representation finetuning. At present, there is no benchmark for making direct comparisons between these proposals. Therefore, we introduce AXBENCH, a large-scale benchmark for steering and concept detection, and report experiments on Gemma-2-2B and 9B. For steering, we find that prompting outperforms all existing methods, followed by finetuning. For concept detection, representation-based methods such as difference-in-means, perform the best. On both evaluations, SAEs are not competitive. We introduce a novel weakly-supervised representational method (Rank-1 Representation Finetuning; ReFT-r1), which is competitive on both tasks while providing the interpretability advantages that prompting lacks. Along with AXBENCH, we train and publicly release SAE-scale feature dictionaries for ReFT-r1 and DiffMean.github.com/stanfordnlp/axbench

Section: Introduction
In order to be useful, language models (LMs) must follow user instructions and be aligned to human goals and  values. While prompting and finetuning are now widely used to instill such behaviour in LMs, both methods have limitations: circumvention via jailbreaks and continued training, reliance on dataset quality, and uninterpretability (Anwar et al., 2024). Interpretability researchers have thus proposed a new class of representation-based interventions for steering LMs, which hope to address these issues. These methods include learning steering vectors from small labelled datasets, self-supervised sparse autoencoders (SAEs), among other techniques. Since steering may enable lightweight and interpretable control over model outputs, it has emerged as a potential alternative to finetuning and prompting (see §2).
Unfortunately, Pres et al. (2024); Braun et al. (2024) note that existing benchmarks for steering only evaluate a few methods at merely toy scales. To assess whether representation steering is a viable alternative to existing model control techniques, we need to evaluate it in a more realistic setting, e.g. over open-vocabulary concepts and on long-form generation, and compare it to prompting and finetuning baselines.
In this work, we introduce AXBENCH, a benchmark for evaluating LM control methods at scale using synthetic data. AXBENCH takes in a list of natural language descriptions of concepts and samples relevant training and evaluation data from an LLM. We evaluate model-control methods, including prompting and finetuning baselines, along two utility axes: concept detection C and model steering S . For the former, we use labelled synthetic data as ground truth; for the latter, we evaluate long-form generations using an LLM judge. The labelled training data enables comparison between supervised dictionary-learning methods (SDLs) and unsupervised methods like SAEs. The benchmark includes tasks generated from SAE concept lists for GemmaScope (Lieberum et al., 2024), covering two layers each from instruction-tuned Gemma-2-2B and Gemma-2-9B (Gemma Team et al., 2024). However, AXBENCH is by nature extensible to arbitrary concept descriptions: we intend to add new evaluation tasks as better feature-labelling techniques and new approaches to steering emerge.
We evaluate a variety of steering methods-including a novel weakly-supervised method we introduce, ReFT-r1along with prompting, full finetuning, and two parameterefficient finetuning methods (LoRA and LoReFT). On steering, only ReFT-r1 is competitive with finetuning and prompting baselines, while SAEs fall behind both ReFT-r1 and difference-in-means (Marks and Tegmark, 2024) on both axes. While representation steering methods largely lag behind incumbent model-control techniques, ReFT-r1 is evidence that steering can be pushed further with the availability of comprehensive evaluation benchmarks. Finally, along with AXBENCH, we train and publicly release SAE-scale feature dictionaries for ReFT-r1 and DiffMean. 1 We call this approach supervised dictionary learning (SDL; Figure 2).
this section cite: ['b0', 'b46', 'b6', 'b32', 'b35']

Section: Related work
Representation-based control. Interventional/causal interpretability has emerged as the dominant paradigm for understanding neural networks in the LLM era, enabling the reverse-engineering of circuits underlying specific behaviours (Giulianelli et al., 2018;Vig et al., 2020;Geiger et al., 2021;2022;Meng et al., 2022;Chan et al., 2022;Wang et al., 2023;Goldowsky-Dill et al., 2023;Geiger et al., 2024;Guerner et al., 2024;Geiger et al., 2024). An important assumption in much of this work is the linear representation hypothesis, which claims that linear subspaces of representations in neural networks encode concepts (Mikolov et al., 2013b;Pennington et al., 2014;Bolukbasi et al., 2016;Elhage et al., 2022;Park et al., 2023;Nanda et al., 2023).
Intervening on representations has thus emerged as an alternative to finetuning and prompting for LM control.
Representation-based steering by adding fixed vectors to activations, or clamping activations to a certain value along fixed directions, is one such intervention-based tool for model control (Zou et al., 2023;Li et al., 2024;Turner et al., 2024;Marks and Tegmark, 2024;Liu et al., 2024;van der Weij et al., 2024;Rimsky et al., 2024). Finetuning-based approaches such as ReFT (Wu et al., 2024a) enable optimisation of steering directions on a dataset. Steering vectors need not be computed from labelled data; SAEs enable scalable discovery of steering vectors from unlabelled data. In the same class of approaches, latent adversarial training (Casper et al., 2024) and circuit breakers (Zou et al., 2024) are representation-based control methods that increase the adversarial robustness of LLMs.
Sparse autoencoders. Sparse autoencoders (SAEs) aim to enable self-supervised and thus scalable decomposition of the representation space into meaningful concepts (Templeton et al., 2024;Chalnev et al., 2024;Makelov, 2024;O'Brien et al., 2024;Gao et al., 2024). SAEs are trained to reconstruct LLM hidden representations in a higherdimensional latent space with a sparsity penalty, based on the assumption that concepts must be represented sparsely in order to prevent interference. The latents are then labelled with natural-language descriptions using automatic interpretability pipelines (e.g. Juang et al., 2024), which can then be used to identify useful latents to steer the LM.
Recent work reports mixed results when evaluating SAEs for steering; SAEs (but also several other steering methods) suffer from a tradeoff between model control and capabilities preservation (Mayne et al., 2024;Chalnev et al., 2024;Durmus et al., 2024;Bhalla et al., 2025). However, Karvonen et al. (2024) report Pareto-optimal performance when using SAEs to prevent models from producing regular expressions in code. Overall, evaluating SAEs remains an open problem because there is no ground-truth set of features to compare against.
this section cite: ['b17', 'b56', 'b15', 'b37', 'b9', 'b19', 'b20', 'b45', 'b5', 'b13', 'b42', 'b62', 'b30', 'b35', 'b33', 'b47', 'b7', 'b63', 'b51', 'b8', 'b34', 'b43', 'b14', 'b25', 'b36', 'b8', 'b12', 'b3', 'b27']

Section: AXBENCH
AXBENCH is a benchmark which takes in a list of natural language descriptions of concepts and synthetically generates the appropriate training and evaluation data for each concept using an LLM (Figure 2). The training and evaluation data consists of labelled pairs of instructions and responses, where the responses are either positive examples expressing the presence of the concept of interest, or negative examples that represent the unsteered behaviour of the model (see §3.1 for details).
We evaluate along two axes: concept detection C and model steering S . For the former, we measure classification performance on a held-out set of labelled data. 2 For the latter, we use an LLM judge to rate steered outputs on three relevant axes (see §3.3).
In this work, we use natural language concept lists for GemmaScope SAEs as input, and generate training and evaluation data for the following representation sites: layers 10 and 20 of instruction-tuned Gemma-2-2B, and layers 20 and 31 of instruction-tuned Gemma-2-9B. We sample 500 concepts for each task to generate data; we term this dataset CONCEPT500. These eight tasks (4 sites × 2 axes) form the core training and evaluation testbeds for AXBENCH. Below, we describe the data generation process and evaluation setup for both axes.
this section cite: []

Section: Synthetic concept dataset generation
We construct a small training dataset
D train = {(x + c,i , y + )} n/2 i=1 ∪ {(x - c,i , y -)} n/2 i=1 .
with n examples and a concept detection evaluation dataset D concept of the same structure and harder examples, where y + and y -are binary labels indicating whether the concept c is present. We set n = 144 for our main experiments. 3   We query gpt-4o-mini-2024-07-18 to generate the data; the prompts used in this pipeline are presented in Appendix J.2. Generating the data requires the following steps (note that only the evaluation set includes hard negatives):
1. Genre labelling & seed instructions: We consider three genres: text, code, and math. We prompt the classification task over n classes can also be formulated into a binarised one over n features. 3 Using a small training dataset ensures our methods are practical and cost-effective alternatives to SAEs.
LLM to pick the genre g c for each concept. 4 We then randomly select seed instructions from our instruction pool which belong to genre g c ; see Appendix I for dataset details and for an explanation of how we sample instructions for different genres from existing public instruction datasets. We then prompt the LLM to generate responses to these instructions. 5
this section cite: []

Section: 2.
Positive examples: For each randomly sampled instruction from the instruction pool, we prompt the LLM to generate a response that incorporates the concept c. We use the generated concept-conditioned responses concatenated with their instructions (using the LM's chat template) as our positive set. 3. Negative examples: To evaluate the generalisation ability of each method, we independently sample seed instructions from all genres for negatives. 6 These instructions are shared across concepts in order to save generation costs (i.e., (x - c , y -) n/2 0 is independent of the concept c). We sample responses from the LM we plan to steer (not the LLM) without any additional instructions. We use the paired instructions and responses as our negative set. 4. Hard negative examples (evaluation only): For each concept, we find contrasting concepts that are semantically related to our concept of interest but which should not activate the concept. We find these by (a) generat-4 Genre labelling increases input diversity. For example, inputs related to concepts such as programming code contains syntactic errors should contain code instead of descriptions of coding errors.
5 Each example costs less than $0.00006. 6 We sample instructions based on overall genre distribution: 70% from text, 15% from code, and 15% from math.
ing a list of phrases that are semantically relevant to our concept, (b) filtering for those which are polysemous, and (c) finding alternative senses of those words which our concept should not activate on. This results in a set of contrast concepts c contrast , each of which is a specific sense of a polysemous word w contrast . We then ask the LLM to generate responses incorporating w contrast into the sentence where w contrast should express the sense related to c contrast . We use the contrastive responses paired with their instructions as our hard negative set.
The negative training set is not applicable to all methods (e.g. full finetuning only needs the positive training set for model steering).
this section cite: []

Section: C Concept detection
A popular LM interpretability method is to train probes (Conneau et al., 2018;Hewitt and Manning, 2019;Belinkov et al., 2017) that measure to what extent LM representations encode properties of interest, e.g. linguistic features. In recent years, the goal of concept detection has broadened to the open-vocabulary setting, with unsupervised methods becoming more common (Bills et al., 2023;Huben et al., 2024;Choi et al., 2024).
Task description. Formally, given a Transformer-based LM with a hidden dimension size of d, we define a concept classifier as a parameterized function Ψ Detect that maps a model representation h ∈ R d into a binary label ŷ indicating the relative presence of a concept:
Ψ Detect (h) = ŷ ∈ R 1 (1
)
where Ψ is any function, e.g. a neural network.
Evaluation dataset. To evaluate a concept classifier, we measure how accurately it can predict ground-truth labals on the labelled evaluation set from D concept (see §3.1).
Evaluation metrics. Since our labels are at the sequencelevel, we need to aggregate token-label scores from Ψ to evaluate it. Given a sequence of token representations h l = [ h l 1 , h l 2 , . . . , h l n ] with n tokens at layer l ∈ [1, m], we maxpool the detection scores to get a sequence-level prediction:
ŷDetect = max Ψ Detect (h l )(2)
We then normalize ŷDetect between [0, 1] by min-max normalisation over the evaluation dataset for each concept. The predicted score represents how strongly a concept is present in a sequence, which we can compare to the true label.
this section cite: ['b11', 'b22', 'b4', 'b24', 'b10']

Section: S Model steering
Representation-based steering has emerged as a potential alternative to existing model-control methods (e.g. finetun-ing and prompting) and a practical application of various interpretability methods (see §2). Unlike concept detection, model steering assesses causal efficacy in controlling model behaviour. Previous evaluation benchmarks for steering are not general-purpose; they either rely on a limited set of tasks (Zou et al., 2023;Makelov, 2024;Bhalla et al., 2025) or condition generation on a fixed prefix (Chalnev et al., 2024). To the best of our knowledge, we are the first to evaluate model steering methods in the open-vocabulary setting at scale.
Task description. Given a prompt x, the model's original generation can be written as ŷ = LM(x). We produce the model's counterfactual generation conditioned on the concept-based intervention Φ Steer (h):
ŷSteer = LM x, h ← Φ Steer (h)(3)
where h ← Φ Steer (h) is an in-place representation modification. We use the open-source intervention library pyvene to perform such interventions on PyTorch implementations of models (Wu et al., 2024b).
Evaluation dataset. We evaluate these steering methods in the instruction-following setting, where we sample instructions from Alpaca-Eval (Li et al., 2023) and prompt the LM to generate a response while intervening on its forward pass in-place using one of the steering methods.
this section cite: ['b62', 'b34', 'b3', 'b8', 'b62']

Section: Evaluation metrics.
For the intervened model generation, we evaluate ŷSteer based on the harmonic mean of the following scores, each of which the LLM rates using a discrete score of 0, 1, or 2:
1. Concept score represents how well the concept is incorporated into the response. 2. Instruct score represents how well the response is related to the instruction. 3. Fluency score represents how fluent the response is.
Since we compute the harmonic mean, the overall score also ranges from 0 to 2, but heavily penalises poor performance on any of these three subscores. For each concept, we randomly sample 10 instructions from Alpaca-Eval and sample continuations for each steering factor (see discussion on steering factor in §5.2). To ensure a fair comparison, we partition our instructions into two equally sized sets, selecting the best factor from one set and evaluating it on the holdout set. Our judge prompts with further discussion can be found in Appendix J.3. We also validate our LLM-generated overall score with human evaluation in Appendix M.
this section cite: []

Section: Methods
In this section, we describe the interpretability methods we evaluate along with our baseline prompting and finetun-ing methods. For each method, we label which axes it is evaluated on using C and S . All of our interpretability methods except SAEs are SDLs that learn rank-1 subspaces for targeted concepts.
Notation. Given a LM, the hidden representations of dimensionality d for a token sequence of length n in layer l of the LM are represented as h l = [ h l 1 , h l 2 , . . . , h l n ] ∈ R n×d . The set of representations concatenated from all of the training set inputs is denoted as H ∈ R s×d , where s = h |h|. We denote H + as the subset of H including only positive training inputs and H -for the negative inputs (see §3.1 for training dataset details). Finally, per-method projection vectors w and representations h i are the same shape: R d×1 .
this section cite: []

Section: C S Difference-in-means (DiffMean).
DiffMean uses the difference between averaged representations from two classes of inputs as a steering vector (Marks and Tegmark, 2024). The projection vector w DiffMean is defined as:
w DiffMean = 1 |H + | h + i ∈H + h + i mean of positives - 1 |H -| h - i ∈H - h - i mean of negatives(4)
We compute detection scores with the dot product, i.e. Ψ DiffMean Detect (h i ) = h i •w DiffMean . 7 Our steering operation is simple activation addition:
Φ DiffMean Steer (h i ) = h i + αw DiffMean
where α is the steering magnitude, which depends on the steering factor and is optimized as a hyperparameter, as described in §5.2.
this section cite: ['b35']

Section: C S Principle component analysis (PCA).
For PCA, we use the first principal component of the positive set of hidden representations as the projection vector. 8 We first subtract the mean H + from each h + , gathering the centered vectors into a matrix H ∈ R |H + |×d . We then find the top principal component w PCA ∈ R d×1 of H, i.e. the unit vector that captures the largest variance along its direction, using sklearn.decomposition.PCA (Pedregosa et al., 2011). We follow the same detection and steering setup as DiffMean.
this section cite: ['b44']

Section: C S Linear artificial tomography (LAT).
LAT searches for a single latent direction that can separate positive examples by learning from their pairwise activation differences (Zou et al., 2023). Concretely, we create pairwise activation differences δ by randomly partitioning H into pairs (h i , h j ) (with i ̸ = j) and computing δ = hi-hj ∥hi-hj ∥ , where the denominator ensures each difference is unitnormalized. We gather all these pairwise differences into a matrix ∆ ∈ R |H| 2 ×d . We then perform PCA (using sklearn) on ∆; then w LAT ∈ R d×1 is the top principal component of ∆. We follow the same detection and steering setup as DiffMean.
this section cite: ['b62']

Section: C S Linear probe (Probe).
The linear probe learns to classify tokens as concept-relevant by projecting representations h i onto a learned direction w Probe ∈ R d×1 just as in DiffMean. To convert this into a probability, we apply the sigmoid activation, and then minimise binary cross-entropy loss with the true labels:
min wProbe 1 |h| hi∈h L BCE (y, Sigmoid(h i • w Probe ))) (5
)
where y is the token-level class label indicating whether this token belongs to a positive or negative example. The detection and steering setup is then identical to DiffMean.
this section cite: []

Section: C S Supervised steering vector (SSV).
The supervised steering vector method directly learns an intervention that maximises the language-modelling probability of the positive responses. For a sequence of token representations h, we apply an intervention to each token representation:
Φ SSV (h i ) = h i + w SSV (6
)
where w SSV ∈ R d×1 is a learned vector. As described in §3.3, we backpropagate gradients by training with the language modeling loss, similar to supervised fine-tuning (SFT):
min wSSV n t=1 log P LM y t | y <t , x; h ← Φ SSV (h)(7)
where y i is the i-th output token, y <i are the preceding tokens, and x is the prompt. For evaluating concept detection and model steering SSV follows the same setup as DiffMean.
We apply ReLU to get the detection scores.
this section cite: []

Section: C S Rank-1 representation finetuning (ReFT-r1).
We introduce a novel method based on ReFT (Wu et al., 2024a) which jointly learns concept detection and steering on supervised data by combining the training objectives of linear probing and supervised steering.
We compute latents for concept detection as:
Ψ ReFT-r1 Detect (h i ) = ReLU(h i • w ReFT-r1 )(8)
During training we perform a representation-level intervention on each h i based on the latents of the sequence h:
Φ ReFT-r1 (h i ) = h i + 1 k TopK(Ψ ReFT-r1 Detect (h)) 1 w ReFT-r1(9)
where w ReFT-r1 ∈ R d×1 is a learned vector. Finally, the training objective combines language modelling loss subject to this intervention, along with L1 regularisation on the nontop-k latents:
min wReFT-r1    - n t=1 log P Φ ReFT-r1 LM (y t | y <t , x) + λ ai / ∈TopK(Ψ(h)) ∥a i ∥ 1    (10
) Detection and steering is identical to DiffMean.
this section cite: []

Section: C S Sparse autoencoders (SAE).
Sparse autoencoders are a self-supervised dictionary learning method (see §2). We use pretrained SAEs from GemmaScope, which are the best available SAEs for Gemma-family LLMs (Lieberum et al., 2024). 9 The SAEs we used are trained to learn two dictionary matrices, {W enc , W dec } ∈ R d×z where z is the number of latents. For our evaluating concept c, we use {w enc , w dec } ∈ R d×1 as the detection and steering representations, respectively:
Ψ SAE Detect (h i ) = σ (h i • w enc + b enc )
where σ is an activation function (in our case, JumpReLU) and b enc is a learned bias. 10 For steering, we use activation addition as DiffMean. Note that Templeton et al. (2024) use activation clamping; we report ablations in Appendix F.
this section cite: ['b32', 'b51']

Section: C S SAEs with AUROC selection (SAE-A).
Given that other methods have access to a training dataset, to enable fair comparison we attempt to use our training dataset for SAE feature selection. For each feature, we compute its max-pooled activations per Equation ( 2) over each training example, compute AUROC over the dataset given true labels, and select the highest-scoring feature by this metric.
this section cite: []

Section: C Bag-of-Words (BoW).
For the BoW baseline, we first construct a featurizer that tokenizes text by whitespace and counts word frequencies. The vocabulary for this featurizer is derived from the training dataset. We then train a logistic regression classifier to predict class probabilities, framing the task as binary classification. To mitigate overfitting, we incorporate a regularization term. This BoW approach leverages statistical biases inherent in LLM-generated data.
this section cite: []

Section: C Gradient-based baselines.
We test two gradient-based attribution methods, which are applicable only to concept detection: Input × gradients (I×G) and Integrated gradients (IG; Sundararajan et al., 2017). For both, we train a classification head on the hidden representations of some layer and apply the methods to produce token-level attribution scores Ψ Detect (h i ). Implementation details are in Appendix H.
this section cite: ['b50']

Section: C S Prompting baseline.
For concept detection, we use the same LLM judge as described in §3.3 to rate the presence of a concept on a scale of 0 to 2. For model steering, we use an LLM to engineer a prompt given a concept, which we use to steer our local model by prepending it to the actual instruction. We provide prompt templates and examples in Appendix J and Appendix O.foot_6
this section cite: []

Section: S Finetuning baselines.
We test full-parameter supervised finetuning (SFT) and two parameter-efficient finetuning methods: Low-rank adaptation (LoRA; Hu et al., 2022) and low-rank representation finetuning (LoReFT; Wu et al., 2024a). In all cases, we finetune to minimise the languagemodelling loss on the responses in the positive split of the dataset; the negative training split is discarded. We then use the finetuned models as baselines for steering.
For all of our SDLs except SSV, we constrain any learned subspace to have a unit norm, following the same setup as SAEs. With a unit-norm constraint, we find that SSV is hard to use for steering models. For prompting and finetuning baselines, we randomly score one generation on the testing instruction set (since the factor is not a parameter for those methods), resulting in the same number of observations for those methods.
this section cite: ['b23']

Section: Evaluation
Datasets. We synthetically generate training and validation datasets (see §3.1) for 500 concepts, which we release as CONCEPT500. The concepts are sampled from the Neuronpedia SAE concept list for GemmaScope as described in Appendix B. For each concept, we include 144 examples for training and ≈72 samples for evaluating concept detection. 12 In this paper, we train and evaluate all methods, and report results on CONCEPT500. For SFT, we only train and evaluate on the first 20 concepts due to limited resources.
For evaluating steering, we use the instructions from the Alpaca-Eval dataset (Li et al., 2023). For each concept, we sample 10 instructions. We generate up to 128 tokens for each instruction over 14 steering factors. We split the instructions into two equal sets -one for selecting the best factor and the other for evaluation.
We additionally release training and evaluation datasets Models. Our evaluations rely on access to and control over the LLM's representations. To reduce training cost, we prefer to use models for which pretrained SAEs are available. We thus evaluate our methods on two open models, Gemma-2-2B-it and Gemma-2-9B-it (henceforth referred to without the -it suffix), from the Gemma-family, with corresponding SAEs released as GemmaScope. We evaluate our methods with model representations from the residual streams of layers 10 and 20 for Gemma-2-2B and layers 20 and 31 for Gemma-2-9B. We use SAEs from GemmaScope that are trained for these layers. 13 To ensure a fair comparison, we perform separate hyperparameter-tuning for each method. Details can be found in Appendix K.
this section cite: ['b62']

Section: Results

this section cite: []

Section: C Concept detection
For concept detection, CONCEPT500 consists of passages of text with ground-truth labels for each concept. Each method provides us with token-level concept scores obtained from the representation of that token at a particular layer. To compute a passage-level score, we take the mean of the token-level concept scores. See Appendix N for a visualization of token-level concept scores. 13 For Gemma-2-2B, we follow the common practice to use SAEs for the base LM, as SAEs are not available for the instruction-tuned model at the time of publication (Lieberum et al., 2024).
this section cite: ['b32']

Section: AUROC.
In Table 1, we report the average area under the ROC curve (AUROC) for each method over all concepts. Overall, we find that DiffMean, Probe, and ReFT-r1 are the best performers with no statistically significant difference (p < 0.05) between any of them under a paired t-test. Prompt, SAE-A, and SSV are not far behind and significantly outperform the remaining methods. LAT also performs better than random. Vanilla SAEs are thus significantly outperformed by five supervised methods, all of which are much cheaper to train using a limited amount of synthetic data. The remaining methods (PCA, IG, and IxG) perform poorly; PCA's better-than-random performance is nevertheless impressive given its unsupervised nature. Additional results are given in Appendix C.
F1 score under class imbalance. In real-world text, positive instances of concepts are much rarer than negative instances. We thus report F1 on both the balanced setting (50% positive instances) and an imbalanced setting with 3600 additional negative examples (≈1% positive). We choose classification threshold by maximising F1, binarise the resulting predictions, and report statistics on this discrete classification. Figure 3 shows that the relative ordering of methods does not change substantially between the two settings; despite their sparsity, SAEs perform poorly, but LAT and PCA also degrade substantially.
this section cite: []

Section: S Model steering
For model steering, we take concept labels from CON-CEPT500 and apply the (pre)trained steering methods to the base model and sample generations. We score the generations using an LM judge as described in §3.3. We addi-
Method Gemma-2-2B Gemma-2-9B Avg. L10 L20 L20 L31 Prompt 0.698 0.731 1.075 1.072 0.894 LoReFT 0.701 0.722 0.777 0.764 0.741 SFT 0.637 0.714 --0.676 LoRA 0.637 0.641 0.602 0.580 0.615 ReFT-r1 0.633 0.509 0.630 0.401 0.543 DiffMean 0.297 0.178 0.322 0.158 0.239 SAE 0.177 0.151 0.191 0.140 0.165 SAE-A 0.166 0.132 0.186 0.143 0.157 LAT 0.117 0.130 0.127 0.134 0.127 PCA 0.107 0.083 0.128 0.104 0.105 Probe 0.095 0.091 0.108 0.099 0.098 SSV 0.072 0.001 0.024 0.008 0.026  tionally benchmark prompting, full-finetuning (SFT), and two parameter-efficient finetuning methods (LoReFT and LoRA) as non-steering baselines.
For steering methods, we note that steering factor is an important hyperparameter. We select the optimal steering factor for each method independently for every concept based on which factor achieves the highest overall steering score, as given by the LLM judge. Our actual steering magnitude (i.e., α, as described in §4) is the product of the steering factor and the maximal activations aggregated over the evaluation dataset for concept detection. 14Overall scores. We report the mean overall score for each method (i.e. the harmonic mean of three subscores: fluency, instruction-following, and concept presence) in Table 2. Prompting, along with slightly worse finetuning baselines, outperforms all steering methods on average, except
Method Gemma-2-2B Gemma-2-9B Avg. L10 L20 L20 L31 Prompt 90.0% 91.5% 97.6% 99.1% 94.5% LoReFT 88.9% 88.2% 88.6% 90.3% 89.0% SFT 90.0% 87.5% --88.8% LoRA 85.0% 83.4% 79.9% 81.5% 82.5% ReFT-r1 85.2% 82.3% 83.6% 76.0% 81.8% DiffMean 63.2% 55.2% 64.3% 52.2% 58.7% SAE 50.0% 50.0% 50.0% 50.0% 50.0% SAE-A 49.3% 46.6% 48.5% 50.7% 48.8% LAT 43.5% 48.2% 42.7% 48.6% 45.8% PCA 42.1% 42.9% 42.2% 45.4% 43.1% Probe 40.4% 44.0% 41.9% 45.6% 43.0% SSV 38.8% 32.0% 32.5% 34.0% 34.3% Winrate. We compute winrates against SAEs by comparing overall scores on each concept under each setting. We treat ties as 0.5 wins and 0.5 losses. We report the results in Table 3. Again, ReFT-r1 (88.0%) and DiffMean (61.6%) achieve winrates of greater than 50% against SAEs, and relative rankings are similar to those for overall score. We note that DiffMean and ReFT-r1 show higher winrates on earlier layers in both models.
Steering factor. We compare the effect of changing the steering factor on instruct vs. concept scores in Figure 4. We notice that increasing the factor monotonically reduces instruct score in all methods, i.e. larger steering vectors harm capabilities; this agrees with prior findings (Durmus et al., 2024;Chalnev et al., 2024). However, the effect varies by layer for concept score: concept score increases then decreases in earlier layers, while it roughly monotonically increases with steering factor in later layers. In all cases, ReFT-r1 traces a Pareto-optimal path, achieving the highest concept score for any chosen instruct score.
this section cite: ['b12', 'b8']

Section: Discussion
Simple yet powerful baselines. While representationlevel interventions have been shown to be useful in both enhancing model capabilities and for safety (see §2), they fail to outperform standard prompting and finetuning baselines on AXBENCH. This is sobering evidence of the current limitations of steering techniques. However, our results suggest that joint learning of concept detection and steering (as in ReFT-r1) may be the key to advancement.
SDL vs. SAEs. We have shown that SDL methods can achieve similar scalability and better performance at a lower cost compared to SAEs. Unlike SAEs, SDL methods require concepts to be known a priori; however, SDLs can be easily augmented with new features without retraining. We also note that SDLs depend on high-quality data generators, whereas SAEs rely on high-quality concept discriminators. These methods are not mutually exclusive and can complement each other.
SAE concept label quality. The concept lists used in this paper were adapted from Neuronpedia's autointerpretability pipeline, which is often skewed towards token-level concepts and misses high-level abstractions.
While we tried to do post-hoc SAE feature selection to mitigate this, the poor performance of SAEs is at least partially a reflection of the limitations of auto-interpretability.
It would be interesting to explore whether the SAE performance on AXBENCH improves as better feature labelling methods are used and labels become less shallow (e.g. Choi et al., 2024). We conduct a preliminary study assessing the steering performance of SAEs and SDLs using a limited set of higher-quality concept labels (see Appendix E.4). Our results show that, although higher-quality labels improve SAE performance, they do not narrow its gap to SDLs such as ReFT-r1. Future work might explore rule-based concepts, as in generic instruction-following benchmarks such as IFEval (Zhou et al., 2023).
this section cite: ['b10', 'b61']

Section: Conclusion
We introduced AXBENCH, a new benchmark for evaluating LM control methods at scale using synthetic data. To answer the question in the title of this work: our evaluation shows that even at SAE scale, representation steering is still far behind simple prompting and finetuning baselines. Simultaneously, we showed that a novel steering method, ReFT-r1, is capable of closing the gap to some extent; representationbased steering has not yet exhausted its potential. No matter the outcome, we believe that comprehensive evaluation benchmarks like AXBENCH are necessary for continued progress on this problem.
this section cite: []

Section: References
Ref_id:b0 Title: Foundational challenges in assuring alignment and safety of large language models Year: (2024)
Ref_id:b1 Title: What do neural machine translation models learn about morphology? Year: ()
Ref_id:b2 Title: Association for Computational Linguistics Year: (2017-07)
Ref_id:b3 Title: Towards unifying interpretability and control: Evaluation via intervention Year: (2025)
Ref_id:b4 Title: Language models can explain neurons in language models Year: (2023)
Ref_id:b5 Title: Man is to computer programmer as woman is to homemaker? Debiasing word embeddings Year: (2016)
Ref_id:b6 Title: A sober look at steering vectors for LLMs Year: (2024)
Ref_id:b7 Title: Defending against unforeseen failure modes with latent adversarial training Year: (2024)
Ref_id:b8 Title: Improving steering vectors by targeting sparse autoencoder features Year: (2024)
Ref_id:b9 Title: Causal scrubbing: A method for rigorously testing interpretability hypotheses Year: (2022)
Ref_id:b10 Title: Scaling automatic neuron description Year: (2024-10)
Ref_id:b11 Title: What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties Year: (2018-07)
Ref_id:b12 Title: Evaluating feature steering: A case study in mitigating social biases Year: (2024)
Ref_id:b13 Title: Toy models of superposition Year: (2022)
Ref_id:b14 Title: Scaling and evaluating sparse autoencoders Year: (2024)
Ref_id:b15 Title: Causal abstractions of neural networks Year: (2021)
Ref_id:b16 Title: Inducing causal structure for interpretable neural networks Year: (2022)
Ref_id:b17 Title: Under the hood: Using diagnostic classifiers to investigate and improve how language models track agreement information Year: (2018-11)
Ref_id:b18 Title: Decoding the thought vector Year: (2017)
Ref_id:b19 Title: Localizing model behavior with path patching Year: (2023)
Ref_id:b20 Title: A geometric notion of causal probing Year: (2024)
Ref_id:b21 Title: Enhancing automated interpretability with output-centric feature descriptions Year: (2024)
Ref_id:b22 Title: A structural probe for finding syntax in word representations Year: (2019-06)
Ref_id:b23 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b24 Title: Sparse autoencoders find highly interpretable features in language models Year: (2024)
Ref_id:b25 Title: Open source automated interpretability for sparse autoencoder features Year: (2024)
Ref_id:b26 Title: Speech and Language Processing. Online Year: (2025)
Ref_id:b27 Title: Sieve: SAEs beat baselines on a real-world task (a code generation case study) Year: (2024)
Ref_id:b28 Title: Sparse autoencoders reveal universal feature spaces across large language models Year: (2024)
Ref_id:b29 Title: Autoencoding beyond pixels using a learned similarity metric Year: (2016)
Ref_id:b30 Title: Inference-time intervention Year: (2024)
Ref_id:b31 Title: AlpacaEval: An automatic evaluator of instruction-following models Year: ()
Ref_id:b32 Title: Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2 Year: (2024)
Ref_id:b33 Title: Incontext vectors: Making in context learning more effective and controllable through latent space steering Year: (2024)
Ref_id:b34 Title: Sparse autoencoders match supervised features for model steering on the IOI task Year: (2024)
Ref_id:b35 Title: The geometry of truth: Emergent linear structure in large language model representations of true/false datasets Year: (2024)
Ref_id:b36 Title: Can sparse autoencoders be used to decompose and interpret steering vectors? Year: (2024)
Ref_id:b37 Title: Locating and editing factual associations in GPT Year: (2022)
Ref_id:b38 Title: Distributed representations of words and phrases and their compositionality Year: (2013-06)
Ref_id:b39 Title: Understanding and controlling a maze-solving policy network Year: (2023)
Ref_id:b40 Title: Human evaluation and correlation with automatic metrics in consultation note generation Year: (2022-05)
Ref_id:b41 Title: Annotation alignment: Comparing LLM and human annotations of conversational safety Year: (2024-11)
Ref_id:b42 Title: Emergent linear representations in world models of selfsupervised sequence models Year: (2023)
Ref_id:b43 Title: Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models Year: (2023)
Ref_id:b44 Title: Scikit-learn: Machine learning in Python Year: (2011)
Ref_id:b45 Title: GloVe: Global vectors for word representation Year: (2014-10)
Ref_id:b46 Title: Towards reliable evaluation of behavior steering interventions in LLMs Year: (2024)
Ref_id:b47 Title: Steering Llama 2 via contrastive activation addition Year: (2024)
Ref_id:b48 Title: Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP Year: (2024-11)
Ref_id:b49 Title: Extracting latent steering vectors from pretrained language models Year: (2022-05)
Ref_id:b50 Title: Axiomatic attribution for deep networks Year: (2017)
Ref_id:b51 Title: Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread Year: (2024)
Ref_id:b52 Title: Understanding and controlling a maze-solving policy network Year: (2023-03)
Ref_id:b53 Title: Maze-solving agents: Add a top-right vector, make the agent go to the top-right. Alignment Forum Year: (2023-03)
Ref_id:b54 Title: Steering language models with activation engineering Year: (2024)
Ref_id:b55 Title: Deep feature interpolation for image content changes Year: (2017-07-21)
Ref_id:b56 Title: Investigating gender bias in language models using causal mediation analysis Year: (2020)
Ref_id:b57 Title: Interpretability in the wild: a circuit for indirect object identification in GPT-2 small Year: (2019)
Ref_id:b58 Title: Implicit semantic data augmentation for deep networks Year: (2016)
Ref_id:b59 Title: Representation finetuning for language models Year: (2024)
Ref_id:b60 Title: pyvene: A library for understanding and improving PyTorch models via interventions Year: (2024-06)
Ref_id:b61 Title: Instruction-following evaluation for large language models Year: (2023)
Ref_id:b62 Title: Representation engineering: A top-down approach to AI transparency Year: (2023)
Ref_id:b63 Title: Improving alignment and robustness with circuit breakers Year: (2024)
Ref_id:b64 Title: Try to avoid copying words from the definition of Year: ()
Ref_id:b65 Title: Ensure that your response relates to '[Concept goes here]', even if the overall meaning is not fully coherent. **Formatting Guidelines:** -Return only the response to the instruction. -Write the final content (or appropriate format for the genre) in plain text Year: ()
Ref_id:b66 Title: ** Return only the final content, following the guidelines above. Generate response given instruction without mentioning given concept Given the following instruction Year: ()
Ref_id:b67 Title: Provide a response that continues or addresses the instruction naturally Year: ()
Ref_id:b68 Title: Avoid any mention of '[Concept goes here]' in the continuation, regardless of coherence. **Formatting Guidelines:** -Return only the response to the instruction. -Write the final content (or appropriate format for the genre) in plain text Year: ()
Ref_id:b69 Title: Answer:** Return only the final content, following the guidelines above Year: ()
