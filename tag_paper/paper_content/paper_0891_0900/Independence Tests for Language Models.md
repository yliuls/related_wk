Title: Independence Tests for Language Models
Abstract: Motivated by liability and intellectual property concerns over open-weight models we consider the following problem: given the weights of two models, can we test whether they were trained independently-i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. We compute the p-values by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures between the original two models versus these copies. We report p-values on pairs of 21 open-weight models (210 total pairs) and find we correctly identify all pairs of non-independent models. In the unconstrained setting we make none of the prior assumptions and allow for adversarial evasion attacks that do not change model output. We thus propose a new test which matches hidden activations between two models, which is robust to these transformations and to changes in model architecture and can also identify specific non-independent components of models. Though we no longer obtain exact p-values from this test, empirically we find it reliably distinguishes non-independent models like a p-value. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch.

Section: Introduction
Consider the ways in which two models could be related: one model may be a finetune of the other; one could be spliced and pruned from certain parts of the other; both models could be separately fine-tuned from a common ancestor; finally, they could be independently trained from each other. We consider the problem of determining whether two models are independently trained versus not from their weights, which we formalize as a hypothesis testing problem in which the null hypothesis is that the weights of the two models are independent. We concretely treat only the weight initialization as random and thus consider two models with different random initial seeds as independent, even if both models were trained on the same data, or one model was distilled from the outputs of the other.
A solution to this independence testing problem would help auditors track provenance of open-weight models. This is pertinent because while open-weight models enable broader access and customization, they also pose potential risks for misuse as they cannot be easily monitored or moderated (Kapoor et al., 2025). Model developers would also gain an enhanced ability to protect their intellectual property (IP) (Mensch, 2024;Peng et al., 2023) and enforce custom model licenses (Dubey et al., 2024;DeepSeek-AI et al., 2024).
We consider two settings of the independence testing problem. In the constrained setting, we make assumptions on training and initialization (essentially, that the training algorithm is equivariant to permuting the hidden units of the random initialization) that enable us to obtain provably valid p-values. The main idea is that under these assumptions we can cheaply simulate many exchangeable copies of each model's weights and compare the value of some test statistic (e.g., cosine similarity of model weights) on each of these copies with the original model pair. The assumptions generally hold in practice but preclude robustness to adversarial evasion attacks and architectural changes.
For the constrained setting, we evaluate various test statistics on 21 models of the Llama 2 architecture (Touvron et al., 2023), including 12 fine-tunes of Llama 2 and nine independently trained models, obtaining extremely small p-values We share code at https://github.com/  ahmeda14960/model-tracing. for all 69 non-independent model pairs. Notably, our tests retain low p-values over different fine-tuning methods (e.g., different optimizers) and on models fine-tuned for many tokens from the base model such as Llemma (Azerbayev et al., 2024), which was fine-tuned on an additional 750B tokens from Llama 2 (i.e., 37.5% of the Llama 2 training budget). We also confirm that the leaked Miqu-70B model from Mistral is derived from Llama 2-70B.
For the unconstrained setting we develop a test robust to simple modifications to model weights and architecture, such as permuting hidden units, that can violate the assumptions of the constrained setting if an adversary applies them after fine-tuning. Though we are not able to obtain provably exact p-values in the unconstrained setting, we derive a test whose output empirically behaves like a p-value and reliably distinguishes non-independent models from independent models. In particular, we first align the hidden units of two models-which may each have different activation types and hidden dimensions-and then compute some measure of similarity between the aligned models. Because of the alignment step, the test is robust to changes in model architecture and various adversarial evasion attacks (including those that break prior work). Moreover, it can localize the dependence: we can identify specific components or weights that are not independent between two models, even when they have different architectures.
We evaluate our unconstrained setting test on 141 independent model pairs and find that its output empirically behaves like a p-value in the sense that it is close to uniformly distributed in [0, 1] over these pairs. In contrast, it is almost zero for all dependent pairs we test (including those for which we simulate a somewhat strong adversary by retraining entire layers from scratch). We also employ our test to identify pruned model pairs, which occur when one reduces the layer dimensions by retaining only select activations and weights from a pre-trained model; for example, we identified the precise layers of Llama 3.1 8B from which each Llama 3.2 3B and Llama 3.2 1B layer was derived.
The work most closely related to ours is due to Zeng et al. (2024), who considered our constrained setting; they develop various tests to determine whether a model as a whole is independent of another by computing the cosine similarity of the products of certain weight matrices in both models. They show that their tests are robust to simple adversarial transformations of model weights that preserve model output; however, we detail in Appendix G.1 other transformations to perturb dependent models that evades detection by their tests. Additionally, unlike Zeng et al. (2024), in the constrained setting we obtain exact p-values from our tests. Jin et al. (2024) propose crafting specific queries that are likely to produce different responses among independently trained models; their method does not require access to weights but also does not produce exact p-values.
this section cite: ['b9', 'b14', 'b18', 'b4', 'b3', 'b24', 'b1', 'b29', 'b29', 'b8']

Section: Methods

this section cite: []

Section: Problem formulation
Let f : Θ × X → Y denote a model mapping parameters θ ∈ Θ and an input X ∈ X to an output f (X; θ) ∈ Y. We represent a model training or fine-tuning process as a learning algorithm A : Θ → Θ that takes as input a set of initial parameters corresponding to either a random initialization or, in the case of fine-tuning, base model parameters. Specifically, A includes the choice of training data, ordering of minibatches, and all other design decisions and even the randomness used during training-everything other than the initial model weights.
Given two models θ 1 , θ 2 ∼ P for some joint distribution P ∈ P(Θ 1 × Θ 2 ), our goal is to test the null hypothesis
H 0 : θ 1 ⊥ θ 2 ,(1)
where ⊥ denotes independence of two random variables.
One example of a case where θ 1 and θ 2 might not be independent is if θ 2 is fine-tuned from θ 1 , i.e., Θ 1 = Θ 2 (meaning the two models share the same architecture) and θ 2 = A(θ 1 ) for some learning algorithm A. We treat learning algorithms as deterministic functions. Thus, for θ 1 = A 1 (θ 0 1 ) and θ 2 = A 2 (θ 0 2 ), then θ 0 1 ⊥ θ 0 2 (i.e. two models with independent random initializations) implies our null hypothesis.
Deep learning models are often nested in nature. For example, Transformer models include self-attention layers and MLP layers as submodels. We formalize the notion of a submodel via the following definition.
Definition 1. A model f : Θ×X → Y contains a submodel g : Θ ′ × X ′ → Y ′ if there exists a projection operator proj : Θ → Θ ′ such that for all θ ∈ Θ we have f (x; θ) = f out (g(f in (x); proj(θ))) for some functions f in : X → X ′ and f out : Y ′ → Y (which may depend on θ).
Many of our experiments will involve Transformer models specifically containing MLP layers with Gated Linear Unit (GLU) activations, which are widely used among language models. It thus will be useful to define this type of MLP presently through the following example.
Example 1: (GLU MLP) Let G, U ∈ R h×d and D ∈ R d×h . Let σ : R → R be an element-wise activation function. For x ∈ R d and θ = (G, U, D) ∈ Θ h mlp , let f mlp (x; θ) := D(σ(Gx) ⊙ (U x)). Likewise, for X ∈ R s×d let f mlp (X; θ) ∈ R s×d denote the result of broadcasting f mlp over the rows of X. ♢
In addition to the basic independence testing problem above, we also consider the problem of localized testing: testing whether various pairs of submodels among two overall models are independent or not. A prototypical example of a localized testing problem is identifying which layers of a larger model (e.g., Llama 3.1-8B) were used to initialize a smaller model (e.g., Llama 3.2-3B) (in this case, we treat the layers as different submodels).
this section cite: []

Section: Constrained Setting

this section cite: []

Section: TESTING FRAMEWORK
Algorithm 1 (PERMTEST) encapsulates our framework for computing p-values against the null hypothesis in the constrained setting, wherein we simulate T exchangeable copies of the first model θ 1 by applying transformations to its weights. The exchangeability of these copies holds under some assumptions on the learning algorithm and random initialization that produced the original model. We capture these assumptions in the following definitions; together, they define the constrained setting.
Definition 2 (Π-invariance). Let Π ⊂ Θ → Θ. A distribution P ∈ P(Θ) is Π-invariant if for θ ∼ P and any π ∈ Π, the parameters θ and π(θ) are identically distributed.
Algorithm 1: Test for computing p-values (PERMTEST) Input: Model weights θ 1 , θ 2 Parameters :test statistic ϕ; discrete transformation class Π; permutation count T Output: p-value p ∈ (0, 1] 1 n ties ← 0;
2 for t ∈ 1, . . . , T do 3 π t ∼ Unif(Π); 4 ϕ t ← ϕ(π t (θ 1 ), θ 2 ); 5 s ← s + 1{ϕ t = ϕ(θ 1 , θ 2 )}; T t=1 1{ϕ t < ϕ(θ 1 , θ 2 )}; 8 return p Definition 3 (Π-equivariance). Let Π ⊂ Θ → Θ, π ∈ Π, and θ 0 ∈ Θ. A learning algorithm A is Π-equivariant if and only if π(A(θ 0 )) = A(π(θ 0 )).
The main idea underlying PERMTEST is that so long as θ 1 = A(θ 0 1 ) and θ 0 1 ∼ P for some Π-equivariant learning algorithm A and Π-invariant distribution P , we can simulate T exchangeable (but not independent) copies {π t (θ 1 )}
T t=1 of θ 1 by sampling π t i.i.d.
∼ Unif(Π). This allows us to efficiently compute an exact p-value without actually repeating the training process of θ 1 . In effect, Definitions 2 and 3 imply that π commutes with A-i.e., π(A(θ 0 1 )) = A(π(θ 0 1 )). Under exchangeability, the p-value output by PERMTEST will be uniformly distributed over {(i + 1)/(T + 1)} T i=0 . Standard initialization schemes for feedforward networks are symmetric over the hidden units of the network, and so one example of a class of transformations with respect to which any such initialization is invariant is the set of permutations over the hidden units of the network. Moreover, the gradient of the model's output with respect to the hidden units is permutation equivariant; thus, any learning algorithm whose update rule is itself a permutation equivariant function of gradients (e.g., SGD, Adam, etc.) satisfies Definition 3 with respect to these transformations. A (contrived) example of a learning algorithm that is not permutation equivariant is one that uses different learning rates for each hidden unit depending on the index of the hidden unit.
Example 2 (Permuting hidden units): Let θ = (G, U, D) ∈ Θ h mlp parameterize a GLU MLP, where recall f mlp (x; θ) := D(σ(Gx) ⊙ (U x)) for some element-wise activation function σ : R → R. Abusing notation, let Π be the set of h × h permutation matrices such that for π ∈ Π we define π(θ) = (πG, πU, Dπ T ). Observe f mlp (x; θ) = f mlp (x; π(θ)) and π(∇ θ f mlp (x; θ)) = ∇ π(θ) f (x; π(θ)) for all inputs x. ♢
The assumptions we make in the constrained setting suffice for PERMTEST to produce a valid p-value, as we show in the following theorem, whose proof uses symmetry of the initialization and training process (full proof in Appendix A). 1 Importantly, the result of the theorem holds (under the null hypothesis) without any assumptions on θ 2 ; so, a model developer of θ 1 testing other models with our methods can have confidence in the validity of our test without trusting the provider of θ 2 . Of course, if θ 2 does not satisfy the equivariance assumption on training (as in the unconstrained setting), then PERMTEST is unlikely to produce a low p-value even in cases where θ 1 and θ 2 are not independent (e.g., if an adversary finetunes θ 2 from θ 1 but then afterwards randomly permutes its hidden units). Theorem 1. Let ϕ : Θ × Θ → R be a test statistic and Π ⊂ Θ → Θ be finite. Let A : Θ → Θ be Π-equivariant and let P ∈ P(Θ) be Π-invariant. For
θ 0 1 ∼ P , let θ 1 = A(θ 0 1 ). Let θ 2 ∈ Θ be independent of θ 1 . Then p = PERMTEST(θ 1 , θ 2 ) is uniformly distributed on { i+1 T +1 } T i=0 .
We also generalize Theorem 1 to apply to randomized learning algorithms that satisfy a notion of equivariance in distribution (including dropout) in Appendix B. However, throughout the main text we will continue to treat learning algorithms as deterministic for the sake of simplicity.
this section cite: []

Section: TEST STATISTICS
We have shown PERMTEST produces a valid p-value regardless of the test statistic ϕ we use. The sole objective then in designing a test statistic is to achieve high statistical power: we would like p = PERMTEST(θ 1 , θ 2 ) to be small when θ 1 and θ 2 are not independent. The statistics in this section apply to any model pair sharing the same architecture.
Prior work (Xu et al., 2024) proposed testing whether two models are independent or not based on the ℓ 2 distance between their weights, summed over layers. Specifically, for a model with L layers parameterized by
Θ = Θ 1 × ... × Θ L , with θ 1 = (θ (ℓ) 1 ) L ℓ=1 and θ 2 = (θ (ℓ) 2 ) L ℓ=1 , let ϕ ℓ2 (θ 1 , θ 2 ) := - L i=1 ℓ 2 (θ (ℓ) 1 , θ (ℓ)
2 ). We can obtain p-values from ϕ ℓ2 by using it within PERMTEST. However, a major limitation is that in order to obtain a p-value less than 1/(T + 1) we must recompute ϕ ℓ2 at least T times; then the statistical power of our test using ϕ ℓ2 is therefore bottlenecked by computation.
To address this limitation, we propose a family of test statistics whose distribution under the null is identical for any model pair. The test statistics all share the following general form based on Algorithm 2 (MATCH): for m, n ∈ N and M : Θ → R n×m , let
ϕ M (θ 1 , θ 2 ) := SPEARMAN(MATCH(M (θ 1 ), M (θ 2 )), [1, ..., n]),(2)
1 As a result of the test yielding exact p-values, we can directly control for the false positive rate via the significance threshold.
where SPEARMAN is the Spearman rank correlation (Algorithm 3). Equation ( 2) is applicable to any model architecture Θ for which we can define a suitable matrix valued function M of model parameters. For example, M could extract a weight matrix or activation matrix (based on some set of inputs) from a layer of the model, where each row corresponds to a hidden unit of the model. We use MATCH to align the rows of the two extracted matrices and compute the Spearman correlation of this alignment with the identity map between rows. We describe matching in Algorithm 2, wherein cossim denotes cosine similarity function and LAP denotes the algorithm of Ramshaw & Tarjan (2012) we use to solve the matching problem.
The idea is that for two dependent models, each row of M (θ 1 ) should be similar to its counterpart in M (θ 2 ); thus, the alignment found by SPEARMAN will be close to the identity map. Meanwhile, so long as M is a Π-equivariant map (Definition 4), then ϕ M (θ 1 , θ 2 ) under the null yields valid p-values (see Theorem 2 and proof in Appendix A); so we can use the more computationally-efficient Algorithm 3 to convert statistics to p-values instead of running PERMTEST.
Definition 4. (equivariant map) A matrix-valued function M : Θ → R n×m is Π-equivariant with respect to a class of transformations Π : Θ → Θ if there exists a bijection between Π and the set of n × n permutation matrices such that M (π(θ)) = πM (θ) for all θ ∈ Θ and π ∈ Π.
Theorem 2. Let M : Θ → R n×m be a Π equivariant map and let P ∈ P(Θ) be Π-invariant. Let θ 1 , θ 2 ∈ Θ be independent random variables, with
θ 1 = A(θ 0 1 ) for θ 0 1 ∼ P 1 . Then ϕ M (θ 1 , θ 2 ) is uniformly distributed on [0, 1). Algorithm 2: Cosine similarity matching (MATCH) Input: Matrices W 1 , W 2 with h rows Output: Permutation π : [h] → [h] 1 for i ∈ 1, . . . , h do 2 for j ∈ 1, . . . , h do 3 C i,j ← cossim((W 1 ) i , (W 2 ) j ); 4 π ← LAP(C); 5 return π
Taking various such functions M yields different test statistics. We focus our experiments on Transformer models consisting of a series of L Transformer blocks that each contain a GLU MLP submodel, and we take M (θ) to be either the up projection weights or the hidden-layer activations of one of these MLP submodels. In particular, let U (ℓ) (θ) ∈ R h×d denote the first layer up projection weights of the MLP in the ℓ-th block, where h is the hidden dimension and d is the input dimension, and let H (ℓ) (θ) ∈ R h×(N •s) denote the (flattened) hidden activations that obtain from passing N length s input sequences X ∈ R N ×s×d to the same MLP module (the test is valid for any X; we will specify later how we choose X in our experiments). The two main test statistics we employ in our experiments are ϕ U (ℓ) and ϕ H (ℓ) .
Both U (ℓ) and H (ℓ) are equivariant with respect to permuting the hidden units of the corresponding MLP, so we can directly interpret the outputs of ϕ U (ℓ) and ϕ H (ℓ) as p-values. Moreover, we can separately permute the hidden units of the MLP in the ℓ-th block without changing the inputs or outputs of the other blocks. Thus, as we show in Theorem 3 (proof in Appendix A), we can aggregate the p-values from ϕ U (ℓ) and ϕ H (ℓ) across blocks using Fisher's method ( (Mosteller & Fisher, 1948)) to obtain a more powerful test in Algorithm 4 (FISHER).
Algorithm 3: Deriving p-values from Spearman corre- lation (SPEARMAN) Input: Permutations π 1 , π 2 : [h] → [h] Output: p-value p ∈ (0, 1] 1 r ← 1 -6 (π1[i]-π2[i]) 2 h(h 2 -1) ; 2 t ← r h-2 1-r 2 ; 3 p ← P(T n-2 > t) ; 4 return p Algorithm 4: Aggregating p-values (FISHER) Input: p-values { p (i) } L i=1
Output: p-value p ∈ (0, 1]
1 ξ ← L i=1 log p (i) ; 2 p ← 1 -P(χ 2 2L < -2ξ); 3 return p
Theorem 3. Consider block indices i, j ∈ [L] with i ̸ = j for models with L blocks. Suppose for ℓ ∈ {i, j} that 1. M (ℓ) : Θ → R h×N is equivariant with respect to Π (ℓ) , i.e., for any θ ∈ Θ and π (ℓ) ∈ Π (ℓ) we have
M (π (ℓ) (θ)) = π (ℓ) M (θ).
2. A is a Π (ℓ) -equivariant learning algorithm and
P ∈ P(Θ) is a Π (ℓ) -invariant distribution. Let θ 1 , θ 2 ∈ Θ. If θ 1 ⊥ θ 2 for θ 1 = A(θ 0 1 ) with θ 0 1 ∼ P , then MATCH(M (i) (θ 1 ), M (i) (θ 2 )) ⊥ MATCH(M (j) (θ 1 ), M (j) (θ 2 )). Recall ϕ U (ℓ) and ϕ H (ℓ) are functions of MATCH(M (ℓ) (θ 1 ), M (ℓ) (θ 2 )) respectively for M (ℓ) = U (ℓ)
and M (ℓ) = H (ℓ) , both of which satisfy the assumptions of the theorem. Thus, the result of the theorem applies to both these test statistics, and the independence of the p-values from these test statistics across blocks follows directly from the independence of the statistics themselves.
this section cite: ['b25', 'b21', 'b17']

Section: Unconstrained Setting
For the unconstrained setting, our goal is to design a robust test that applies to models of different architectures and is robust to output-preserving transformations of model weights. Recall our tests for the constrained setting satisfy neither of these desiderata: these tests assume both models have the same number of hidden units, and it is easy to fool them without changing the output of a model by permuting the order of the hidden units in the model.
Our robust test reposes on the design of ϕ M in equation ( 2). The goal is to identify two matrix valued functions of model parameters M, M ′ : Θ → R n×m that jointly satisfy the following condition: any output-preserving transformation of model parameters must transform both M and M ′ in the same way. Then, whereas previously we would correlate MATCH(M (θ 1 ), M (θ 2 )) with the identity permutation, we instead define
ϕ M,M ′ := SPEARMAN(MATCH(M (θ 1 ), M (θ 2 )), MATCH(M ′ (θ 1 ), M ′ (θ 2 )). (3)
The above goal is aspirational in the sense that for any nontrivial deep learning model we are not able to fully enumerate the set of transformations of model parameters to which model output is invariant; nonetheless, it will serve as a useful guiding principle for designing our robust test under the framework of equation (3). We organize the description of our full robust test-which is generally applicable to a variety of model architectures-into two parts: first, in Section 2.3.1 we instantiate equation ( 3) to obtain a test for GLU MLP models. Then, in Section 2.3.2 we use our GLU MLP test as a primitive for designing a test that applies to general deep learning models (including those which do not contain any GLU MLP submodels). h2}×N be the output of the up projection operation and let H gate (θ k ) = G k X ∈ R max{h1,h2}×N be the output of the gate projection operation (with appropriate zero-padding when h 1 ̸ = h 2 ). Due to the element-wise product operation, we conjecture that in general it is not possible to permute the rows of G k while preserving the output of θ i without permuting the rows U k in the same way, and so we use ϕ M,M ′ with M = H gate and M ′ = H up for our GLU MLP test. Henceforth, we will shorthand this test as ϕ MATCH .
2.3.1. TESTING GLU MODELS Recalling our definition of a GLU MLP model in Example 1, for k ∈ {1, 2} let θ k = (G k , U k , D k ) ∈ Θ h k mlp , and with in- puts X ∈ R d×N let H up (θ k ) = U k X ∈ R max{h1,
As with the constrained setting, we focus much of our experiments on Transformer models, which recall consist of a series of L Transformer blocks that each contain a GLU MLP submodel. Adopting the notational conventions of Section 2.2.2, we can apply our GLU MLP test to the ℓ-th block by taking M = H (ℓ) gate and M ′ = H (ℓ) up , where like before (in the case of ϕ H (ℓ) ) we obtain the activation inputs for each block by computing a forward pass through the full model over a set of length s sequences of input tokens.
We can aggregate the results of these tests over blocks using FISHER, like we do for ϕ U (ℓ) and ϕ H (ℓ) in the constrained setting. Alternatively, we can apply the test to all possible O(L 2 ) pairs of blocks between two Transformer models if we suspect that certain blocks from one model served as the initializations for different blocks in the other model. Specifically, we can test the i-th block of θ 1 and the j-th block of θ 2 using ϕ (i,j)
MATCH := SPEARMAN(MATCH(H (i) gate (θ 1 ), H (j) gate (θ 2 )), MATCH(H (i) up (θ 1 ), H (j) up (θ 2 ))
). This test is relevant for pruned models, where only select blocks (layers) of θ 2 may be used to initialize the smaller θ 1 ; or, if an adversary takes only certain layers, or even only certain activations, of a pre-trained model and injects other layers.
this section cite: []

Section: BEYOND GLU MODELS
Thus far we have focused on models f : X × Θ → Y containing a GLU MLP submodel. In particular, recalling Definition 1, we have assumed for some proj mlp :
Θ → Θ h mlp that f (x; θ) = f out (f mlp (f in (x); proj mlp (θ))).(4)
Now, our goal is to test more general types of models. In particular, we generalize to an arbitrary alternative submodel
f alt : R d × Θ alt → R d with proj alt : Θ → Θ alt such that f (x; θ) = f out (f alt (f in (x); proj alt (θ))).(5)
In order to test whether two models θ 1 , θ 2 ∈ Θ of the more general form in equation ( 5) are independent, we will first construct proxy models of the form in equation ( 4) and then apply our previous test ϕ MATCH to these proxy models. We construct these proxy models by leveraging the fact that f alt shares the same input and output space with f mlp . Specifically, for k ∈ {1, 2} we first learn parameters θ k ∈ Θ h mlp so that f mlp (• ; θ k ) approximates f alt (• ; proj alt (θ k )). We then return ϕ MATCH ( θ 1 , θ 2 ). We capture this two-stage process in Algorithm 5.
Perhaps surprisingly, we show that Algorithm 5 is effective in practice at distinguishing independent versus non independent models. The hidden dimension h and input distribution
Algorithm 5: Generalized robust test Input: Model parameters θ 1 , θ 2 ∈ Θ Parameters :distribution P over R d Output: p ∈ [0, 1] 1 for k ∈ {1, 2} do 2 θ i ← arg min θ E x∼P f alt (x; proj alt (θ k )) -f mlp (x; θ k ) 2 3 return p ← ϕ MATCH ( θ 1 , θ 2 )
P with which we learn the GLU MLP are hyperparameters of the test. See Section 3.2 for details.
this section cite: []

Section: Experimental Results

this section cite: []

Section: Constrained setting
We first validate validate the effectiveness of our tests in the constrained setting on open-weight language models -21 models trained with the Llama-7B architecture with public documentation on ground truth model independence. These models all contain L = 32 GLU MLPs, each part of its own Transformer block. We run experiments with three different tests. Each test comprises two elements: a test statistic along with a method for computing p-values from the statistic. For the first test, we use ϕ ℓ2 and compute pvalues via PERMTEST with T = 100. For the other two tests, we compute p-values by directly aggregating the outputs of (respectively) ϕ U (ℓ) and ϕ H (ℓ) over ℓ ∈ [L] using FISHER. We obtain the inputs to the GLU MLP in the ℓ-th required to compute ϕ H (ℓ) by sampling sequences of tokens uniformly at random from the models' vocabulary and computing a forward pass through the full model while storing the MLP hidden layer activations. The equivariant transformation class Π is the set of permutations over both the hidden units of each MLP (see Example 2) and the embedding dimension of the model (i.e., the inputs passed to the both the MLP and self-attention layers in each block); we defer the precise definition of Π in this case to Appendix D.
this section cite: []

Section: BASELINE STATISTICS
We employ two test statistics from prior work as baselines: Jensen-Shannon divergence between next token output distributions (ϕ JSD , (Lin, 2006)), and ϕ ℓ2 (Xu et al., 2024)) with PERMTEST (details in Section 3.2.2). We computed ϕ JSD using input sequences sampled from WikiText-103 (Merity et al., 2017;Xu et al., 2024) (consistent with prior work). Since the Jensen-Shannon divergence is (by definition) invariant to any transformation of weights that does not affect model output, we cannot compute meaningful p-values using PERMTEST; instead, in our experiments we report the raw value of the test statistic itself.
this section cite: ['b25', 'b15', 'b25']

Section: LLAMA FAMILY EXPERIMENTAL RESULTS
The 21 models we evaluated include 6 base models (trained from scratch), so we have six disjoint sets of the models based on Llama-2-7b-hf stemming from a diverse mix of industry labs and non-profits (Azerbayev et al., 2024;Sudalairaj et al., 2024;Liu et al., 2024;Li et al., 2023). We consider any pair of models in the same tree as dependent and all other pairs as independent. We include examples of further fine-tunes (e.g., llemma 7b) of fine-tunes (e.g., CodeLlama-7b-hf) among the models we test. We will mostly refer to models using by their Huggingface identifiers, without the organization names for clarity.
We evaluated four test statistics: ϕ U (ℓ) (cosine similarity of weights), ϕ H (ℓ) (cosine similarity of hidden activations), ϕ ℓ2 (ℓ 2 distance), and ϕ JSD (Jensen-Shannon Divergence). As we describe in Section 2.2.2, for ϕ U (ℓ) and ϕ H (ℓ) we report aggregated p-values over all blocks using FISHER.
We report results for a subset of these pairs involving base model Llama-2-7b-hf in Table 1 while deferring the rest and the full experimental setup details to Appendix E.
θ 1 = Llama-2-7b-hf, p-values θ 2 =? Indep.? ϕ JSD (log) ϕ ℓ 2 ϕ U (ℓ) ϕ H(
ℓ) llama-7b-hf ✓ -11.10 0.98 0.60 0.25 vicuna-7b-v1.1 ✓ -10.40 0.63 0.16 0.64 Amber ✓ -10.69 0.75 0.36 0.88 open-llama-7b ✓ -8.38 0.26 0.36 0.71 vicuna-7b-v1.5 ✗ -10.87 0.01 ε ε CodeLlama-7b-hf ✗ -10.62 0.01 ε ε llemma-7b ✗ -10.24 0.01 ε ε Orca-2-7b ✗ -10.34 0.01 ε ε Table 1. We report various constrained setting test statistics with θ1 as Llama-2-7b-hf and θ2 ranging over the listed models.
The "independent" column is the ground truth. Here, ε = 2.2e-308 (numerical underflow for a 64-bit float). We find our proposed tests ϕ U (ℓ) and ϕ H (ℓ) distinguish independent versus non-independent model pairs with high statistical power.
Consistent with prior work (Xu et al., 2024), we find that ϕ JSD does not reliably distinguish independent versus dependent model pairs. For example, CodeLlama-7b-hf exhibits a larger divergence with Llama-2-7b-hf than the independently-trained models llama-7b-hf and Amber.
All other test statistics reliably distinguish independent versus dependent pairs; in particular, the p-values we obtain using the other test statistics are negligible for all dependent pairs (for ϕ ℓ2 , because we run PERMTEST with T = 99 for computational reasons, we cannot obtain a p-value less than 0.01.Notably, in contrast to our findings, prior work (Xu et al., 2024) argued that the ℓ 2 distance between model parameters is not a reliable indicator of independence, in the sense that the ℓ 2 distance between dependent pairs is sometimes larger than that of independent pairs (similar to the case of ϕ JSD ); the key difference is that Xu et al. (2024) report the raw ℓ 2 distance whereas we obtain p-values from the raw distances using PERMTEST. We hypothesize that PERMTEST effectively standardizes the raw distances. We further evaluated the efficacy of our tests through ablations by training two models with the same OLMo-7B architecture on the same dataset that only differ on the choice of random initialization of randomness, and report results in Appendix E.1. We also verify that Miqu-70B is not independent from Llama 2-70B (Mensch, 2024) and report further details in Appendix E.2.
this section cite: ['b1', 'b23', 'b13', 'b12', 'b25', 'b25', 'b25', 'b14']

Section: Unconstrained setting
For the unconstrained setting, we first assess the previous 21 models of the Llama-7B architecture. We compute ϕ MATCH with the gate and up-projection matrices M = H ℓ gate and M ′ = H ℓ up of each MLP in block ℓ ∈ [L], and aggregate them with FISHER. We obtain the activations in the MLPs by using input sequences sampled from WikiText-103 and computing a forward pass through the full model, with results on all model pairs in Appendix F. We find that the distribution of ϕ MATCH on independent model pairs is close to uniform (Figure 2), whereas across all non-independent model pairs the statistic is at most ε. Unlike the constrained setting, where the p-values are valid by construction, the output of the robust test does not enjoy such theoretical guarantees; however, Figure 2 suggests that even in the unconstrained setting our statistic ϕ MATCH behaves like a p-value.  We also validated our tests on the Mistral architecture-we compared the weights of the hybrid StripedHyena-Nous-7B (Poli et al., 2023) with Mistral-7B-v0.1 and find non-independent parameters via ϕ U (ℓ) . We compute ϕ U (ℓ) on all parameters, which allows us to identify non-independence between specific parameters of the models -such as the self-attention matrices -rather than as models as a whole, and report values of ϕ U (ℓ) among certain parameters in Table 6 in Appendix F.1. From the small p-values, we infer that the embedding layer and some self-attention matrices were likely shared between the two models.
this section cite: []

Section: SIMULATING STRONG-ISH ADVERSARIES
A significant difficulty in evaluating the robustness of our test ϕ MATCH to adversarial transformations is that we cannot exhaustively enumerate all such transformations. Recalling that ϕ MATCH specifically considers the MLP layers contained within two models, we attempt to fool it by randomly reinitializing and retraining these MLP layers individually, thus simulating a somewhat strong adversary.
We reinitialize the first GLU MLP module of a model θ 1 with an MLP with double the width, and using Algorithm 5 (generalized robust test), we train θ1 with random Gaussians as the training distribution P . We retrain each of the 32 MLPs (keeping other layers fixed) of vicuna-7b-v1.5 (a finetune of Llama-2-7b-hf) for 10k gradient steps (until the loss curve plateus). (Additional hyperparameters and a learning curve are in Appendix F.2.) For all 32 runs, we compute ϕ MATCH for the retrained model with the original Llama-2-7b-hf and find ϕ MATCH remains very small between the non-independent models even after an MLP has been retrained. For example, retraining the first MLP module, ϕ (1) MATCH on the first MLP was less than ε = 2.2e-308, indicating that the two models are not independent. We find the same is true for the other MLP layers as well (i.e. ϕ (ℓ) MATCH when evaluated on retrained layer ℓ), with full results in Table 7 of Appendix F.2.
this section cite: []

Section: GENERALIZING TO DIFFERENT ARCHITECTURES
As we describe in Section 2.3.2, we can also apply our test to model architectures which do not contain GLU MLP submodels. For example, the GPT-2 architecture uses a standard 2-layer MLP rather than a GLU MLP. We apply our test (Algorithm 5) to GPT2 PMC and gpt2, where the former is a finetune of the latter (Radford et al., 2019). We use 30k training steps with an isotropic Gaussian input distribution to learn the GLU MLP parameters with which we replace the original MLP submodels in each model. The test yields a value of 3.034e-61, thus distinguishing the two models as dependent. We show additional results on independent and non-independent models (of Llama and GPT architectures) in Appendix F.4.
this section cite: ['b20']

Section: Fine-grained forensics and Localized testing
Finally, we use ϕ MATCH on models pairs with different dimensions, specifically on pruned model pairs, when model dimensions are reduced by preserving only select weights.
In particular, we were able to identify the specific Transformer blocks of Llama-3.1-8B whose weights were likely used in initializing Llama-3.2-3B and Llama-3.2-1B, as Meta reported that the first two models were pruned from the third (MetaAI, 2024). We match ϕ (i,j) MATCH with block i from θ 1 and j from θ 2 , such that ϕ (i,j) MATCH is less than 1e-4. We report the matched layers between the Llama-3.1 and Llama-3.2 models in Figure 3 and in Appendix F.3. We also identify which hidden units were most likely shared between the blocks when MLP dimension is reduced (from 14336 to 8192) during pruning, from the permutation π returned from the up projection matching, MATCH(H
(ℓ) θ1,up , H (ℓ)
θ2,up ).We plot the activation matching for Llama-3.1-8B and Llama-3.2-3B in Appendix F.3.
this section cite: []

Section: Related & Future Work
A related line of work known as model fingerprinting (Xu et al., 2024;Zhang et al., 2025;Jin et al., 2024;Yang & Wu, 2024) plants a secret signal in the weights of a model so that anyone who knows the key can detect the fingerprint from query access to the model (or fine-tunes of the model). For example, Xu et al. (2024) propose fingerprinting a model by fine-tuning on a secret random string; fingerprint detection then resolves to prompting a putative fingerprinted model with a prefix of the string. Unlike Xu et al. (2024), we do not intervene on the training process of the models we test; however, we do require access to model weights.
Finally, a separate line of work on text watermarking aims to attribute model-generated text by planting a watermark when sampling text from the model (Christ et al., 2024;Kirchenbauer et al., 2023;Kuditipudi et al., 2024;Aaronson & Kirchner, 2023). Because it intervenes on sampling, text watermarking is inapplicable to open-weight models, the focus of both model fingerprinting and our setting. Recent work demonstrates that models can directly learn to generate watermarked text but also finds the learned watermark is not robust to further fine-tuning (Gu et al., 2024).
Future work can consider differentiating between fine-tunes of the same base model to reconstruct a complete "family tree" of model lineage is possible (e.g. infer Llemma is a direct fine-tune of CodeLlama) (Yax et al., 2025), and whether robustness against adversarial attacks is solvable with exact guarantees warrants further exploration.
this section cite: ['b25', 'b30', 'b8', 'b25', 'b25', 'b2', 'b10', 'b11', 'b0', 'b7', 'b28']

Section: A. Proofs of Main Theorems
Proof of Theorem 1. From our assumptions on A and P and the fact that {π t } T t=1 are independently drawn, it follows that the collection {π t (θ 1 )} T t=1 comprises T exchangeable copies of θ 1 . The independence of θ 1 and θ 2 thus implies {(π t (θ 1 ), θ 2 )} T t=1 comprises T exchangeable copies of (θ 1 , θ 2 ), and so the claim follows by symmetry -ϕ(θ 1 , θ 2 ) is identically distributed as {ϕ(π t (θ 1 ), θ 2 )} T t=1 , so ϕ(θ 1 , θ 2 ) will have uniform rank among the other values. Ties (ϕ t = ϕ(θ 1 , θ 2 )) randomly contribute to p, so symmetry still holds and the p-values will be uniformly distributed under the null.
Proof of Theorem 2. As M is a Π-equivariant map, if θ 1 ⊥ θ 2 then letting π = LAP(C) in MATCH is equivalent in distribution to sampling π ∼ Unif(Π). Then the output of MATCH is identical in distribution for any pair of independent models, and can be converted to a p-value using SPEARMAN and the distribution for the Spearman correlation coefficient (t-distribution with h -2 degrees of freedom).
Proof of Theorem 3. Let θ ′ 1 ∼ A(π (i) 1 • π (j) 2 (θ 0 1 )) for π 1 , π 2 i.i.d.
∼ Unif(Π). Then θ ′ 1 is an independent copy of θ 1 since taking the composition π
(i) 1 • π (j) 2 (θ 1 ) yields an independent copy of θ 1 for any π 1 , π 2 ∈ Π. From θ 1 ⊥ θ 2 , it follows for ℓ ∈ {i, j} that MATCH(M (ℓ) (θ ′ 1 ), M (ℓ) (θ 2 )) is identically distributed to MATCH(M (ℓ) (θ 1 ), M (ℓ) (θ 2 )).
The result then follows from the fact MATCH is equivariant with respect to permuting the rows of its arguments: in particular, for any π ∈ Π we have MATCH(πW 1 , W 2 ) = πMATCH(W 1 , W 2 ).
this section cite: []

Section: B. Randomized Learning Algorithms
One notable (non-contrived) category of deep learning algorithms that are not permutation equivariant are those with random dropout masks to hidden units during training. In particular, once we fix a specific setting of mask values to specify a deterministic learning algorithm, this algorithm will not be permutation equivariant unless the individual dropout masks are all permutation invariant (which is highly unlikely). We provide a generalized statement of Theorem 1 for randomized algorithms. Definition 5. Let Π ⊂ Θ → Θ. Let π ∈ Π and θ 0 ∈ Θ, with θ ∼ A(θ 0 ), θ = π( θ) and θ ′ ∼ A(π(θ0)). A randomized learning algorithm A : Θ → P(Θ) is Π-equivariant if and only if θ d = θ ′ . Theorem 4. Let ϕ : Θ × Θ → R be a test statistic and Π ⊂ Θ → Θ be finite. Let A : Θ → P(Θ) be Π-equivariant and let P ∈ P(Θ) be Π-invariant. Let θ1, θ2 ∈ Θ be independent random variables, with θ1 ∼ A(θ 0 1 ) for θ 0 1 ∼ P1. Then p = PERMTEST(θ1, θ2) is uniformly distributed on { i T +1 } T i=1 .
Proof. The proof is identical to that of Theorem 1.
this section cite: []

Section: C. Transformer Architecture and Notation
We consider models with the Llama Transformers architecture and define the notation henceforth, although this can easily be extended to other Transformer architectures.
Following the definition of fmlp in Example 1, we can define an abstraction of the full Llama language model architecture consisting of L Transformer blocks sandwiched between an input and output layer. For the sequel, we will abuse notation in applying fmlp to multi-dimensional tensors by broadcasting along the last axis. We use d, n ∈ N to respectively denote the model dimension and sequence length, where ΘLM = Θin × Θ ×L block × Θout with Θblock denoting the parameter space of each Transformer block and Θin, Θout denoting the parameter spaces the input and output layers. We decompose Θblock = Θattn × Θmlp and use frest : Θattn × R n×d → R n×d to denote all remaining parts of the Transformer besides the MLP. The inputs to frest are the input and output of the MLP, and the output of frest is fed directly to the MLP of the next layer. In particular, frest takes the input and output to the MLP of layer i, and first performs the residual connection following the MLP of layer i, then the self-attention and normalization components of layer i + 1, and returns the input to the MLP of layer i + 1. We use fin : Θin × X → R n×d and fout : Θ (L) block × R n×d → Y to respectively denote the input and output layers, i.e. the elements before the first MLP and after the last MLP. Putting everything together gives the following definition of the model; we introduce the notation X (i) θ in the definition as a matter of convenience to track intermediate activations.
Definition 6. (GLU Transformer model) Let θ = (θin, {θ (i) block } L i=1 , θout) ∈ ΘLM and X ∈ X , with θ (i) block = (θ (i) attn , θ (i) mlp ). Then fLM(X; θ) = fout(X (L) θ ; θout) for X (0) θ = fin(X; θin) and X (i) θ = frest(X (i-1) θ , fmlp(X (i-1) θ )).(6)
For a Llama model, table 2 describes the shapes of the model weight matrices for i = 1, . . . , L, for V (vocab size), demb (the hidden dimension), and dmlp (MLP hidden dimension). Following Definition 6, we have θin = (E), θ
block = (θ (i) attn , θ (i) mlp ) where Parameter name Notation embedding E ∈ R V ×demb input layernorm γ input,i ∈ R 1×demb attention query matrix W Q,i ∈ R demb×demb attention key matrix W K,i ∈ R demb×demb attention value matrix W V,i ∈ R demb×demb attention output matrix W O,i ∈ R demb×demb post-attention layernorm γ post-attn, i ∈ R 1×demb MLP gate projection G i ∈ R dmlp×demb MLP up projection U i ∈ R dmlp×demb MLP down projection D i ∈ R demb×dmlp final layernorm γ final ∈ R 1×demb linear output O ∈ R demb×V(i)
post-attn ), θ
mlp = (Gi, Ui, Di), and θout = (γfinal, L). We now describe a forward pass of the model.
this section cite: []

Section: We define the softmax function on a vector
v = (v1, . . . , vn), softmax(v), as softmax(v)i = e v i n k=1 e v k . On batched input X ∈ R N ×n×m where each X (b) = [w1| . . . |wm] ∈ R n×m with column vectors wi, we define the softmax as softmax(X (b) ) = [softmax(w1)| . . . |softmax(wm)], softmax(X) = [softmax(X (1) )| . . . |softmax(X (N ) )].
For a forward pass of the model fLM(X; θ), consider an input sequence of tokens X ∈ {0, 1} N ×V as one-hot vectors where n is sequence length. Then
We feed the input through:
1. (fin) Embedding layer:
X (0) θ = fin(X; θin) = XE ∈ R N ×d emb 2.
(fattn, fmlp, fpost) For each Transformer block i = 0, 1, . . . , L, through fattn, fmlp, and fpost:
(a) Input layernorm:
X (i) LN 1 = X (i) θ Var(X (i) θ ) + ε ⊙ γinput,i
(with variance over the last axis) for some offset ε (typically 1e-6).
(b) Causal multi-head self-attention: Split X (i) LN 1 on the first axis into nheads X (i) LN 1 ,j , . . . , X (i) LN 1 ,nheads . On each head X (i) LN 1 ,j , X (i) SA,j = self-attn(X (i) LN 1 ,j ) = softmax X (i) LN 1 ,j W T Q,i (X (i) LN 1 ,j W T K,i ) T √ demb X (i) LN 1 ,j W T V,i W T O,i
and concatenate X (i) SA,j along the first axis again as X (i) SA . (c) Dropout and residual connection:
X (i) DR 1 = X (i) LN 1 + Dropout(X (i) SA ) (d) Post-attention layernorm: X (i) LN 2 = X (i) DR 1 Var(X (i) DR 1 ) + ε ⊙ γpost-attn,i
(with variance over the last axis) for some offset ε. Then we have fattn(X
(i-1) θ ; θ (i) attn ) = X (i) LN 2 . Parameter name θ π emb (θ) π mlp (θ) embedding E Eπ emb E input layernorm γ input,i γ input,i π emb γ input,i attention query matrix W Q,i W Q,i π emb W Q,i attention key matrix W K,i W K,i π emb W K,i attention value matrix W V,i W V,i π emb W V,i attention output matrix W O,i π T emb W O,i W O,i post-attention layernorm γ post-attn, i γ post-attn, i π emb γ post-attn, i MLP gate projection G i G i π emb π mlp,i G i MLP up projection U i U i π emb π mlp,i U i MLP down projection D i π T emb D i D i π T mlp,i final layernorm γ final γ final π emb γ final linear output O π T emb O O Table 3.
Transformations πemb and πmlp applied to a Llama-architecture model.
(e) Next, we feed through fmlp, the multi-layer perceptron:
fmlp(X (i) LN 2 ; θ (i) mlp ) = X MLP i = [σ(X LN 2 i G T i ) ⊙ (X LN 2 i U T i )]D T i
for some activation σ (e.g., SiLU). (f) Finally, we feed through fpost, dropout and the residual connection:
fpost(θ (i) mlp ) = X (i+1) θ = X DR 1 i + Dropout(X MLP i ) 3. (fout) Final layernorm on the output X (N +1) θ
from the final Transformer block:
X (L) LN = X (L) θ Var(X (L) θ ) + ε ⊙ γfinal
(with variance over the last axis) for some offset ε. Then, linear output embedding and softmax mapping to output probabilities:
fout(X (L) θ ) = softmax(X (L) LN O T ),
which defines the entire forward pass fLM(X; θ).
this section cite: []

Section: D. Model Transformation Class
We describe two sets of equivariant transformations Π on a Transformer model as described in Appendix C. (Abusing notation), the first set, Πemb, consists of elements πemb where πemb ∈ R d emb ×d emb is a permutation matrix. The second set, Πmlp, consists of elements πmlp where πmlp ∈ R d mlp ×d mlp is a permutation matrix. 1. πemb(θ): Applying an embedding permutation πemb ∈ R d emb ×d emb by left or right multiplying all relevant matrices by ξembed (permuting rows or columns).
2. πmlp(θ): Applying MLP permutations πmlp,i ∈ R d mlp ×d mlp to MLP layers.
These permutations are applied such that the outputs of the original model θ and the permuted model Π(θ) remain aligned. We describe the details in Table 3.
this section cite: []

Section: E. Additional Constrained Setting Experimental Results
We report p-values from the statistics ϕ ℓ 2 , ϕ U (ℓ) , and ϕ H (ℓ) on all 210 model pairs (from 21 Llama 2-architecture models) in Figures 4, 5, and 6, where the model names are colored by base model (ground truth). For all statistics, the p-values on independent model pairs are uniformly distributed, while they are all significant at 0.01 (smaller for ϕ U (ℓ) and ϕ H (ℓ) ) for fine-tuned model pairs.    Table 5. Results of ϕ U (ℓ) (aggregated with FISHER) with θ1 as Llama-2-70b-hf and θ2 ranging over the listed models.
this section cite: []

Section: E.1. Identically distributed, Independent models
We further evaluated the efficacy of our tests through ablations by training two models with the same architecture on the same dataset that only differ on the choice of random initialization of randomness. Specifically, we ensure that our test does not incorrectly detect two similar (trained using the same learning algorithm) but independent (randomly initialized) models, as non-independent.
To verify this, we randomly initialized a model with the OLMo (7B) architecture (Groeneveld et al., 2024) and trained it on the Dolma v1 7 dataset ( (Soldaini et al., 2024)). We trained a second model with independently chosen initialization and data ordering.
We keep checkpoints for both seeds after 100M, 1B, 10B, and 18B train tokens and evaluate the statistics ϕ U (ℓ) , ϕ H (ℓ) , and ϕMATCH on the two models at each training checkpoint, reported in Table 4. We highlight that the p-values are broadly distributed, validating our tests support independence even on two similarly-trained but independent models.
this section cite: ['b5', 'b22']

Section: E.2. Tests for Larger Models
Next, we evaluated our tests on larger models. We ran ϕ U (ℓ) on four 70B parameter models with the Llama 2-70B architecture shown in Table 5, and in particular, we verify that Miqu-70B is not independent from Llama 2-70B.
this section cite: []

Section: F. Additional Unconstrained Setting Experimental Results
We report values of ϕMATCH on all model pairs in Figure 7. The statistic is low (< ε = 10 -308 ) for all non-independent model pairs, and uniformly distributed for independent model pairs, empirically acting as a p-value.
this section cite: []

Section: F.1. Striped Hyena Experiments
We report ϕ U (ℓ) on specific parameters from StripedHyena-Nous-7B and Mistral-7B-v0.1 shown in Table 6. ϕ U (ℓ) on parameters from StripedHyena-Nous-7B and Mistral-7B-v0.1, some with low p-values.
this section cite: []

Section: F.2. MLP Retraining Experiments
We retrain each of the 32 MLP layers by feeding in random inputs through the original MLP (gate, up, and down projection matrices.) We train for 10000 gradient steps using MSE loss and an Adam Optimizer with a learning rate of 0.001 and batch size of 5000. A sample learning curve is in Figure 8. The MLP retraining results for all 32 MLP layers of vicuna-7b-v1.5, compared with Llama-2-7b-hf are in Table 7, showing that the statistic is robust to retraining of all layers.
this section cite: []

Section: F.3. Localized Testing
As described in 4.4.2, we can run ϕMATCH on all pairs of Transformer blocks between two models (of different architecture), as long as they share the GLU structure. In addition to the Llama 3 results, we report results of matched blocks on the Sheared-LLaMa and Nvidia-Minitron models, which are both pruned from Llama models.
In particular, we were able to identify the specific Transformer blocks of θ8B = Llama-3.1-8B whose weights were likely used in initializing θ3B = Llama-3.2-3B and θ1B = Llama-3.2-1B, as Meta reported that the Llama-3.2-3B and Llama-3.2-1B models were pruned from Llama-3.1-8B ((MetaAI, 2024)). We use ϕMATCH on all pairs of MLP blocks, where (d θ 8B , h θ 8B , N θ 8B ) = (4096, 14336, 32),(d θ 3B , h θ 3B , N θ 3B ) = (3072, 8192, 28), and (d θ 1B , h θ 1B , N θ 1B ) = (2048, 8192, 16). We match blocks when the statistic ϕ (i,j)
MATCH from block i of model 1 and block j of model 2 is less than 1e-4, reported in Tables 8 and 9 (with the same for the other matchings in this section).
this section cite: []

Section: θ1 θ2
Independent? ϕ
MATCH gpt2 GPT2 PMC ✗ 3.034e-61 gpt2 artgpt2tox ✗ 1.049e-75 gpt2 distilgpt2 ✗ 1.079e-63 Llama-3.2-1B Llama-3.2-3B ✗ 2.011e-70 openai-gpt gpt2 ✓ 0.359 openai-gpt distilgpt2 ✓ 0.770 gpt Llama-3.2-1B ✓ 0.481 Table 13. ϕ (1) MATCH on models distilled with a GLU MLP. Parameter name θ Rot(θ) = θ ′ embedding E ER emb input layernorm γ input, i γ ′ input, i attention query matrix W Q,i
R i W Q,i diag(γ input, i ) R emb diag( 1 γ ′ input, i ) attention key matrix W K,i R i W K,i diag(γ input, i ) R emb diag( 1 γ ′ input, i ) attention value matrix W V,i W V,i diag(γ input, i ) R emb diag( 1 γ ′ input, i ) attention output matrix W O,i R T emb W O,i post-attention layernorm γ post-attn, i γ ′ post-attn, i MLP gate projection G i G i diag(γ post-attn,i ) R emb diag( 1 γ ′ post-attn,i ) MLP up projection U i c i U i diag(γ post-attn,i ) R emb diag( 1 γ ′ post-attn,i ) MLP down projection D i 1 ci R T emb D i final layernorm γ final γ ′ final linear output O O diag(γ final ) R emb diag( 1 γ ′ final )
Table 14. Output-preserving rotation applied to a Llama-architecture model.
this section cite: []

Section: G. Output-Preserving Transformations
An adversary could apply a particular rotation scheme by multiplying weight matrices by an orthogonal rotation matrix U that will also preserve outputs. We describe such a transformation which breaks the invariants proposed by (Zeng et al., 2024) by manipulating layernorms. While this list may not be exhaustive, the following six transformations (with the first two described previously) "camouflage" the language model while preserving outputs:
T1. Permuting the rows of the embedding matrix (and subsequent matrices due to residual connections) by a permutation ξemb ∈ R d emb ×d emb T2. Permuting the MLP matrices (N different permutations for each Transformer block) by permutations ξ1, . . . , ξN ∈ R d mlp ×d mlp T3. Rotating the embedding matrix (and subsequent matrices due to residual connections) by an orthogonal rotation matrix Remb ∈ R d emb ×d emb T4. Rotating the query and key attention matrices (N different rotations for each Transformer block) by orthogonal rotation matrices R1, . . . , RN ∈ R d emb ×d emb T5. Replacing all layernorms (input, post-attention, final) with vectors in R 1×d emb with non-zero elements T6. Scaling the MLP matrices by a constant non-zero factor
Consider a model θ of Llama architecture (Appendix C). Consider orthogonal matrices Remb, R1, . . . R32 as described, as well as new layernorms γ ′ input,1 , . . . , γ ′ input,32 , γ ′ post-attn,1 , . . . , γ ′ post-attn,32 in R 1×d emb with non-zero elements. Finally, consider non-zero constants c1, . . . , c32, which we use to transform the layernorms. We apply the rotation with these parameters to θ, to get a new "rotated" model, Rot(θ). We generalize the set of transformations above as applying Rot(θ) to a model θ".
We transform all the original matrices of θ as in Table 14 (for i = 1, . . . , 32). Note that the transformations T1 and T2 are elements of Πemb and Πmlp and the remaining transformations T3 to T6 are described in Table 14. Importantly, T5 is the transformation that (Zeng et al., 2024)'s invariants are not robust to; our unconstrained setting test ϕMATCH is robust to all 6 transformations, which we show in Table 15.
which follows from Remb being orthogonal. Then we have the output from the unrotated self-attention is w = softmax zW T Q,i (zW
T K,i ) T dkey zW T V,i W T O,i , and the output from the rotated self-attention with input z ′ is softmax   z ′ (RiWQ,idiag(γinput, i)Rembdiag(
1 γ ′ input, i )) T (z ′ (RiWK,idiag(γinput, i)Rembdiag( 1 γ ′ input, i )) T ) T dkey   z ′ (WV,idiag(γinput, i)Rembdiag( 1 γ ′ input, i )) T (R T emb WO,i) T = softmax   z ′ diag( 1 γ ′ input, i )R T emb diag(γinput, i)W T Q,i R T i (z ′ diag( 1 γ ′ input, i )R T emb diag(γinput, i)W T K,i R T i ) T dkey   z ′ diag( 1 γ ′ input, i )R T emb diag(γinput, i)W T V,i W T O,i Remb = softmax   z ′ diag( 1 γ ′ input, i )R T emb diag(γinput, i)W T Q,i WK,idiag(γinput, i)Rembdiag( 1 γ ′ input, i )(z ′ ) T dkey   zW T V,i W T O,i Remb = softmax zWQ,iW T K,i z T dkey zW T V,i W T O,i Remb = wRemb = w ′ .
Then y and y ′ respectively from before the layernorm are added as residual connections as v = y + w and v ′ = y ′ + w ′ = vRemb. v is passed into the post-attention layernorm, which returns
u = LNi(v) = v Var(v) + ε ⊙ γpost-attn,i = v Var(v) + ε diag(γpost-attn,i).
Similar to the input layernorm, the rotated post-attention layernorm on v ′ returns
u ′ = LN ′ i (v ′ ) = v ′ Var(v ′ ) + ε ⊙ γ ′ post-attn,i = vRemb Var(vRemb) + ε ⊙ γ ′ post-attn,i = v Var(v) + ε Rembdiag(γ ′ post-attn,i ) = u diag( 1 γpost-attn,i
)Rembdiag(γ ′ post-attn,i ).
Then the output from the unrotated MLP layer on u is t = [σ(uG T i ) ⊙ (uU T i )]D T i and the output from the rotated MLP on u ′ is
t ′ = [σ(u ′ (Gidiag(γpost-attn,i)Rembdiag( 1 γ ′ post-attn,i )) T ⊙ (u ′ (ciUidiag(γpost-attn,i)Rembdiag( 1 γ ′ post-attn,i )) T )]( 1 ci R T emb Di) T = [σ(u diag( 1 γpost-attn,i )Rembdiag(γ ′ post-attn,i )diag( 1 γ ′ post-attn,i )R T emb diag(γpost-attn,i)G T i )⊙ (ciu diag( 1 γpost-attn,i )Rembdiag(γ ′ post-attn,i )diag( 1 γ ′ post-attn,i )R T emb diag(γpost-attn,i))U T i ] 1 ci D T i Remb = [ciσ(uG T i ) ⊙ (uU T i )] 1 ci D T i Remb = tRemb.
Then the output from the self-attention is added as a residual connection, and the final output from the unrotated Transformer block is s = t + v, and the output from the rotated Transformer block is
s ′ = t ′ + v ′ = sRemb.
Suppose a is the output after all Transformer layers in θ and a ′ is the output after all Transformer layers in θ ′ .
Then the outputs after the final layernorms are b = v Var(a) + ε diag(γfinal) Independence Tests for Language Models b ′ = b diag( 1 γfinal )Rembdiag(γ ′ final ), and the logits from the linear output layer are bO T = b diag( 1 γfinal )Rembdiag(γ ′ final )diag(γfinal)R T emb diag( 1 γ ′ final )O T = b ′ (O ′ ) T , which are the same for both models.
We attempted to undo such a transformation that an adversary may apply by solving the least squares problem: We solve for a rotation A that minimizes |AX -Y | where X is a weight matrix of the first model and Y is the corresponding weight matrix of the second model. Although this will provide a potential rotation to undo this transformation, we find that this solution will also find a matrix A that aligns two independent model pairs as well. This makes undo-ing the rotation this way unreliable. The same holds for X and Y that are activations over multiple inputs.
this section cite: ['b29', 'b29']

Section: 
i 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 j : ϕ (i,j) MATCH (θ 8B , θ 3B ) < 1e -4 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 i 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 j : ϕ (i,j) MATCH (θ 8B , θ 3B ) < 1e -4 16 17 18 19 20 21 22 23 24 25 26 27 28    5 17 18 19 20 21 22 25 27 28 29 31 32 Table 11. θ1 = Sheared-LLaMa 1.3B blocks matched with θ2 = Llama-2-7B blocks using ϕMATCH Finally, we compare Llama 3.1 8B with nvidia/Llama-3.1-Minitron-4B-Depth-Base, a pruned model by reducing from 32 to 16 Transformer blocks and are able to identify the likely shared blocks.
(i,j) MATCH (θ 1 , θ 2 ) < 1e -
i 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 j : 90 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 32 Table 12. θ1 = nvidia/Llama-3.1-Minitron-4B-Depth-Base blocks matched with θ2 = Llama-2-7B blocks using ϕMATCH
ϕ (i,j) MATCH (θ 1 , θ 2 ) < 1e -
this section cite: []

Section: F.4. MLP Distillation Experiments
As we mentioned in section 3.2.2, we present further results on experiments where we distill a model without a GLU MLP and then test the efficacy of our approach. These models do not use GLU MLPS (instead, a different feed-forward network) and a GLU MLP is distilled as the first FFN using Algorithm 5. The cases of two non-independent models still have very small values of ϕ (1) MATCH .
this section cite: []

Section: G.1. Breaking HuREF Invariants
Only transformations T3 and T5 are required to break the invariants from (Zeng et al., 2024). Their first invariant is Ma = E(WQ,i) T WK,i)E T at layer i, and for M ′ with an embedding matrix rotation Remb where the layernorms γinput,i are replaced with γ ′ input,i , we have the invariant is
Ma = E ′ (W ′ Q,i ) T ((W ′ K,i ) T ) T E ′T M ′ a = (ERemb) diag( 1 γ ′ input,i )R T emb diag(γinput,i)W T Q,i R T i RiWK,idiag(γinput,i)Rembdiag( 1 γ ′ input,i ) (R T emb E) = ERembdiag( 1 γ ′ input,i )R T emb diag(γinput,i)W T Q,i WK,idiag(γinput,i)Rembdiag( 1 γ ′ input,i )R T emb E,
and in general Ma ̸ = M ′ a unless the layernorm weights are equal constants. The other two invariants also do not hold due to changing the layernorms. (Note that our notation for Transformers is different than theirs.) Assuming in their invariant M f that W1 and W2 are the gate and down projection matrices of an MLP (this is not stated explicitly in the paper but can be inferred from experiments), the remaining invariants do not hold either.
Empirically, we compute the invariants between Llama2-7b and independently trained models and between Llama2-7b and rotated finetuned models (including Llama2-7b) in Table 15. We can see there is little distinction between the independent vs. non-independent model pairs.  (Zeng et al., 2024) between Llama-2-7b-hf and independent and nonindependent models.
θ1 = Llama-2-7b-hf, θ2 = Independent? Ma M b Mc ϕMATCH ϕ U (ℓ) ϕ H (ℓ) ϕJSD vicuna-7b-v1.5 ✗ 1.0 0.9883 0.9922 < ε < ε < ε -10.874 Nous-Hermes-llama-2-7b ✗ 1.0 1.0 1.0 < ε < ε < ε -12.101 llama-7b-hf ✓0
this section cite: ['b29', 'b29']

Section: G.2. Invariance of Outputs under Rotation
These transformations are particularly important because they preserve outputs as we show in Theorem ??, and hence generally can go undetected, though ϕMATCH is robust to them.
Theorem 5. For any input sequence X ∈ {0, 1} n×V , the outputs of models θ and Rot(θ) = θ ′ are aligned, i.e. fLM(X; θ) = fLM(X; θ ′ ).
Proof. First, note that an element-wise product of two one-dimensional vectors is equivalent to multiplying by the diagonal matrix of the second vector, i.e. for v, γ ∈ R 1×m , v * γ = vdiag(γ).
We use this in our layernorm calculations.
Let the output from the unrotated embedding layer be y = fin(X, E) = EX (for X ∈ {0, 1} n×V ).
Then the output from the rotated embedding layer is y ′ = fin(X, E ′ ) = (ERemb)(x) = yRemb. Now consider Transformer block i with input y and the rotated Transformer block with input yRemb. y is passed into the input layernorm, which returns z = LNi(y) = y Var(y) + ε ⊙ γinput,i = y Var(y) + ε diag(γinput,i).
The rotated input layernorm on y ′ returns
z ′ = LN ′ i (y ′ ) = y ′ Var(y ′ ) + ε ⊙ γ ′ input,i = yRemb Var(yRemb) + ε ⊙ γ ′ input,i = y Var(y) + ε Rembdiag(γ ′ input,i ) = z diag( 1 γinput,i )Rembdiag(γ ′ input,i ),
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2023)
Ref_id:b1 Title: An Open Language Model for Mathematics Year: (2024)
Ref_id:b2 Title: Undetectable Watermarks for Language Models Year: (2024-07-03)
Ref_id:b3 Title:  Year: (2024)
Ref_id:b4 Title: The Llama 3 Herd of Models Year: (2024)
Ref_id:b5 Title: Accelerating the science of language models Year: (2024-08)
Ref_id:b6 Title: URL Year: ()
Ref_id:b7 Title: On the Learnability of Watermarks for Language Models Year: (2024)
Ref_id:b8 Title: ProFLingo: A Fingerprinting-based Intellectual Property Protection Scheme for Large Language Models Year: (2024)
Ref_id:b9 Title: Position: On the Societal Impact of Open Foundation Models Year: (2025)
Ref_id:b10 Title: A Watermark for Large Language Models Year: (2023-07)
Ref_id:b11 Title: Robust Distortion-free Watermarks for Language Models Year: (2024)
Ref_id:b12 Title: 3621ee907def47c1b952ade25c67698-Paper-Conference. pdf. Lin, J. Divergence measures based on the Shannon entropy Year: (2006)
Ref_id:b13 Title: Towards Fully Transparent Open-Source LLMs Year: (2024)
Ref_id:b14 Title: Mistral CEO confirms Miqu model leak Year: (2024-08)
Ref_id:b15 Title: Pointer Sentinel Mixture Models Year: (2017)
Ref_id:b16 Title: Llama 3.2: Revolutionizing edge AI and vision with open, customizable models Year: (2024)
Ref_id:b17 Title: Questions and Answers Year: (1948)
Ref_id:b18 Title: Intellectual Property Protection of DNN Models Year: (2023-07)
Ref_id:b19 Title: Moving Beyond Transformers with Hybrid Signal Processing Models Year: ()
Ref_id:b20 Title: Language Models are Unsupervised Multitask Learners Year: (2019)
Ref_id:b21 Title: On Minimum-Cost Assignments in Unbalanced Bipartite Graphs Year: (2012)
Ref_id:b22 Title: Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research Year: (2024-08)
Ref_id:b23 Title: LAB: Large-Scale Alignment for ChatBots Year: (2024)
Ref_id:b24 Title: Open Foundation and Fine-Tuned Chat Models Year: (2023)
Ref_id:b25 Title: Instructional Fingerprinting of Large Language Models Year: (2024-06)
Ref_id:b26 Title: URL Year: ()
Ref_id:b27 Title: A Fingerprint for Large Language Models Year: (2024)
Ref_id:b28 Title: PhyloLM: Inferring the Phylogeny of Large Language Models and Predicting their Performances in Benchmarks Year: (2025)
Ref_id:b29 Title: HUman-REadable Fingerprint for Large Language Models Year: (2024)
Ref_id:b30 Title: REEF: Representation Encoding Fingerprints for Large Language Models Year: (2025)
