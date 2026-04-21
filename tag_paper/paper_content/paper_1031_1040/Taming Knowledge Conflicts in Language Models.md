Title: Taming Knowledge Conflicts in Language Models
Abstract: Language Models (LMs) often encounter knowledge conflicts when parametric memory contradicts contextual knowledge. Previous works attribute this conflict to the interplay between "memory heads" and "context heads", attention heads assumed to promote either memory or context exclusively. In this study, we go beyond this fundamental assumption by uncovering a critical phenomenon we term the superposition of contextual information and parametric memory, where highly influential attention heads simultaneously contribute to both memory and context. Building upon this insight, we propose Just Run Twice (JUICE), a test-time attention intervention method that steers LMs toward either parametric beliefs or contextual knowledge without requiring fine-tuning. JUICE identifies a set of reliable attention heads and leverages a dual-run approach to mitigate the superposition effects. Extensive experiments across 11 datasets and 6 model architectures demonstrate that JUICE sets the new state-of-the-art performance and robust generalization, achieving significant and consistent improvement across different domains under various conflict types. Finally, we theoretically analyze knowledge conflict and the superposition of contextual information and parametric memory in attention heads, which further elucidates the effectiveness of JUICE in these settings. Our code is available at https:  //github.com/GaotangLi/JUICE.

Section: Introduction
Language Models (LMs) store vast amounts of information during pretraining as parametric knowlege. During
this section cite: []

Section: Our Finding
Figure 1. Our finding goes beyond the prior notion of exclusive "memory head" and "context head", where we show that memory and contexts are encoded in attention heads in superposition. inference, they leverage this parametric memory alongside the provided context to generate the next token. However, conflicts can arise when parametric memory contradicts contextual information-a phenomenon known as knowledge conflict (Xu et al., 2024). In such cases, the model may become uncertain about which source of knowledge to trust. These conflicts are particularly prevalent in real-world applications, especially in context-heavy Large Language Models (LLMs) systems like retrieval-augmented generation (RAG) (Gao et al., 2023), LLM agents (Xi et al., 2025), and tool-augmented LLMs (Qu et al., 2025). Depending on the application, user may require an LLM to either remain faithful to its parametric memory or prioritize contextual reliance for accurate and reliable outputs.
Prior works have explored the behavior of LMs under knowledge conflicts, either by treating the model as an oracle to analyze how different contexts influence its predictions (Xie et al., 2024) or by treating the context as an oracle to evaluate how effectively the model follows it (Longpre et al., 2021). While these studies provide valuable insights into knowledge conflicts, the intrinsic mechanisms underlying these conflicts and corresponding mitigation strategies largely remain unexplored. Some studies have taken important steps to characterize (Yu et al., 2023) and intervene (Jin et al., 2024b) in knowledge conflicts, primarily focusing on a single conflict type (e.g., substitution-based conflicts). While pioneering, these efforts leave opportunities for more comprehensive understanding of diverse conflict types and the development of fine-grained approaches to address knowl- edge conflicts. In addition, much of the existing literature predominantly adopts a single-sided perspective on knowledge conflict, focusing on enhancing contextual reliance and addressing issues commonly referred to as "RAG hallucination" (Goyal et al.;Huang et al., 2023;Shi et al., 2024b).
In contrast, we advocate for a unified method capable of flexibly steering the model toward either parametric or contextual knowledge, offering broader utility.
In this paper, we begin by treating LMs as an oracle and considering the setting of factual recall, a task requiring pure memorization. We then treat contexts as providing misleading information (Shi et al., 2023) and systematically explore various types of knowledge conflicts over diverse domains, including sentence-level (substitution), and paragraph-level (coherent) conflicts (Sec. 2), to uncover their underlying mechanisms and design effective intervention strategies. Starting with empirical analysis, our findings go beyond the hypothesis posited in (Jin et al., 2024b) that model components exclusively contribute to either parametric or contextual knowledge, uncovering the phenomenon of "superposition of contextual information and parametric memory" (CP superposition), as shown in Fig. 1. We revealed the inconsistent behaviors of model components under different degrees of knowledge conflicts and the counteracting effects of multiple individually effective interventions.
Building on these insights, we propose Just Run Twice (JUICE), a simple yet effective method for steering LMs towards either parametric or contextual knowledge without finetuning. JUICE operates in two stages: (1) a head identification stage, where two sets of attention heads that yield consistent improvements with positive or negative scaling are identified using a minimal number of samples, and (2) an dual-run inference stage, where the model runs twice: first saving the outputs of the identified heads, and then using scaled versions of these saved outputs to intervene during the second run. Intuitively, this approach ensures that the identified components are consistently effective, mitigating the superposition effects, and therefore provide more accu-rate steering directions through residual head activations.
We evaluate JUICE in two distinct settings: enhancing parametric beliefs and enhancing contextual reliance. For the first setting, we use six factual association datasets covering diverse domains, each tested under three levels of knowledge conflict. In the second setting, we evaluate five datasets spanning diverse fields and formats, including open-domain question answering and sentence completion. Extensive experimental results demonstrate the consistent state-of-the-art performance of JUICE. Fig. 2 illustrates the strong performance of JUICE under the Gemma-2b model, with detailed results provided in Tab. 3. We also show the robustness of JUICE against key hyperparameters and paraphrased input.
Finally, we analyze our empirical observations from a theoretical perspective, conceptualizing knowledge conflict as the result of conflicting tasks at inference, which arise from distinct tasks during training. In a succinct setup, we demonstrate the existence of attention heads that simultaneously contribute to both parametric and contextual knowledge and show how standard training encourages the formation of such heads. We further provide theoretic justifications for the effectiveness of JUICE under these settings.
Our main contributions can be summarize as follows:
• Problem. We conduct a systematic and principled study of knowledge conflicts in LMs, considering both parametric and contextual perspectives and covering various types of datasets over diverse domains.
• Mechanism. We reveal the limitations of naive intervention methods by uncovering a critical phenomenon we term the "superposition of contextual information and parametric memory", where the relative role of a model component in parametric versus contextual knowledge is not exclusive.
• Algorithm. We propose JUICE, a simple yet effective method to steer an LM toward parametric or contextual knowledge without finetuning, leveraging a dual-run approach to mitigate the superposition effects.
• Experiment. Through extensive experiments across 11 datasets and 6 architectures, we set the new state-of-theart performance and robust generalization, achieving significant and consistent improvements.
• Theory. We provide a theoretical analysis of knowledge conflicts, conceptualizing the superposition of contextual information and parametric memory. This analysis further justifies the effectiveness of JUICE under these conditions.
this section cite: ['b58', 'b14', 'b56', 'b43', 'b57', 'b32', 'b61', 'b18', 'b20', 'b47']

Section: Problem Setup
In this paper, we study how language models respond to varying degrees of knowledge conflict and propose methods to regulate these behaviors. We identify two complementary perspectives on knowledge conflict: (1) when the input context is irrelevant or potentially misleading, we treat the LM as an oracle, aiming to enhance its reliance on parametric beliefs;
(2) when the input context is accurate, but the LM's prior knowledge may be outdated or incorrect, we aim to increase the model's dependence on contextual knowledge. Both perspectives hold intrinsic value and merit further investigation.
this section cite: []

Section: Parametric Datasets
In this setup, we treat the input context as potentially misleading information and the language model as an oracle.
For our study, we carefully curate six datasets encompassing distinct types of knowledge conflicts in factual recalls. Below, we detail the specific design choices differing from prior studies and the underlying rationales:
this section cite: []

Section: Diverse Factual Domains:
We create six datasets spanning various domains of factual knowledge: World Capital, Athlete Sport, Book Author, Official Language, Company Headquarter, and Company Founder. This setting will allow us to investigate the transferrability across unrelated domains of intervention methods, a critical aspect that is missing in the prior work (Jin et al., 2024b;Yu et al., 2023).
this section cite: ['b61']

Section: Sentence-level Conflict (Substitution-based):
This is the exclusive approach adopted in prior works (Yu et al., 2023;Jin et al., 2024b). A typical input takes the form (e.g., "The name of the capital city of {s} is {a c }. The name of the capital city of {s} is"), where a c represents the substituted contextual answer that conflicts with the parametric answer a p . In our experiment, we aim to enhance the model's ability to output a p , despite the conflicting presence of a c .
this section cite: ['b61']

Section: Paragraph-level Conflict (Coherent Counterfactual):
Recent work (Xie et al., 2024) demonstrates that language models rely more on context when it is coherent. In this scenario, the context extends beyond a single substitution, reinforced by coherent and persuasive evidence, often generated by advanced models like GPT-4. This presents a highly challenging case, as models almost inevitably output the contextual answer a c over the parametric answer a p . In our experiment, we focus on enhancing the model's ability to output a p , despite these difficult conditions.
There is also a trivial type of knowledge conflict: when no conflict is present, in which case we still expect the model to respond faithfully. Detailed examples are provided in Appen. C. Importantly, different from (Xie et al., 2024), which focuses solely on altering the model's predictions regardless of their correctness, we explicitly ensure that conflicting contexts include factually incorrect answers. For evaluation, we primarily rely on the exact match (accuracy) metric with respect to the factually correct answer. Our curated dataset is available at https://huggingface.  co/datasets/gaotang/ParaConfilct.
this section cite: ['b57', 'b57']

Section: Contextual Datasets
In this setup, we treat the input context as the desired target and consider the prior knowledge of the language model as an unreliable source of information. This approach enables a more unified and versatile evaluation of baseline methods.
Since this setup has been extensively studied, we adopt the dataset choice of a seminar work (Shi et al., 2024b) by using two context-oriented knowledge conflict benchmarks: Memo-Trap (Liu & Liu, 2023) and NQ-Swap (Longpre et al., 2021). The details of these datasets can be found in Appen. D. We evaluate performance using exact match (accuracy) with respect to the contextual answer.
this section cite: ['b31', 'b32']

Section: Models
We benchmark our studies using six existing open-sourced base language models: Gemma-2b (Team et al., 2024), Llama2-7B (Touvron et al., 2023), Llama3-8B (Dubey et al., 2024), Phi2-2.7b (Javaheripi et al., 2023), StableLm2-1.6b (Bellagente et al., 2024), and Olmo-7b (Groeneveld et al., 2024). We conduct our analysis in Sec. 3 mainly using Gemma and evaluate the effectiveness of the intervention methods using all backbone models.
this section cite: ['b50', 'b51', 'b9', 'b21', 'b2', 'b19']

Section: Interpreting and Resolving Knowledge Conflicts
In this section, we analyze how the internal structure of language models (LMs) influences their parametric versus contextual tendencies through causal analysis. We quantify these tendencies by measuring the expected change in the probability of the output token (parametric versus contextual) when perturbations are applied to specific model components. These perturbations are implemented by scaling the activation outputs. Formally, given a distribution over input triplets (X, y p , y c ), where X := {x i } n i=1 is the input prompt set, encompassing various conflicting forms (e.g., clean input, substitution conflicts, and coherent conflicts), y p and y c represent the parametric and contextual answers, respectively, we measure:
E (x,y) P y |x, do(M (i) = αM (i) ) -P (y|x) . (1)
Here, M (i) refers to a specific model component with index i, and y is set to either y p or y c upon our needs. While (x, y) can be drawn from an arbitrary distribution, we use Gemma and World Capital as a concrete example in this section.
Previous works analyzing model internals typically adhere to two "locate-and-edit" principles (Xu et al., 2024): 0 5 10 15 20 25 Layer Number 0.8 0.6 0.4 0.2 0.0 Change in Probs Influence of Knock Out -Clean Entire Layer Attention MLP 0 5 10 15 20 25 Layer Number 0.2 0.1 0.0 0.1 0.2 Change in Probs Influence of Knock Out -Substitution Conflict Entire Layer Attention MLP 0 5 10 15 20 25 Layer Number 0.0 0.2 0.4 0.6 0.8 Change in Probs Influence of Knock Out -Coherent Conflict Entire Layer Attention MLP Figure 3. Influence of Knock Out (Zero Out) Model Components in changing the probability of outputting the parametric answer tokens (ap) on the World Capital dataset. Three different scenarios are considered: clean inputs, substitution conflict inputs, and coherent conflict inputs. We find that (1) removing (nearly) all components leads to decreases in probability of outputting ap in clean prompts, (2) removing components leads to both increase and decrease in outputting ap in substitution conflict prompts, and (3) removing (nearly) all components leads to increases in probability of outputting ap in coherent conflict prompts.
• Identify a circuit (specific model components) that is exclusively responsible for a particular functionality.
• Apply targeted interventions to these circuits to achieve the desired control or behavior.
In our motivating experiments, we demonstrate the need for additional criteria when performing interventions to address the complexities of model internals and knowledge conflicts.
this section cite: ['b58']

Section: Analysis
Observation 1: Inconsistent Behaviors of Model Components Under Different Degrees of Knowledge Conflict.
In our first set of experiments, we examine how model components exhibit significantly different functionalities when faced with varying degrees of knowledge conflict. We set M (i) to represent either the entire MLP, attention module, or both within layer i. For the intervention method, we fix it to be knocking out (i.e., zero-ablating). The goal is to promote parametric knowledge, setting y = y p in Eq. 1. Fig. 3 illustrates these findings, revealing the following trends: (1) removing (nearly) all components decreases the probability of outputting parametric answers for clean prompts;
(2) removing components leads to both increase and decrease in outputting parametric answers for substitution conflicts; and
(3) removing (nearly) all components increases the probability of outputting parametric answers for coherent conflict prompts. Quantitatively, the number of components yielding consistent parametric gains across all three conflict types is 0 for the entire layer, 1 for the MLP module, and 6 for the Attention module (out of 26 layers in Gemma). These results suggest that the same model component may exhibit different influences on parametric and contextual knowledge depending on residual streams received from prior layers.
Prior work (Jin et al., 2024b) introduces the notion of "mem-ory heads" and "context heads", positing that there are attention heads exclusively responsible for promoting parametric or contextual knowledge. Specifically, promoting contextual knowledge involves knocking out parametric heads, and vice versa. While this approach achieves success in single-typed conflicts, we find its limitations when extended to multiple kinds of conflicts. Tab. 1 ranks the top-4 memory heads based on their effectiveness in substitution conflicts and evaluates their influence in coherent conflicts. Surprisingly, half of the top-performing "memory heads" in substitution conflicts become "context heads" in coherent conflicts. This shows that even the most influential model component could have completely opposite functionality.  reduce performance. This behavior likely arises from the dependence of a model component's functionality on input residual streams, as highlighted in Observation 1. Modified activations from earlier layers may alter downstream behavior, leading to counteracting effects. Our findings collectively suggest a phenomenon we term the "superposition of contextual information and parametric memory" (CP Superposition), where the roles of "context" or "memory" of model components depend on the inputs they receive. Next, we discuss how we could propose effective methods while acknowledging such superpositions.
this section cite: []

Section: Our Approach: Just Run Twice (JUICE)
We introduce Just Run Twice (JUICE), a test-time intervention method for addressing knowledge conflicts. Fig. 4 illustrates the core idea, and Alg. 3 provides the detailed algorithm. JUICE operates in two stages.
Stage 1 (Head Identification). This stage identifies two sets of attention heads that consistently achieve the desired effect with either positive or negative scaling. Each head is assigned a score based on the expected change in the desired probability value under individual scaling, computed across a small head selection dataset spanning multiple conflict types. To ensure consistency, only heads with non-negative scores across all conflict types are selected. The top K, based on aggregated scores, are retained. This process ensures reliability for individual head activations.
this section cite: []

Section: Stage 2 (Dual-run Inference).
To mitigate counteracting effects from multiple interventions, the model runs twice.
In the first run, the outputs of the identified heads are saved.
In the second run, scaled versions of these saved outputs are added to the corresponding activations. Intuitively, the firstrun activations serve as more reliable steering directions. We validate this intuition through experiments in Sec. 4.4 and analyses in Sec. 5.
this section cite: []

Section: Practical Implementation.
The key hyperparameters of JUICE include the size of the head selection dataset D, the number of intervened heads K, and the scaling factors at inference. In practice, we fix K to be a constant number (e.g., 5) and determine the scaling factors using the validation set. We fix |D| to be 4 for all primary experiments. Additionally, we test the generalizability of JUICE by using a head identification set from a single domain and evaluating its performance across other domains.
this section cite: []

Section: Intervention Experiment
In this section, we analyze the intervention performance of JUICE and compare it against different baselines. Due to the page limit, we only present three models in the main paper. A more comprehensive experiment section with additional model results can be found in Appen. D.
this section cite: []

Section: Enhancing Parametric Beliefs
Setups. We use the datasets and evaluation metric detailed in Sec. 2.1. Notably, we have three different conflict types: No Conflict (Type 1), Substitution Conflict (Type 2), and Coherent Conflict (Type 3). For presentation clarity, we use the number to represent these conflict types in Tab. 3.
Baselines. We compare our methods against the following baselines: (1) Prompt: We instruct the LM to generate answers solely based on internal memory; (2) PH3: (Jin et al., 2024b) leverages patching-based methods to identify and prune "context" and "memory" heads, demonstrating 5. While PH3 bears an appealing ability to identify "crossrelation heads" (Jin et al., 2024b), its transferability is largely limited to closely related datasets (i.e., heads identified from the world capital dataset are effective for the official language dataset but not for the company headquarters dataset). In contrast, our method achieves high performance across diverse domains, with heads only being selected from the world capital domain.
this section cite: []

Section: Enhancing Contextual Reliance
Setups and Baselines. We use the datasets and evaluation metric detailed in Sec. 2.2. We compare our methods against the previously mentioned baselines and an additional one: CAD (Shi et al., 2024b), a decoding-based method that leverages contrastive decoding (Li et al., 2023b) to encourage the language model to attend to its context.
Results. Tab. 4 presents the results of these intervention methods across the models. The main conclusions from the prior subsection are still valid. JUICE consistently outperforms all baselines on average and is versatile in promoting contextual knowledge as well.
this section cite: []

Section: Robustness of JUICE
In this section, we examine the robustness of JUICE against variations in key hyperparameters and paraphrased prompts. Using Gemma as our backbone model, we systematically vary one hyperparameter at a time to isolate its effects on performance. Specifically, we evaluate the impact of three hyperparameters: the size of the head identification set |D|, the number of intervened attention heads K, and the magnitude of the scaling factors. Additionally, we investigate robustness to paraphrased prompts by employing multiple curated templates for each conflict type, selecting one at random during evaluation. Detailed experimental setups and additional analyses are provided in Appendix D.3.
Figure 5 illustrates the robustness of JUICE across these hyperparameters. The results demonstrate that JUICE maintains consistently high performance across a wide range of hyperparameter values, highlighting its stability and effectiveness.
Tab. 7 in Appendix D.3 presents the results of JUICE when applied to paraphrased prompts. Our findings show that JUICE is highly robust to variations in input prompt formats, consistently maintaining its effectiveness across diverse templates. Notably, JUICE continues to demonstrate superior performance, effectively shifting the model's reliance from context to parametric memory.
this section cite: []

Section: JUNE vs. JUICE: Effect of Running Twice
We conduct an additional experiment to demonstrate the effectiveness of the dual-run design. Following the same setup as in Tab. 2, we compare the intervened logit value of Run Once versus Run Twice when combining multiple individually effective interventions. As shown in Fig. 6, single-pass interventions are unstable and prone to performance degradation. In contrast, the dual-run design delivers consistently effective interventions.
this section cite: []

Section: Theoretical Analysis
In the previous sections, we have conducted a comprehensive empirical analysis to identify the phenomenon of CP superposition and demonstrated the effectiveness of JUICE across a variety of setups. In this section, we aim to formalize our observations and understand the underlying mechanisms behind both observations. Specifically, we conceptualize knowledge conflicts as arising naturally within the weight matrices of the attention module, shaped through the training process via gradient descent. Under such condi-tions, we elucidate that JUICE provides a superior approach compared to naive single-pass interventions. A more detailed theoretical analysis can be found in Appen. G. We first provide a brief overview of the model and task setup.
Model Setup. We use a two-layer Transformer with one attention head per layer, absolute positional encoding, and residual connections. The input is a sequence of tokens z 1:T ∈ [N ] T , where T is the sequence length, and N is the vocabulary size. Each token z t is mapped to a d-dimensional embedding ϕ(z t ), and a positional embedding p t ∈ R d is added. The input to the model is: x T := ϕ(z t ) + p t for t = 1, . . . , T . We denote X (l) = [x 1 , . . . , x T ] as the representation of the embeddings at layer l. These embeddings are updated through two layers as follows:
X (l+1) = X (l) + W (l) OV X (l) σ MSK ⊙ X (l) W (l) KQ X (l)
where σ is the column-wise softmax function. Finally, the embeddings are mapped back to the vocabulary space through a linear layer parameterized by W lin ∈ R d×N . The i-th column vector is denoted as µ(i).
Task Setup. We consider two tasks in parallel: Factual Recalls and Induction. They correspond to parametric and contextual tasks, respectively. A diagram illustration of the whole theoretical task setup can be found in Fig. 7.
In the factual recall task (Nichani et al., 2024), the goal is to learn associations between the subject token space S and the answer token space A, based on a bijective ground truth mapping G * : S → A. This models knowledge triples like (China, capital, Beijing), where the subject token (China, capital) maps to the answer token (Beijing). Non-critical tokens like "the" and "of" also constitute part of a factual sentence, and we assume these tokens are from the noise token space N . Sequences z 1:T +1 ∈ [N ] T +1 are generated as follows:
1. Sample a fact s ∈ S and index i ∈ [T -1] uniformly at random, and set z i = s.
2. For all k ∈ [T -1]\{i}, sample z k uniformly from N without replacement.
3. Set z T = q, the query token and z T +1 = G * (s).
In the induction task (Olsson et al., 2022), the goal is to predict a token b ∈ N following the second occurence of a trigger word q (e.g. ...qb. . . q → b). Sequences z 1:T +1 ∈ [N ] T +1 are generated as follows:
1. Sample j ∈ [T -2]\{1} uniformly, set z j = q, and sample z j+1 from N .
2. For all other token z k , sample uniformly at random from N \{z j+1 } without replacement.
3. Set z T = q and z T +1 = z j+1 .
In summary, the vocabulary space consists of V = S ∪ A ∪ {q} ∪ N . We remark that we use the same trigger token q as the fixed query token in the factual recall task to induce knowledge conflicts.
Assumption 5.1 (Near-orthogonal Initialization). All embedding, unembedding, and positional vectors are initialized randomly.
This ensures near-orthogonality among all embeddings and unembeddings, such that ⟨ϕ(z i ), ϕ(z j )⟩ ≈ δ ij (1[i = j]) when the embedding dimension d is large. Our setting is similar to recent works (Bietti et al., 2024;Ghosal et al., 2024;Jiang et al., 2024b;Nichani et al., 2024).
this section cite: ['b40', 'b41', 'b3', 'b17', 'b40']

Section: CP Superposition
We first examine how knowledge conflict arises in our simplified model, starting by demonstrating its existence.
this section cite: []

Section: Proposition 5.2 (Existence of a Perfect Solver).
There exists a two-layer transformer that can solve both induction and factual recall tasks with the perfect accuracy.
The construction can be achieved as follows. By setting W
OV as a random matrix and defining
W (1) KQ = C T -1 t=1 p t p ⊤ t+1 ,(2)
W (2) KQ = C 1 W 1 OV ϕ(q) ϕ(q) ⊤ + C 2 s∈S ϕ(s)ϕ(q) ⊤ , (3) W (2) OV = C 3 k∈N µ(k)ϕ(k) ⊤ + C 4 s∈S µ (G * (s)) ϕ(s) ⊤ ,(4)
where C 1 , C 2 , C 3 , C 4 are appropriate scaling factors and C is a large constant. In this setup, the first layer implements a "copy from previous embedding" behavior, while the second layer learns the critical tokens and associated memory required for the tasks. Notably, the construction of the second layer inherently forms a superposition, which leads to knowledge conflicts.
Next, we analyze how this construction could naturally emerge from training via gradient descent with a crossentropy loss over the two tasks. We assume a perfectly learned first layer and focus on the dynamics of the second layer, as it suffices to illustrate the core idea. For simplicity, we assume a linear attention model and strictly orthogonal embeddings (i.e., all initialized vectors are orthogonal), which are common in the existing literature (Li et al., 2023c;Ahn et al., 2023;Zhang et al., 2024;Mahankali et al.). The top row shows two distinct tasks that a two-layer transformer learns during training; the bottom row depicts the conflicting task encountered at inference. Here, zj denotes noisy tokens, s is the subject token, a is the answer token associated with s, and q is the trigger and fixed query (EOS) token.
Proposition 5.3 (Learning the Second Superposition Layer via Gradient Descent, Informal). In a simplified setup using one-layer attention only transformer, the superposition head as constructed in Eq.3 and Eq.4 can be trained via gradient descent from zero initialization using the cross-entropy loss.
We defer the proof to Appen. G. This proposition tells us that the standard training objectives of language models encourages superposition. In practice, the first layer may also learn associative memories required by different tasks. Such formulation of the weight matrices naturally results in knowledge conflicts at the inference time.
this section cite: ['b0', 'b65', 'b34']

Section: Knowledge Conflict
We now define and analyze the knowledge conflict task:
1. Sample an index j ∈ [T -2]\{1}, set z j = q, and sample z j+1 from N .
2. Sample an index i ∈ [T -1]\{j, j + 1} and s ∈ S. Set z i = s.
3. Set z T = q.
this section cite: []

Section: Corollary 5.4 (Knowledge Conflict).
Under the knowledge conflict inference setting, the model capable of solving both factual recall and induction from Proposition 5.2 may output either the inductive token or the factual token. More specifically, if exp(C 1 )C 3 < exp(C 2 )C 4 , then the model outputs the factual recall answer G * (s); otherwise, the model outputs the induction answer z j+1 .
This corollary highlights how distinct, well-defined training tasks can overlap at inference. The conflict arises naturally due to the associative memory structure of the weight matrices tied to specific tokens. The model's output preference depends on the relative strengths of coefficients C 1 , . . . , C 4 , which are influenced by factors like the learning rate and the number of (task) samples. Notably, the coefficient C i should be sample-dependent in practice. (Yu et al., 2023) found that models are more likely to generate the parametric answer when the corresponding fact appears frequently in the pretraining data, aligning with our results.
Finally, we manifest the effectiveness of the dual-run design over single-pass intervention.
Proposition 5.5 (Effectiveness of JUICE). Consider the model from Prop. 5.2 and the case when its inductive part dominates (i.e., exp(C 1 )C 3 >> exp(C 2 )C 4 ), then the intervention by JUNE/PH3 of deleting the two attention heads is not as effective as JUICE. In particular, in this case JUNE/PH3 does not result in the parametric answer, while JUICE does.
Both attention heads from Prop. 5.2 can be identified as "influential context heads" in the above setting. However, when the first head is removed, the second head no longer functions for the induction task but instead transitions into a factual memorizer. A single-pass intervention method may still remove the second head, as it was initially classified as a "context head". By instead deleting activations from the original run-arguably a more reliable source-JUICE achieves more precise control over the model's behavior and steers it as desired.
this section cite: ['b61']

Section: Conclusion
This work presents a unified and principled study of knowledge conflicts in language models, revealing the phenomenon of superposition of contextual information and parametric memory. We propose Just Run Twice (JUICE), a simple yet effective test-time intervention that reliably steers models toward either parametric beliefs or contextual information without requiring fine-tuning. JUICE consistently and significantly achieves effective intervention performance across different datasets under various conflict types. Our theoretical analysis further reveals the underlying mechanisms of knowledge conflict and the effectiveness of JUICE. These findings not only enhance our fundamental understanding of LMs' knowledge representation mechanism but also offer a practical method for improving model controllability in real-world applications. We discuss possible limitations and future works in Appen. F.
this section cite: []

Section: References
Ref_id:b0 Title: Transformers learn to implement preconditioned gradient descent for in-context learning Year: (2023)
Ref_id:b1 Title: Linear algebraic structure of word senses, with applications to polysemy Year: (2018)
Ref_id:b2 Title: Stable lm 2 1.6 b technical report Year: (2024)
Ref_id:b3 Title: Birth of a transformer: A memory viewpoint Year: (2024)
Ref_id:b4 Title: Scaling laws for associative memories Year: (2023)
Ref_id:b5 Title:  Year: (2020)
Ref_id:b6 Title: Language model behavior: A comprehensive survey Year: (2024)
Ref_id:b7 Title: Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence Year: (2022-12)
Ref_id:b8 Title: URL Year: ()
Ref_id:b9 Title: The llama 3 herd of models Year: (2024)
Ref_id:b10 Title: A mathematical framework for transformer circuits Year: (2021)
Ref_id:b11 Title: Toy models of superposition Year: (2022)
Ref_id:b12 Title: Getting sick after seeing a doctor? diagnosing and mitigating knowledge conflicts in event temporal reasoning Year: (2024-06)
Ref_id:b13 Title: URL Year: ()
Ref_id:b14 Title: Retrieval-augmented generation for large language models: A survey Year: (2023)
Ref_id:b15 Title: Transformer feed-forward layers are key-value memories Year: (2021-11)
Ref_id:b16 Title: URL Year: ()
Ref_id:b17 Title: Understanding finetuning for factual knowledge extraction Year: (2024)
Ref_id:b18 Title: Context-parametric inversion: Why instruction finetuning may not actually improve context reliance Year: ()
Ref_id:b19 Title: Accelerating the science of language models Year: (2024)
Ref_id:b20 Title: A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions Year: (2023)
Ref_id:b21 Title: Phi-2: The surprising power of small language models Year: (2023)
Ref_id:b22 Title: Enhancing robustness in large language models: Prompting for mitigating the impact of irrelevant information Year: (2024)
Ref_id:b23 Title: Do llms dream of elephants (when told not to)? latent concept association and associative memory in transformers Year: (2024)
Ref_id:b24 Title: Massive values in self-attention modules are the key to contextual knowledge understanding Year: (2025)
Ref_id:b25 Title: Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models Year: (2024)
Ref_id:b26 Title: Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models Year: (2024)
Ref_id:b27 Title: Natural questions: a benchmark for question answering research Year: (2019)
Ref_id:b28 Title: Contradoc: Understanding self-contradictions in documents with large language models Year: (2023)
Ref_id:b29 Title: Contrastive decoding: Open-ended text generation as optimization Year: (2023)
Ref_id:b30 Title: How do transformers learn topic structure: Towards a mechanistic understanding Year: (2023)
Ref_id:b31 Title: The memotrap dataset Year: (2023)
Ref_id:b32 Title: Entity-based knowledge conflicts in question answering Year: (2021)
Ref_id:b33 Title: Interpreting key mechanisms of factual recall in transformer-based language models Year: (2024)
Ref_id:b34 Title: One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention Year: ()
Ref_id:b35 Title: Memorization capacity of multi-head attention in transformers Year: ()
Ref_id:b36 Title: Copy suppression: Comprehensively understanding an attention head Year: (2023)
Ref_id:b37 Title: Locating and editing factual associations in gpt Year: (2022)
Ref_id:b38 Title: Mass-editing memory in a transformer Year: (2022)
Ref_id:b39 Title: Fact finding: Attempting to reverse-engineer factual recall on the neuron level Year: (2023)
Ref_id:b40 Title: Understanding factual recall in transformers via associative memories Year: (2024)
Ref_id:b41 Title: -context learning and induction heads Year: (2022)
Ref_id:b42 Title: merge conflicts!" exploring the impacts of external distractors to parametric knowledge graphs Year: (2023)
Ref_id:b43 Title: Tool learning with large language models: A survey Year: (2025)
Ref_id:b44 Title: A mechanistic explanatory strategy for xai Year: (2024)
Ref_id:b45 Title: How much knowledge can you pack into the parameters of a language model Year: (2020-11)
Ref_id:b46 Title: Mitigating knowledge conflicts in llm generation via identifying and reweighting context-aware neurons Year: ()
Ref_id:b47 Title: Large language models can be easily distracted by irrelevant context Year: (2023)
Ref_id:b48 Title: Trusting your evidence: Hallucinate less with context-aware decoding Year: (2024)
Ref_id:b49 Title: Blinded by generated contexts: How language models merge generated and retrieved contexts when knowledge conflicts Year: (2024-08)
Ref_id:b50 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b51 Title: Llama 2: Open foundation and finetuned chat models Year: (2023)
Ref_id:b52 Title: Interpretability in the wild: a circuit for indirect object identification in gpt-2 small Year: ()
Ref_id:b53 Title: Knowledge editing for large language models: A survey Year: (2024)
Ref_id:b54 Title: Resolving knowledge conflicts in large language models Year: ()
Ref_id:b55 Title: How easily do irrelevant inputs skew the responses of large language models? Year: ()
Ref_id:b56 Title: The rise and potential of large language model based agents: A survey Year: (2025)
Ref_id:b57 Title: Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts Year: (2024)
Ref_id:b58 Title: Knowledge conflicts for llms: A survey Year: (2024)
Ref_id:b59 Title: Intuitive or dependent? investigating llms' behavior style to conflicting prompts Year: (2024)
Ref_id:b60 Title: Making retrieval-augmented language models robust to irrelevant context Year: ()
Ref_id:b61 Title: Characterizing mechanisms for factual recall in language models Year: (2023-12)
Ref_id:b62 Title: Discerning and resolving knowledge conflicts through adaptive decoding with contextual informationentropy constraint Year: (2024-08)
Ref_id:b63 Title: Mitigating temporal misalignment by discarding outdated facts Year: (2023-12)
Ref_id:b64 Title: URL Year: ()
Ref_id:b65 Title: Trained transformers learn linear models in-context Year: (2024)
Ref_id:b66 Title: Context-faithful prompting for large language models Year: (2023-12)
Ref_id:b67 Title: URL Year: ()
