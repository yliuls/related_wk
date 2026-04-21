Title: Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning
Abstract: Decision-making models for individuals, particularly in high-stakes scenarios like vaccine uptake, often diverge from population optimal predictions. This gap arises from the uniqueness of the individual decision-making process, shaped by numerical attributes (e.g., cost, time) and linguistic influences (e.g., personal preferences and constraints). Developing upon Utility Theory and leveraging the textualreasoning capabilities of Large Language Models (LLMs), this paper proposes an Adaptive Textual-symbolic Human-centric Reasoning framework (ATHENA) to address the optimal information integration. ATHENA uniquely integrates two stages: First, it discovers robust, group-level symbolic utility functions via LLMaugmented symbolic discovery; Second, it implements individual-level semantic adaptation, creating personalized semantic templates guided by the optimal utility to model personalized choices. Validated on real-world travel mode and vaccine choice tasks, ATHENA consistently outperforms utility-based, machine learning, and other LLM-based models, lifting F1 score by at least 6.5% over the strongest cutting-edge models. Further, ablation studies confirm that both stages of ATHENA are critical and complementary, as removing either clearly degrades overall predictive performance. By organically integrating symbolic utility modeling and semantic adaptation, ATHENA provides a new scheme for modeling human-centric decisions. The project page can be found at https://yibozh.github.io/Athena.

Section: Introduction
Consider the widely debated vaccine dilemma [1], from a population-level perspective aimed at optimizing collective well-being (e.g., achieving herd immunity at minimal societal cost), models would invariably predict near-universal vaccine adoption. However, this population optimum consistently fails to predict actual individual behavior. The reality is a broad spectrum of personal choices, because each individual undertakes their own unique cognitive calculus: balancing a vaccine's perceived efficacy and protection against its perceived risks [2], all filtered through their personal beliefs, constraints, and even considerations of potential "free-riding" on the immunity of others [3]. This divergence between the theoretical population optimum and observed individual actions highlights a critical limitation: models designed to maximize collective outcomes do not adequately explain or predict individuals' choices. Instead, individual choices are profoundly shaped by who we are, when the decision is made, and the unique constraints we face. This internal 'cognitive calculus', unique to each individual and situation, presents a profound challenge for human decision modeling.
For decades, researchers have attempted to model this 'cognitive calculus' with Utility Theory [4][5][6], which assumes individuals select options that maximize expected gain. Operationally, this involves defining a parametric utility function, denoted as f : X → R, that maps a vector of structured attributes X (e.g., monetary cost and time) for each option to a scalar utility score. Such pre-defined and explicit specifications of f are the basis for classic discrete choice models [7], which have been widely adopted across economics [8][9][10], transportation [7,[11][12][13], and public policy [14][15][16]. In these models, the utility scores derived from f for each available option are used to probabilistically determine the likelihood of an individual selecting a particular option. However, even these utilitybased models encounter fundamental barriers when attempting to capture the full depth of human decision-making. Real-world human decisions, as seen in the vaccine dilemma, frequently deviate from these mathematical formulations. Individuals exhibit behaviors that appear inconsistent or irrational [17,18], yet these are often driven by subjective feelings and personal experiences. Such deviations reflect that traditional models, with their reliance on pre-specified functions, struggle to capture personalized decisions [19]. The clue for deciphering this deviation is covered within individual attributes, some of which are structured and quantifiable, while others are unstructured and semantic (e.g., individual preference and constraints).
Addressing these unstructured and semantic dimensions, which are pivotal for capturing personalized decision mechanisms, calls for new modeling paradigms. LLMs with their strong textual-reasoning capability offer a clear advance [20], providing new mechanisms for identifying the utility function f and for integrating semantic individual context directly into the decision modeling process. Specifically, LLMs enhance our ability to model human choice by: a) Guiding the discovery of more accurate and robust parametric forms for f . Through LLM-augmented symbolic regression [21][22][23], it becomes feasible to identify data-driven mathematical structures that capture underlying group-level choice patterns more effectively than pre-specified forms. b) Enabling the infusion of individual-level textual information into the human decision modeling [24][25][26]. By encoding personal preferences, constraints, and narratives, LLMs allow each decision to reflect the nuanced motivations and situational factors that traditional numeric features alone cannot convey. This paper introduces an Adaptive Textual-symbolic Human-centric Reasoning framework (ATHENA). ATHENA achieves personalized decision modeling by uniquely integrating two sequentially structured steps: First, at the group level, it focuses on discovering robust, symbolic utility functions. Second, it implements individual-level, LLM-powered semantic adaptation guided by optimal utility functions discovered in previous steps. The outcome is a customized semantic template for each person, specifically designed to empower an LLM to model their choices by incorporating their unique preferences and constraints. We empirically validate ATHENA on two real-world human decision-making tasks: travel mode choice and vaccine uptake decisions. The model consistently outperforms traditional utility-based, machine learning, and LLM-based models, with at least 6.5% improvement in F1 score. Further ablation experiments reveal that removing either the group-level symbolic utility search or the individual semantic adapter lowers performance by at least 18%, underscoring the merit of the full ATHENA framework.
this section cite: ['b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b6', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25']

Section: Related Work
Utility-based Decision-Making Models. Initial explorations into this complex domain were predominantly by utility-based models [27][28][29][30]. These methods aim to capture human behavior within explicit mathematical functions, formulated from empirical data. Established methodologies such as Discrete Choice Models (DCMs) have been widely used, attributable to their interpretability and robust statistical underpinnings [31,10,[32][33][34][35][36][37]. While offering tractability, they may also limit the ability to fully capture complex non-linear patterns and diverse preferences in modern high-dimensional data.
Machine Learning-Driven Decision-Making Models. ML-driven decision-making models aim to directly learn from rich, diverse features. Tree-based ensemble methods, including Random Forests [38], Gradient-Boosting Trees [39], XGBoost [40,41], and LightGBM [42], alongside neural network architectures [43][44][45], exhibited a notable proficiency in fitting complex, non-additive interaction effects. These models effectively integrated large-scale data, but their decision-making processes often lacked transparency despite strong predictive performance. Efforts to enhance transparency via post-hoc explanation frameworks, for instance, SHAP [46][47][48] and Integrated Gradients [49,50], have provided some insights for human behavior. A persistent challenge is these models' vulnerability to distribution shifts, lack of transparency, and limited ability to provide symbolic, interpretable insights needed for personalized utility reasoning.
this section cite: ['b26', 'b27', 'b28', 'b29', 'b30', 'b9', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49']

Section: Symbolic Regression with LLMs.
Classical symbolic regression (SR) typically uses genetic programming to evolve populations of candidate equations via stochastic mutation and crossover [51][52][53]. The goal is to find formulas that balance simplicity, generalizability, and human interpretability [54][55][56]. The recent rise of LLMs has revitalized symbolic regression, enabling new possibilities in scientific discovery when combined with advanced evolutionary algorithms [57]. For example, LLM-SR integrates LLM with evolutionary symbolic regression by treating equations as executable programs. It leverages LLMs' scientific prior knowledge and code generation abilities to iteratively generate, refine, and optimize equation skeletons [22]. LASR integrated LLM-driven abstract textual concepts within evolutionary frameworks, achieving notable performance enhancements on benchmarks, such as the Feynman equation set [58]. The DiSciPLE framework extended these contributions by emphasizing the interpretability and reliability of generated scientific hypotheses, incorporating critical evaluation and simplification to ensure hypotheses are both scientifically rigorous and computationally efficient [59].
this section cite: ['b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b21', 'b57', 'b58']

Section: LLM-based Decision-Making Models.
The advent of LLMs has offered a new opportunity, establishing these models as human-like reasoning engines [60][61][62][63][64][65]. Techniques such as instruction tuning [66][67][68][69], chain-of-thought reasoning [70][71][72][73][74] are elevating LLMs move beyond basic text generation to handle more complex tasks involving step-by-step reasoning and symbolic or numerical problem-solving [75][76][77][78][79][80][81]. Within the specific domain of personalized decision making, preliminary findings suggest that zero-shot and few-shot prompting strategies can enhance the behavioral alignment of LLMs [82,83]. Because a model's knowledge is inherited from generic pre-training priors, its reasoning defaults to universally salient factors -e.g., cost and time in travel mode choice -while overlooking personal preferences such as rail-pass loyalty or transfer aversion, thereby introducing systematic bias [84]. Techniques like persona loading partially mitigate this gap by conditioning responses on inferred preference structures [82,85]. Beyond basic prompting, decision-centric systems add explicit structure to improve reliability and transparency: DeLLMa enumerates plausible states, elicits utilities via pairwise comparisons, and maximizes expected utility; STRUX distills inputs into fact tables with self-reflective evidence; OptiGuide compiles natural-language "what-if" queries into optimization code and invokes solvers; Agent-Driver coordinates tool calls, commonsense/experience memory, and chain-of-thought planning; Personalized Oncology evaluations show chat-LLMs still trail experts, motivating structured, evidence-grounded pipelines [86][87][88][89][90]. Nevertheless, many deployments still treat LLMs as opaque scoring mechanisms, falling short of fully recovering explicit, personalized utility logic.
this section cite: ['b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80', 'b81', 'b82', 'b83', 'b81', 'b84', 'b85', 'b86', 'b87', 'b88', 'b89']

Section: Methods
We consider a classic discrete choice problem, where an individual i faces a finite set of choices J = {1, 2, . . . , J}, where J = |J |. The decision-making process assumes individuals select the option j that maximizes their utility. Each individual's observed choice behavior is represented by a dataset D = {(X i , y i )} N i=1 . For each observation i, X i = {x ij } J j=1 represents the set of feature vectors, where x ij ∈ R K captures the features for choice j and y i ∈ J denotes the observed choice.
The standard approach to modeling discrete choices is the Random Utility Maximization (RUM) framework [91]. It assumes the latent utility for each alternative j is described by U ij = f (X i , j)+ϵ ij , where f (X i , j) is systematic component of utility and ϵ ij is the random error. Assuming ϵ ij are independently and identically drawn (i.i.d.) and follow a Type I Extreme Value distribution [92], the probability of individual i choosing alternative j is given by: i is used to predict personal decisions.
P (y i = j | X i ) = e f (Xi,j;θ) k∈J e f (Xi,k;θ) .(1)
A central challenge lies in specifying the systematic utility component f . Traditional applications of RUM often rely on pre-specified functional forms for f using domain expertise and observed data [93]. This approach may result in a suboptimal representation of the true decision mechanism, while also neglecting individual heterogeneity in choices. Furthermore, traditional decision-making models are not designed to incorporate non-structured semantic information. To address these limitations, we introduce ATHENA for personalized decision modeling designed to identify suitable utility function representations while simultaneously capturing individual-specific preferences. As shown in Fig. 1, ATHENA structures the decision modeling process into two sequential stages:
1. Group-Level Symbolic Utility Discovery: This initial stage focuses on identifying optimal symbolic utility components that capture common decision patterns within distinct demographic groups. The discovery is achieved through a feedback-informed symbolic discovery process powered by LLMs.
2. Individual-Level Semantic Adaptation: Then, the optimal group-level utility functions serve as guidance for the LLM-driven optimization of personalized semantic templates. This adaptation process is designed to incorporate individual-specific preferences and constraints, leveraging the rich semantic reasoning capabilities of LLMs.
this section cite: ['b90', 'b91', 'b92']

Section: Group-Level Symbolic Utility Discovery
The first stage aims to discover an optimal parametric utility function, denoted as f * g , for each demographic group g ∈ G. This function f * g should be constructible from symbolic building blocks and optimally explain the group's choice behavior. Following the choice probability defined earlier Eq. ( 1), the objective is to find the optimal f * g and its associated parameters θ * g such that:
(θ * g , f * g ) = arg min f,θ L g (f (X i , y i ; θ); D g ) = - (Xi,yi)∈Dg log e f (Xi,yi;θ) k∈J e f (Xi,k;θ) .(2)
To automate the symbolic utility discovery of f * g , we design an iterative, feedback-informed generation process powered by LLMs. To effectively guide the automated discovery of utility functions, we constructed two foundational libraries: a domain knowledge concept library (C) and a symbolic library (S). The library C, developed based on input from domain experts, covers high-level conceptual knowledge about domain-specific human behavior. The library S provides the fundamental syntactic building blocks needed for constructing all candidate utility expressions.
Inspired by evolutionary algorithms [94], the core discovery process proceeds iteratively. In each iteration t for each demographic group g, the LLM samples a set of K candidate symbolic utility functions, {f t g,k } K k=1 . This sampling is performed from the LLM's learned distribution ϕ [20,95], conditioned on the group profile g, the domain concept C, the available symbolic building block S, and a feedback B t-1 from preceding iteration:
{f t g,k } K k=1 ∼ ϕ(•|g, C, S, B t-1 )(3)
The feedback B is essential in refining the LLM's sampling strategy. Specifically, B t is constructed at the end of each iteration t and comprises the best-performing and worst-performing candidate functions from that iteration:
B t = {f t g,+ , f t g,-},(4)
where
f t g,+ = arg min k∈K L g (f t g,k , D g ) and f t g,-= arg max k∈K L g (f t g,k , D g ).
Here L g is the group-level loss function, with a similar format as Eq. ( 2). This feedback B t is used to refine the LLM's sampling distribution ϕ through stochastic mutation or crossover [51][52][53], pushing the generation towards more promising types of functions. The iterative discovery process for group g is considered to have converged at iteration T if the absolute difference in the loss of the best-performing function from the current iteration t and that of the previous iteration t -1 falls below a predefined threshold δ:
|L g (f t g,+ , D g ) -L g (f t-1 g,+ , D g )|< δ.(5)
Upon convergence at iteration at T , the optimal group-level symbolic utility function (f * g ) is determined as the function that achieved the minimum loss across all generated candidate functions throughout the entire iterative process:
f * g = arg min f ∈Fg L g (f, D g ),(6)
where F g = T t=1 {f t g,k } K k=1 and T is the iteration at which convergence occurred. This discovered function f * g , along with its fitted parameters θ * g , serves as the learned representation of the systematic utility for group g.
this section cite: ['b93', 'b19', 'b94', 'b50', 'b51', 'b52']

Section: Individual-Level Semantic Adaptation
Following the determination of the group-level optimal symbolic utility functions f * g , the framework transitions to the second stage, leveraging an LLM conditioned on f * g to model individual choice behavior more accurately. While f * g captures the central tendencies of utility for group g, significant intra-group heterogeneity often persists. To account for this, we introduce an individual-level adaptation stage to personalize the utility representation by generating and refining an individualspecific semantic template.
For each individual i ∈ g, the initial semantic template, denoted as P 0 i is generated by the LLM (ϕ). The generation of the initial semantic template is represented as a sampling process from the LLM's distribution:
P 0 i ∼ ϕ(•|f * g , i, C).
In this formulation, ϕ conditions on the optimal group-level symbolic function f * g , the specific individual context i, and the high-level domain concepts from C to generate P 0 i . This initial template Pi 0 is a semantic representation that is designed to be adaptable in subsequent optimization steps. Then, the semantic template P 0 i undergoes an iterative refinement process for each individual i. This optimization is driven by TextGrad [96], which optimizes the template based on the individual's specific data D i = (X i , y i ). The update rule is given by:
P t+1 i ← P t i -η∇L i (P t i , D i ).(7)
The term ∇L i (P t i , D i ) represents the "textual gradient" of the loss function with respect to the semantic template P t i . Since P t i is the textual template, this gradient is not a vector of partial derivatives in the mathematical sense. Instead, it indicates the direction and nature of textual modifications to P t i that would lead to the most improvement in loss. This iterative refinement process continues until a maximum number of iterations T ′ is reached. Then the final optimal semantic template for individual i, denoted as P * i , is determined. The predicted personalized choice ŷi is then represented as sampling from the LLM's output distribution:
ŷi ∼ ϕ( P * i , X i Semantic Adaptation | Symbolic Utility Discovery f * g (X i ; θ * g ) ) (8
)
The overall procedure of ATHENA is summarized in Algorithm 1.
this section cite: ['b95']

Section: Experiments Algorithm 1 ATHENA Optimization Flow
Require: Demographic group g, dataset Dg, domain concept C, symbolic building block S 1: Initialize B0 ← None // Stage 1: Group-Level Symbolic Utility Discovery 2: for t = 1 to T do 3:
this section cite: []

Section: Sample symbolic utility functions {f
t g,k } K k=1 ∼ ϕ(• | g, C, S, B t-
1 ) 4: Update B t ← {f t g,+ , f t g,-} using Eq. (4) 5: Select best function f * g ← arg min f ∈Fg Lg(f, Dg) 6: if stopping condition in Eq. (5) is met then 7: break 8: end if 9: end for // Stage 2: Individual-Level Semantic Adaptation 10: for each individual i ∈ g do 11: Initialize semantic template P 0 i ∼ ϕ(• | f * g , i, C) 12: for t = 1 to T ′ do 13: Update P t+1 i ← P t i -η∇Li(P t i , Di) using Eq. (7) 14: end for 15: end for 16: return {P * i }i∈g, predict decisions using Eq. (8).
This section empirically validates the value of ATHENA, demonstrating its overall effectiveness in personalized decision-making and its robust capability to apply across diverse application domains. We break down our experimental findings to specifically showcase the distinct value added by each core component of the ATHENA framework: 1) group-level symbolic utility discovery and 2) personalized semantic template adaptation. Fig. 2 illustrates the full pipeline using the travel-mode choice as an example.
this section cite: []

Section: Experimental Setup
Datasets. To test ATHENA's ability to generalize across different domains and to adapt to individual preferences, we selected two real-world tasks that reflect fundamentally different personalized decision scenarios: daily transportation choices and public health decisions. (1) Swissmetro Transportation Choice (Swissmetro): is a widely used benchmark in travel mode choice modeling [97][98][99][100][101]. Each record details a trip between major Swiss cities and includes both traveler characteristics (e.g., income, age) and alternative-specific attributes (e.g., travel time, cost). The dataset has a potential choice set of three transportation modes: Train, Car, and Metro. (2) COVID-19 Vaccination Choice (Vaccine): This dataset is derived from a large-scale international survey, conducted across multiple countries [102]. The survey was designed to understand factors influencing COVID-19 vaccine uptake and attitudes. For each participant, it captures demographics, prior beliefs about the vaccine, and their self-reported vaccination status. The modeled choices based on this information include: Unvaccinated, Vaccinated initial doses, no booster, and Vaccinated initial doses plus booster.
this section cite: ['b96', 'b97', 'b98', 'b99', 'b100', 'b101']

Section: Experiment Configurations.
To maintain a reasonable budget for the template-adaptation stage, we restricted the experimental sample to a representative subset of each dataset. Specifically, we used:
(1) Swissmetro: 500 travelers, two trip records per person; (2) Vaccine: 300 respondents, one survey record per person. Within each dataset, we first identified key demographic dimensions (gender, age, and income), then sampled approximately balanced subsets across these strata from the full dataset. This ensures (i) comparable class priors between training and test splits, and (ii) that no demographic group dominates the symbolic-utility discovery process. The predefined demographic grouping follows established practice in choice modeling, supports interpretability, and improves robustness by avoiding the complexity and data requirements of latent clustering methods [103][104][105].
Evaluation metrics. We report Accuracy, F1, AUC, and Cross-Entropy (CE). CE is included because ATHENA produces probabilistic predictions over choices. A lower CE means the model assigns higher probabilities to actual choices, while F1 and AUC capture classification performance; together, they provide complementary views on accuracy and calibration.
this section cite: ['b102', 'b103', 'b104']

Section: Models and baselines.
Both stages of ATHENA, symbolic-utility discovery and individual semantic adaptation, run on the gpt-4o-mini-2024-07-18 and gemini-2.0-flash. To evaluate its performance, we contrasted ATHENA with three baseline groups. (i) LLM-based methods: a plain zero-shot method [106,107], a zero-shot chain-of-thought method [106], a five-example few-shot method [108,109], and TextGrad tuning [96]. (ii) Classical discrete-choice models: Multinomial Logit (MNL) [110], Conditional Logit (CLogit) [111], and Latent-Class MNL [112]. (iii) Standard machine-learning classifiers: logistic regression, random forest, XGBoost [113], a shallow two-layer MLP [114], TabNet for tabular data [115], and a fine-tuned BERT classifier [116]. This spectrum ranges from end-to-end language-model reasoning through discrete choice models to conventional predictive learners, providing a balanced reference for unique modeling capabilities.
this section cite: ['b105', 'b106', 'b105', 'b107', 'b108', 'b95', 'b109', 'b110', 'b111', 'b112', 'b113', 'b114', 'b115']

Section: Overall Performance Analysis
Performance and insights. As shown in Table 1, on the Swissmetro mode choice task, ATHENA with GPT-4o-mini notably outperforms evaluated baselines across Accuracy (Acc), F1-score (F1), and AUC. Over the strongest baseline, it achieves gains of at least 6% in Acc and 6.5% in F1, respectively. Similar improvements are noted on the Vaccine dataset. Notably, our proposed method exhibits higher Cross-Entropy (CE) compared to baselines such as XGBoost. We attribute this to the inherent design of ATHENA, which produces more conservative probability distributions rather than extreme certainties. Specifically, unlike models that might predict a choice with > 90% confidence, ATHENA's framework is less prone to such high probabilities. This characteristic may better reflect the uncertain nature of human decision-making, which our model is designed to accommodate. Overall, the performance enhancements highlight ATHENA's strength in combining symbolic structures with semantic adaptation for effective personalized decision modeling.
Disentangling Semantically Similar Choices. Prompt-only LLMs and classical choice models frequently fail to distinguish between superficially similar options. For example, the few-shot LLM misclassified 75% of true Car trips as the premium Metro service. By introducing symbolic-level structure and performing individual-level semantic adaptation, ATHENA more than doubled the number of correctly classified Car trips, while maintaining high recall for both Train and Metro.
On the Vaccine task, its learned templates encode key interactions such as age-risk trade-offs and prior-infection hesitancy, allowing it to achieve the highest F1 score despite strong semantic similarity between fully vaccinated and booster options. In practice, these interpretable templates enable a better understanding of individual behavior, for instance, identifying who tends to decline vaccination and why, which is crucial for informing high-stakes decision-making. See Appendix A.3 for details.
this section cite: []

Section: Computational Complexity and Scalability.
With T and T ′ fixed, ATHENA's runtime is linear in the number of groups |G| and individuals N , scaled by the average LLM latency τ :
O((|G|KT + N T ′ ) τ tok ).
Both stages parallelize naturally, as group-level searches run independently and individual-level refinements can be batched or distributed. Detailed runtime measurements are provided in Appendix D.
Extended backbone LLM comparisons. On a 100-individual subset, we also tested larger reasoning LLMs (Qwen3-32B, DeepSeek-R1-Distill-Qwen-32B, GPT-4o). With prompt-only baselines, larger reasoning models occasionally yield higher F1/Acc but exhibit volatile calibration (high CE), reflecting the lack of structural constraints. Under ATHENA, backbone differences shrink: the symbolic discovery plus semantic adaptation turns the task into constrained sampling and small, directed improvements, allowing lightweight models to reach near-maximal performance, while stronger reasoning models provide modest, consistent gains on harder interactions (e.g., vaccine risk-trust trade-offs). Full experimental details appear in Appendix C for completeness.
this section cite: []

Section: Ablation Study
We evaluate ATHENA's two components by (i) keeping only the group-level symbolic utility discovery and (ii) keeping only the individual-level semantic adaptation, under identical data and metrics. We do not include a symbolic-only group-level discovery baseline (Stage 1 without LLM), because the Concept Library is accessible only via the LLM. Excluding it would reduce the hypothesis space to symbolic operators alone, changing the problem definition rather than providing a clean ablation.
Group-Level Symbolic Utility Discovery: necessary but not sufficient. When ATHENA retains only the group-level symbolic component, accuracy exceeds the classical MNL by 4.7% on Swissmetro and 19% on Vaccine (Table 2), indicating that only LLM-generated utility expressions can already encode broad demographic regularities. The accuracy trajectories of this symbolic discovery process over 30 iterations (Fig. 3) further demonstrate its effectiveness, illustrating the gradual learning of these group-level trends. Nevertheless, lower F1 score and AUC and elevated cross-entropy, reflecting limited discriminative capacity for similar alternatives. These results highlight the symbolic stage's strength in pruning the hypothesis space to interpretable structures, but also expose its limitations in capturing much heterogeneity.  Individual-Level Semantic Adaptation: powerful only with a solid starting point. Conversely, bypassing symbolic discovery and initiating TextGrad from random templates leads to noteworthy degraded performance: As shown in Table 2, accuracy drops to 60.4% on the Swissmetro dataset and 54.3% on the Vaccine dataset; Swissmetro's CE more than doubles (2.29), and AUC falls below 0.70. Without a sound starting point, gradients are likely to converge to local optima and yield erratic probability outputs, reaffirming the unreliability of unguided adaptation in multi-choice settings.
Take-away. The two stages of ATHENA are complementary: symbolic discovery supplies an interpretable, well-regularized search space, while semantic adaptation injects the individual-level nuance that symbolic rules alone miss.
this section cite: []

Section: Symbolic Utility Discovery Fragment Analysis
Equation (8) shows that an individual prediction is influenced by group-level symbolic utility f * g (X i ; θ * g ). In this section, we demonstrate the building blocks of those utilities are both behaviorally meaningful and reusable across groups. As shown in Fig. 4, each symbolic utility is decomposed into atomic fragments {φ 1 , φ 2 , . . . } and their global importance is quantified.
Fragment score. For every group g we retain the top-K (K = 3) utilities ranked by held-out accuracy Acc(f ). The importance score of a fragment φ m is then
Score (φ m ) = g∈G K k=1 ⊮ φ m ⊂ f * g,k • Acc f * g,k (9)
So a fragment earns points whenever it (i) appears in the top-ranked utilities of many groups and (ii) is embedded in highly predictive expressions.
Fig. 4 visualizes the fragment scores for both datasets. Only a small fraction of fragments dominate, confirming that ATHENA converges to a compact and interpretable symbolic basis. For example, in Vaccine, one of the leading fragment φ 7 = √ Age * (Trust Government + Trust_Science) softens age's impact at higher values while amplifying it for individuals who trust government or science, precisely isolating the cohort most likely to take boosters.
Beyond fragment-level analysis, ATHENA also produces fully interpretable symbolic utilities. Representative full formulas and domain-relevant insights for both Swissmetro and Vaccine are provided in Appendix A.4.
this section cite: ['b7']

Section: Conclusion
This research highlights the critical role of textual-semantic information in overcoming the limitations of traditional utility-based models for human decision-making. By introducing ATHENA, an adaptive textual-symbolic and human-centric reasoning framework is proposed that integrates group-level symbolic regression of utility functions with individual-level, LLM-powered semantic modeling, we offer a more comprehensive and personalized view of choice behavior. Our experiments on transportation mode choice and vaccine uptake demonstrate that this co-design approach clearly outperforms three existing model zoos, including classical utility, machine learning, and purely LLM-based approach, underscoring the benefits of capturing both structured attributes and rich semantic context. These findings suggest that textualized-symbolic reasoning can bridge the gap between theoretical utility optimization and real-world individual choices, paving the way for more adaptive and human-centric decision models.
Limitations. The current implementation of ATHENA has two limitations. 1) Computational Complexity: The proposed framework requires extra computational resources for textual gradient, particularly when scaling to larger populations. 2) Representation on Groups: ATHENA assumes that a shared symbolic utility function can effectively model each demographic group. However, groups with greater internal diversity may produce weaker or less reliable representations. 3) Result Stability: All reported results are based on single representative runs under fixed random seeds, given the computational cost of multi-stage adaptation. Future work will include multi-seed repetitions to further examine the stability of ATHENA's performance.
this section cite: []

Section: References
Ref_id:b0 Title: Imitation dynamics of vaccination behaviour on social networks Year: (1702)
Ref_id:b1 Title: Making sense of perceptions of risk of diseases and vaccinations: a qualitative study combining models of health beliefs, decision-making and risk perception Year: (2011)
Ref_id:b2 Title: On the benefits of explaining herd immunity in vaccine advocacy Year: (2017)
Ref_id:b3 Title: Theory of games and economic behavior: 60th anniversary commemorative edition Year: (2007)
Ref_id:b4 Title: Risk aversion and expected-utility theory: A calibration theorem Year: (2013)
Ref_id:b5 Title: Generalized expected utility theory: The rank-dependent model Year: (2012)
Ref_id:b6 Title: Discrete choice analysis: theory and application to travel demand Year: (1985)
Ref_id:b7 Title: Applied welfare economics with discrete choice models Year: (1981)
Ref_id:b8 Title: The landscape of econometric discrete choice modelling research Year: (2021)
Ref_id:b9 Title: A comparative empirical study of discrete choice models in retail operations Year: (2022)
Ref_id:b10 Title: Equality of opportunity in travel behavior prediction with deep neural networks and discrete choice models Year: (2021)
Ref_id:b11 Title: A systematic comparative evaluation of machine learning classifiers and discrete choice models for travel mode choice in the presence of response heterogeneity Year: (2022)
Ref_id:b12 Title: Departure time choice models in urban transportation systems based on mean field games Year: (2022)
Ref_id:b13 Title: Game-theoretic modeling of pre-disaster relocation Year: (2020)
Ref_id:b14 Title: From government to market? a discrete choice analysis of policy instruments for electric vehicle adoption Year: (2022)
Ref_id:b15 Title: Public preferences and willingness to pay for a net zero nhs: a protocol for a discrete choice experiment in england and scotland Year: (2024)
Ref_id:b16 Title: Modelling dataset bias in machine-learned theories of economic decision-making Year: (2024)
Ref_id:b17 Title: Efficiently irrational: deciphering the riddle of human choice Year: (2022)
Ref_id:b18 Title: The expected utility model: Its variants, purposes, evidence and limitations Year: (1982)
Ref_id:b19 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b20 Title: Mathematical discoveries from program search with large language models Year: (2024)
Ref_id:b21 Title: Llm-sr: Scientific equation discovery via programming with large language models Year: ()
Ref_id:b22 Title: Symbolic regression with a learned concept library Year: (2024)
Ref_id:b23 Title: Using large language models to simulate multiple humans and replicate human subject studies Year: (2023)
Ref_id:b24 Title: Large pre-trained language models contain human-like biases of what is right and wrong to do Year: (2022)
Ref_id:b25 Title: Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in chatgpt Year: (2023)
Ref_id:b26 Title: A state-of-the-art review: Developments in utility theory, prospect theory and regret theory to investigate travellers' behaviour in situations involving travel time uncertainty Year: (2014)
Ref_id:b27 Title: Linking mode choice with travel behavior by using logit model based on utility function Year: (2021)
Ref_id:b28 Title: Travel mode choice and travel satisfaction: bridging the gap between decision utility and experienced utility Year: (2016)
Ref_id:b29 Title: Travel Mode Use, Travel Mode Shift and Subjective Well-Being: Overview of Theories, Empirical Findings and Policy Implications Year: (2016)
Ref_id:b30 Title: Hybrid choice models: The identification problem Year: (2024)
Ref_id:b31 Title: Discrete choice experiments: a guide to model specification, estimation and software Year: (2017)
Ref_id:b32 Title: Rational inattention: The interplay of stakes and prior beliefs in a laboratory study Year: (2025)
Ref_id:b33 Title: Four types of typical discrete choice models: Which are you using? Year: (2012)
Ref_id:b34 Title: Rational inattention in discrete choice models: Estimable specifications of ri-multinomial logit (ri-mnl) and ri-nested logit (ri-nl) models Year: (2023)
Ref_id:b35 Title: Transportation mode choice behavior with multinomial logit model: work and school trips Year: (2024)
Ref_id:b36 Title: Graph neural networks for intelligent transportation systems: A survey Year: (2023)
Ref_id:b37 Title: Understanding travel mode choice behavior: Influencing factors analysis and prediction with machine learning method Year: (2023)
Ref_id:b38 Title: Assessing the performance of gradientboosting models for predicting the travel mode choice using household survey data Year: (2022)
Ref_id:b39 Title: Analysis of travel mode choice in seoul using an interpretable machine learning approach Year: (2021)
Ref_id:b40 Title: A prediction and behavioural analysis of machine learning methods for modelling travel mode choice Year: (2023)
Ref_id:b41 Title: Understanding travel mode choice behavior: Influencing factors analysis and prediction with machine learning method Year: (2023)
Ref_id:b42 Title: Predicting travel mode choice with a robust neural network and shapley additive explanations analysis Year: (2024)
Ref_id:b43 Title: Explaining deep learning-based activity schedule models using shapley additive explanations Year: (2025)
Ref_id:b44 Title: Than: Multimodal transportation recommendation with heterogeneous graph attention networks Year: (2023)
Ref_id:b45 Title: Understanding travel behavior: A deep neural network and shap approach to mode choice determinants Year: ()
Ref_id:b46 Title: A review of explainable artificial intelligence in healthcare Year: (2024)
Ref_id:b47 Title: Explainable ai: A review of machine learning interpretability methods Year: ()
Ref_id:b48 Title: Visualizing deep networks by optimizing with integrated gradients Year: (2019)
Ref_id:b49 Title: Integrated decision gradients: Compute your attributions where the model makes its decision Year: ()
Ref_id:b50 Title: Symbolic regression via genetic programming Year: (2000)
Ref_id:b51 Title: Multifactorial genetic programming for symbolic regression problems Year: (2018)
Ref_id:b52 Title: Self-organizing migrating algorithm: review, improvements and comparison Year: (2023)
Ref_id:b53 Title: Interpretable machine learning for science with pysr and symbolicregression.jl Year: ()
Ref_id:b54 Title: Regularized evolution for image classifier architecture search Year: (2019)
Ref_id:b55 Title: Interpretable scientific discovery with symbolic regression: a review Year: (2024-01)
Ref_id:b56 Title: Reevo: Large language models as hyper-heuristics with reflective evolution Year: (2024)
Ref_id:b57 Title: Symbolic regression with a learned concept library Year: (2024)
Ref_id:b58 Title: Disciple: Learning interpretable programs for scientific visual discovery Year: (2025)
Ref_id:b59 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b60 Title:  Year: (2024)
Ref_id:b61 Title: Mistral 7b Year: (2023)
Ref_id:b62 Title:  Year: (2024)
Ref_id:b63 Title:  Year: ()
Ref_id:b64 Title:  Year: (2024)
Ref_id:b65 Title: Instruction tuning for large language models: A survey Year: (2023)
Ref_id:b66 Title: The flan collection: Designing data and methods for effective instruction tuning Year: (2023)
Ref_id:b67 Title: Instruction tuning with gpt-4 Year: (2023)
Ref_id:b68 Title: Visual instruction tuning Year: (2023)
Ref_id:b69 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b70 Title: Faithful chain-of-thought reasoning Year: ()
Ref_id:b71 Title: Beyond chain-of-thought: A survey of chain-of-x paradigms for llms Year: (2024)
Ref_id:b72 Title: Chain of preference optimization: Improving chain-of-thought reasoning in llms Year: (2024)
Ref_id:b73 Title: Unlocking the capabilities of thought: A reasoning boundary framework to quantify and optimize chain-ofthought Year: (2024)
Ref_id:b74 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b75 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b76 Title: Self-consistency improves chain of thought reasoning in language models Year: (2023)
Ref_id:b77 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b78 Title: Retrieval-augmented generation for large language models: A survey Year: ()
Ref_id:b79 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b80 Title: When do program-of-thought works for reasoning? Year: (2024)
Ref_id:b81 Title: Can large language models capture human travel behavior? evidence and insights on mode choice Year: (2024)
Ref_id:b82 Title: Large language models for travel behavior prediction Year: (2023)
Ref_id:b83 Title: Self-debiasing large language models: Zero-shot recognition and reduction of stereotypes Year: (2024)
Ref_id:b84 Title: Personas improve behavioral alignment of large language models Year: (2024)
Ref_id:b85 Title: DeLLMa: Decision making under uncertainty with large language models Year: (2025)
Ref_id:b86 Title: STRUX: An LLM for decision-making with structured explanations Year: (2025-04)
Ref_id:b87 Title: Large language models for supply chain optimization Year: (2023)
Ref_id:b88 Title: A language agent for autonomous driving Year: (2024)
Ref_id:b89 Title: Leveraging large language models for decision support in personalized oncology Year: (2023)
Ref_id:b90 Title: The structure of random utility models Year: (1977)
Ref_id:b91 Title: Conditional logit analysis of qualitative choice behavior Year: (1972)
Ref_id:b92 Title: Estimation of state-dependent utility functions using survey data Year: (1991)
Ref_id:b93 Title: An overview of evolutionary algorithms for parameter optimization Year: (1993)
Ref_id:b94 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b95 Title: Optimizing generative ai by backpropagating language model feedback Year: (2025)
Ref_id:b96 Title: The acceptance of modal innovation: The case of swissmetro Year: (2001)
Ref_id:b97 Title: Causality and advanced models in trip mode prediction: Interest in choosing swissmetro Year: (2022)
Ref_id:b98 Title: A new breakthrough in travel behavior modeling using deep learning: A high-accuracy prediction method based on a cnn Year: (2025)
Ref_id:b99 Title: Enhanced utility estimation algorithm for discrete choice models in travel demand forecasting Year: (2025)
Ref_id:b100 Title: Incorporating domain knowledge in deep neural networks for discrete choice models Year: (2025)
Ref_id:b101 Title: A survey of covid-19 vaccine acceptance across 23 countries in 2022 Year: (2023)
Ref_id:b102 Title: Gender differences in preferences Year: (2009-06)
Ref_id:b103 Title: Adolescents' risk-taking behavior is driven by tolerance to ambiguity Year: (2012)
Ref_id:b104 Title: Poverty and economic decision making: a review of scarcity theory Year: (2021)
Ref_id:b105 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b106 Title: Large language models for travel behavior prediction Year: (2023)
Ref_id:b107 Title: Language models are few-shot learners Year: (2020)
Ref_id:b108 Title: Can large language models capture human travel behavior? evidence and insights on mode choice Year: (2024-08-26)
Ref_id:b109 Title: A comparative study of mnl and machine learning methods for travel mode choice of medical travel Year: (2024)
Ref_id:b110 Title: Travel time attractiveness in motorcycle dominated cities: An investigation of university students' travel behavior Year: (2021)
Ref_id:b111 Title: Evaluating electric micro-mobility related mode choice stated preferences: A latent class choice approach Year: (2025)
Ref_id:b112 Title: Predicting the travel mode choice with interpretable machine learning techniques: A comparative study Year: (2022)
Ref_id:b113 Title: Deep Learning Year: (2016)
Ref_id:b114 Title: Tabnet: Attentive interpretable tabular learning Year: (2021)
Ref_id:b115 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b116 Title: An international meta-analysis of values of travel time savings Year: (2009)
Ref_id:b117 Title: Meta-analysis of uk values of travel time: An update Year: ()
Ref_id:b118 Title: Factors affecting airport access mode choice for elderly air passengers Year: (2013)
Ref_id:b119 Title: Models of mode choice and mobility tool ownership beyond 2008 fuel prices Year: (2010)
Ref_id:b120 Title: Urban form, travel time, and cost relationships with tour complexity and mode choice Year: ()
Ref_id:b121 Title: Disparities in travel times between car and transit: Spatiotemporal patterns in cities Year: (2020-03-10)
Ref_id:b122 Title: Generation z and travel motivations: The impact of age, gender, and residence Year: (2025)
Ref_id:b123 Title: Rational and irrational vaccine hesitancy Year: (2023-03-12)
Ref_id:b124 Title: Predictors of covid-19 booster vaccine hesitancy among fully vaccinated adults in korea: a nationwide cross-sectional survey Year: (2022)
Ref_id:b125 Title: The nature and extent of covid-19 vaccination hesitancy in healthcare workers Year: (2021-04-20)
Ref_id:b126 Title: Trust in government, intention to vaccinate and covid-19 vaccine hesitancy: A comparative survey of five large cities in the united states, united kingdom, and australia Year: (2022)
Ref_id:b127 Title: Web-based social media intervention to increase vaccine acceptance: A randomized controlled trial Year: ()
Ref_id:b128 Title: Trust in the public health system as a source of information on vaccination matters most when environments are supportive Year: ()
Ref_id:b129 Title: When lack of trust in the government and in scientists reinforces social inequalities in vaccination against covid-19 Year: ()
Ref_id:b130 Title: Swissmetro -Zeroshot-CoT Year: ()
Ref_id:b131 Title: You will receive two blocks of text: <TRIP_INFO> . . . details like trip purpose, luggage, payment, origin, destination . . . </TRIP_INFO> <TRANSPORT_OPTIONS> . . . list of modes with travel time Year: ()
Ref_id:b132 Title: Use only the information in <TRIP_INFO> and <TRANSPORT_OPTIONS> Year: ()
Ref_id:b133 Title: Estimate a probability for each mode so they sum to 1 Year: ()
Ref_id:b134 Title: 'json { "Swissmetro": <float between 0 and 1>, "Train": <float between 0 and 1>, "Car": <float between 0 and 1> } "' No additional text; just the JSON object with normalized probabilities Year: ()
