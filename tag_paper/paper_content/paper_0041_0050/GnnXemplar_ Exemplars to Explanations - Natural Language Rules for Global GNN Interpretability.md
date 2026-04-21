Title: GNNXEMPLAR: Exemplars to Explanations -Natural Language Rules for Global GNN Interpretability
Abstract: Graph Neural Networks (GNNs) are widely used for node classification, yet their opaque decision-making limits trust and adoption. While local explanations offer insights into individual predictions, global explanation methods, those that characterize an entire class, remain underdeveloped. Existing global explainers rely on motif discovery in small graphs, an approach that breaks down in large, real-world settings where subgraph repetition is rare, node attributes are high-dimensional, and predictions arise from complex structure-attribute interactions. We propose GNNXEMPLAR, a novel global explainer inspired from Exemplar Theory from cognitive science. GNNXEMPLAR identifies representative nodes in the GNN embedding space, exemplars, and explains predictions using natural language rules derived from their neighborhoods. Exemplar selection is framed as a coverage maximization problem over reverse k-nearest neighbors, for which we provide an efficient greedy approximation. To derive interpretable rules, we employ a self-refining prompt strategy using large language models (LLMs). Experiments across diverse benchmarks show that GNNXEMPLAR significantly outperforms existing methods in fidelity, scalability, and human interpretability, as validated by a user study with 60 participants.

Section: Introduction and Related Works
Node classification using Graph Neural Networks (GNNs) has found wide-ranging applications across diverse domains, including protein function prediction [13,30], user profiling [47,11], and fraud detection [22]. Despite their success, GNNs, like other deep learning models, are often regarded as black boxes: they achieve high performance, but the reasoning behind their predictions remains largely opaque. To address this, GNN explainers aim to illuminate the decision-making process by identifying the substructures and features in the input graph that the model relies on for its predictions [1,2,48,24,39,18,42,16,17].
this section cite: ['b12', 'b29', 'b46', 'b10', 'b21', 'b0', 'b1', 'b47', 'b23', 'b38', 'b17', 'b41', 'b15', 'b16']

Section: Existing Works and Open Challenges
Existing methods for GNN explanation can be broadly categorized into two types: local and global. Fig. 4 in the Appendix provides the detailed taxonomy. 1. Local explanations focus on individual predictions. They aim to explain why a specific input graph (or node) received a particular label. While informative, these explanations are often instance-specific and do not generalize beyond the particular case being analyzed. 2. Global explanations, in contrast, seek to uncover general patterns that apply across multiple instances. They aim to explain the conditions under which a GNN assigns any input to a particular class, producing a single, interpretable explanation per class.
While local explanation techniques are relatively mature, global GNN explainers remain an emerging area of research, as evident from Fig. 4. In this work, we advance this direction by proposing a global explanation framework for node classification based on exemplar theory [28].
Exemplar theory, rooted in cognitive psychology, explains how humans categorize objects and ideas. It posits that individuals make category judgments by comparing new stimuli with previously encountered instances, called exemplars, stored in memory. These exemplars are not arbitrary; they tend to represent typical members of a category [28]. A new stimulus is assigned to a category based on the number and degree of similarities it shares with exemplars from that category.
We adapt this idea to GNN-based node classification by treating certain representative nodes in the embedding space as exemplars: nodes that encapsulate the characteristics of many others in the same class. For each exemplar, we extract an interpretable signature describing the distribution of features and/or structural patterns that characterizes the exemplar and the population of nodes it represents in the embedding space. Then, for any unseen node, we explain the GNN 's prediction by referencing the signature of the exemplar it most closely resembles. This approach enables global yet instance-relevant explanations grounded in learned graph structure and semantics.
Open Challenges: Existing global GNN explainers [50,42,51,45,1,2,25] have primarily targeted graph classification tasks on small-scale datasets such as molecular graphs. These methods typically aim to identify recurring substructures, commonly referred to as motifs or concepts, that the GNN being explained detects to classify graphs [50,42,51,45,25] Some explainers [1,2] go further by constructing the Boolean logic rules over motifs that aligns with the model's predictions. However, these motif-based explanation strategies face significant challenges when extended to node classification in large, real-world graphs with rich node attributes (Ex. citation and linked documents, financial networks, etc.)
• Attribute-Topology Interaction: In large graphs, GNN predictions are typically a complex function of both the graph structure and high-dimensional node attributes. This makes motif discovery difficult because the classical definition of a motif, grounded in graph isomorphism, handles only discrete node labels. Moreover, in real-world graphs, the repetition of the exact same subgraph with identical node attributes is rare, making the very notion of a motif ambiguous. This calls for a shift from exact, symbolic subgraph matching to an approximate space where repetitions of structurally and semantically similar patterns can be meaningfully observed. We address this by identifying recurring combinations of structure and attributes not in the raw input graph but in the GNN embedding space by locating dense neighborhoods.
• Computational Complexity: Motif discovery involves solving the subgraph isomorphism problem, which is NP-hard, making them prohibitively expensive on large graphs. The problem is further exacerbated when accounting for both topology and node attributes. Consequently, as we will show later in § 4, existing global GNN explainers often fail to scale on large graphs.
• Cognitive and Visual Overload: In small graphs, like molecules, motifs (e.g., functional groups) are compact and human-interpretable. However, in large real-world graphs, even the 2-hop neighborhood of a node can include hundreds or thousands of nodes. Visualizing or interpreting such patterns quickly becomes unwieldy and exceeds human cognitive limits.
this section cite: ['b27', 'b27', 'b49', 'b41', 'b50', 'b44', 'b0', 'b1', 'b24', 'b49', 'b41', 'b50', 'b44', 'b24', 'b0', 'b1']

Section: Contributions
In this work, we present GNNXEMPLAR, which addresses the limitations outlined above.
• Problem formulation: Effective explanations of GNN predictions must satisfy two key criteria:
(1) they should be faithful, meaning they accurately reflect the model's decision-making process, and (2) they should be interpretable, allowing humans to understand the reasoning behind predictions despite the underlying complexity of the graph modality. To address these dual objectives, we take inspiration from Exemplar Theory [28]. Exemplar theory posits that humans categorize new instances by comparing them to representative examples (exemplars) previously encountered. By adapting this principle, we explain the GNN 's prediction for a node by referencing similar, representative nodes in the embedding space -its exemplars. Furthermore, to enhance interpretability, we move beyond subgraph visualizations which are inherently ineffective in large, dense graphs due to their complexity and scale. Instead, we distill the defining characteristics of each exemplar and its associated population into textual explanations. This shift to natural language enables more accessible and cognitively manageable insights into the GNN 's decisionmaking process. We validate this claim through a user survey ( § 4.7).
• Novel methodology: We develop GNNXEMPLAR, which integrates several innovative components. First, exemplar identification is cast as a coverage maximization problem over reverse k-nearest neighbor relationships. We prove that this problem is NP-hard and submodular, and accordingly propose a greedy algorithm with a (1 -1 e ) approximation guarantee to the optimal solution. To uncover the signature characteristics of each exemplar and the population it represents, we leverage LLMs to iteratively propose and refine interpretable logical rules that are consistent with the GNN's predictions.
• Empirical analysis: We conduct extensive experiments on a diverse suite of homophilous and heterophilous graphs. Our analysis reveals that: (1) existing GNN explainers are inadequate for node classification on large graphs with complex node attributes; (2) GNNXEMPLAR provides high-fidelity explanations of GNN predictions; and (3) the text-based explanations are preferred over subgraph visualization, as validated through a user study involving 60 participants.
this section cite: ['b27']

Section: Preliminaries and Problem Formulation
Definition
1 (Graph). A graph is defined as G = (V, E, X) over a node set V, edge set E = {(u, v) | u, v ∈ V} and node attributes X = {x v | v ∈ V} where x v ∈ R d is the set of features characterizing each node.
Definition 2 (Node Classification). In node classification, we are given a single input graph G = (V, E, X), where a subset of nodes V tr ⊂ V is associated with known class labels from the set {y 1 , • • • , y c }. The objective is to train a GNN model Φ such that, for any node v ∈ V \ V tr , the prediction error of its class label is minimized.
Error may be measured using known metrics such as cross-entropy loss, negative log-likelihood, etc.
this section cite: []

Section: Definition 3 (Exemplars).
Exemplars refer to representative nodes within the graph that embody prototypical structural and attribute characteristics shared by many other nodes in the same class.
In the context of node classification, these exemplars serve as anchors for human-understandable explanations. Each exemplar, e, is associated with a subset of training nodes, R e ⊆ V tr , which represents the population that e exemplifies. A node e ∈ V tr is called an exemplar for the class c ∈ Y if |R e | ≥ τ , where
R e =    v ∈ V tr Φ(e) = c, Φ(v) = c, d(z e , z v ) ≤ δ   
where z v ∈ R d is the GNN-learned embedding of node v, d(z e , z v ) is some distance function, δ ∈ R is a distance threshold, and τ ∈ N is a minimum population size hyperparameter.
As we will see later, the thresholds will automatically be discovered from the data by exploiting the k-nearest neighbor relationship along with a greedy selection algorithm.
this section cite: []

Section: Definition 4 (Exemplar Signature).
The signature of exemplar e, denoted as σ e , is a boolean formula composed of interpretable conditions over the structural and attribute properties of the local neighborhood of e. This formula is constructed such that its truth value approximates Φ's predictions over R e , i.e., with high probability ∀v ∈ R e , : σ e (v) = Φ(v). The signature σ e of exemplar e from class c is a boolean function f (π 1 (v), π 2 (v), . . . , π k (v)) ∈ {0, 1}, such that, with a high probability, ∀v ∈ R e , σ e (v) = 1, iff Φ(v) = c. Here, π i are the predicates that evaluate to Boolean conditions over interpretable properties such as:
• Node features (e.g., x v [age] > 30)
• Local structure (e.g., degree(v) ≥ 3)
• Neighborhood composition (e.g., fractionClass(v, hop = 1, c ′ ) > 0.6)
For instance, an exemplar of fraudulent nodes in a transaction graph, might be an account that makes frequent, low-value transfers to many recently created accounts that show no further activity.
Assuming each node has attributes indicating the average transaction value and average frequency of transactions per week, its signature could be a rule like: "Has an average transaction amount below $100, performs more than 10 transactions per week, and is dissimilar in transaction frequency from the majority of its neighbors" Problem 1 (Global GNN Explanation through Exemplar Signatures). Let Φ be a trained GNN that assigns to each node v ∈ V a class label from {y 1 , • • • , y c }. For each class y i , let E i ⊆ V denote a selected set of exemplar nodes that are representative of class y i , and let σ e be the boolean signature associated with exemplar e ∈ E i .
The goal is to construct a global explanation in the form of a Boolean formula f i for each class y i , such that:
f i (v) = e∈Ei σ e (v)
where, ∀v ∈ V tr , Φ(v) = y i ⇔ f i (v) = TRUE. Each f i serves as the global explanation for class y i , built by aggregating exemplar-level signatures via logical OR. Each exemplar signature is free to use all Boolean operators.
The formulation of explanation through exemplars presents two central challenges: 1. How do we identify an optimal set of exemplars? Exemplar selection is a combinatorial optimization problem over the space of all nodes in the graph. Choosing too many exemplars compromises both interpretability and computational efficiency. A small, well-chosen exemplar set enables concise global explanations, as the final explanation takes the form of a disjunction over exemplar-specific rules. 2. How do we derive boolean logic signatures for each exemplar? The space of candidate logical expressions, composed of node attributes and structural features, is exponentially large, rendering exhaustive search intractable. Additionally, these rules must strike a balance between fidelity to the GNN's predictions and human interpretability. A further challenge arises from the need to express these logical rules as natural language descriptions requiring semantic precision.
this section cite: []

Section: Methodology
Fig. 1 presents the pipeline of GNNXEMPLAR. There are two distinct components: first, we identify a budget-constrained set of exemplars, and next, we extract boolean rules, expressed in natural language, through iterative self-refining using LLM. We next detail each of these steps.
this section cite: []

Section: Exemplar Identification.
GNNXEMPLAR aims to identify a set of b exemplar nodes that optimally represent the full training set; b is a tunable hyper-parameter balancing complexity of the explanation with higher expressivity. This selection process is guided by two critical criteria:
• Representativeness: Each exemplar node should be close to a large number of other nodes sharing the same GNN-predicted class label in the embedding space, thereby capturing common structural and attribute patterns learned by the model. We quantify representativeness using the notion of reverse k-nearest neighbors ( § 3.2) in the GNN embedding space. Specifically, if node v appears frequently in the k-NN sets of other nodes with the same predicted label, it is considered representative of those nodes' learned semantics. Such nodes are prioritized for inclusion in the exemplar set, as they enable coverage of large regions of the embedding space and are likely to approximate the model's behavior over similar instances.
• Diversity: The selected exemplars should be well spread out in the GNN embedding space to ensure a wide range of model behaviors are covered ( § 3.3).
By jointly optimizing for representativeness and diversity, GNNXEMPLAR ensures that the selected exemplars collectively span a broad range of the model's learned representations.
this section cite: []

Section: Quantifying Representativeness through Reverse k-NN
The representative power of a node is defined as follows.
Definition 5 (Reverse k-NN and Representative Power). Let h v denote the GNN embedding of node v. The k nearest neighbors of node v, denoted as k-NN(v), are the k nodes from the train set with embeddings closest to h v and sharing the same GNN predicted class label (we use L 2 distance, but other distances may also be used). The reverse k-NN of node v is the set of nodes for which v appears in their k-NN set. Formally,
Rev-k-NN(v) = {u ∈ V tr | v ∈ k-NN(u), Φ(v) = Φ(u)}
The representative power of node v is then defined as
Π(v) = |Rev-k-NN(v)| | {u ∈ V tr | Φ(v) = Φ(u)} |
A high value of Π(v) indicates that node v frequently appears in the k-NN sets of many other nodes, implying it resides in a dense region of the embedding space and captures shared representational characteristics. Thus, it is a strong candidate for inclusion in the exemplar set.
this section cite: []

Section: Sampling for Scalable Computation of Reverse k-NN
Computing the k nearest neighbors for each node consumes O(n log k) ≈ O(n) time, since k ≪ n and n = |V tr | is the number of nodes. Thus, computing k-NN for all nodes and then constructing the Rev-k-NN requires O(n 2 ) time, which becomes prohibitively expensive on large-scale graphs.
To address this, we adopt a sampling-based strategy that enables scalable approximation of reverse k-NN with theoretical [12]. Let S ⊂ V tr be a uniformly sampled subset of z ≪ n nodes. We compute the k-NN only for nodes in S, which reduces the computational cost to O(zn).
We then define the approximate reverse k-NN of a node v ∈ V tr as Rev-k-NN(v) = {u ∈ S | v ∈ k-NN(u), Φ(v) = Φ(u)}. Hence, the approximate representative power of v is
Π(v) = | Rev-k-NN(v)| | {u ∈ S | Φ(v) = Φ(u)} |
Here, Π(v) estimates how broadly node v is retrieved as a nearest neighbor across the sample set. The sample size z controls the trade-off between accuracy and efficiency. Leveraging Chernoff bounds, we establish that even a small z yields high-confidence approximations: Lemma 1. Given an error threshold θ and confidence level 1 -δ, it suffices to sample z ≥ ln ( 2 δ )(2+θ)
θ 2
nodes to ensure that for any v ∈ V tr , P Π(v) -Π(v) ≤ θ ≥ 1 -δ.
this section cite: ['b11']

Section: PROOF. See App. A.3.
This result has two key implications:
• The required number of samples is independent of the number of nodes in the train set, i.e., |V tr |.
• Since z scales logarithmically with ln(1/δ), even a small sample ensures high-probability bounds. Thus, the overall computation cost for reverse k-NN be comes O(n) instead of O(n 2 ).
While we use the notations Rev-k-NN (v) and Π(v) in the subsequent discussion, approximate variants via sampling may be used in practice. We evaluate the quality-efficiency trade-off in App. §A.7.
this section cite: []

Section: Coverage Maximization
We aim to identify a subset of b exemplar nodes that collectively offer the broadest coverage over the training set in the GNN embedding space. Definition 6 (Exemplar Set). Let the representative power of a set of exemplar nodes A ⊆ V be defined as:
this section cite: []

Section: Algorithm 1 Greedy Node Selection
Π(A) = v∈A Rev-k-NN(v) |Vtr| (1
)
Here, Rev-k-NN(v) denotes the reverse k nearest neighbors of node v in the GNN embedding space.
Given a node set V and budget b, we seek a subset of train nodes A * ⊆ V tr of size b that maximizes:
A * = arg max A⊆Vtr,|A|=b Π(A)(2)
Theorem 1. Maximizing the representative power in Eq. 2 is NP-hard.
PROOF. See App. A.4 for a reduction from the classic Set Cover Problem [5]. □ Fortunately, the objective Π(A) is monotone and submodular, making it amenable to greedy approximation.
Theorem 2. The greedy algorithm (Alg. 1) guarantees Π(A greedy ) ≥ 1 -1 e Π(A * ).
this section cite: ['b4']

Section: PROOF. App. A.5 shows the monotonicity and submodularity of Π(A).
An analogous proof for the case using the approximate Rev-kNN is provided in App. A.6. □ Alg. 1 iteratively selects nodes that maximize marginal gain in coverage. By exploiting the transitivity of proximity in the embedding space, this greedy strategy avoids redundancy: if two nodes share most of their reverse k-NN sets, selecting one reduces the marginal utility of the other. Thus, the algorithm promotes both coverage and diversity within the budget.
this section cite: []

Section: Discovery of Exemplar Signatures
Our goal is to analyze the embeddings of an exemplar node and its Rev-k-NN set, and derive a symbolic Boolean rule, expressed in natural language, that closely matches the GNN's predictions on this population (see Prob.1). To this end, we leverage the reasoning capabilities of large language models (LLMs), motivated by two key factors. (1) LLMs have demonstrated strong mathematical and causal reasoning abilities, including over graph-structured data [55,6,26,49]. (2) Given our aim to express logical rules in natural language, LLMs are particularly well suited due to their well-established strengths in linguistic articulation. Self-refinement paradigm: Fig. 2 presents the pipeline of the signature discovery process of an exemplar, which leverages the self-refine platform [26]. We select an initial sample of positive and negative nodes. The positive sample contains nodes from the Revk-NN set, and the negative sample contains those not in the Rev-k-NN. Both samples are drawn uniformly at random. The sample size is a hyperparameter. These node sets are further divided into training and validation sets. Instead of asking the LLM to directly output the boolean rule in natural language, we decouple it into two steps. The LLM is first asked to generate a python code, that takes as input a node and its associated information, passes it through a boolean logic, and returns a True/False answer. The ideal rule returns True for all positive nodes and False for negative nodes. The rule encoded in the Python code is then translated to natural language by the LLM. Initially, the Python code is a trivial solution, such as returning True for any input node, which is iteratively improved through feedback. The iterations stop when either the accuracy of the code (rule) exceeds a certain accuracy threshold on the validation set or the number of iterations reaches an upper limit.
this section cite: ['b54', 'b5', 'b25', 'b48', 'b25']

Section: Prompt specification:
The prompt includes the following components; a visual overview is provided in Fig.  • Training and validation nodes: As discussed above, we provide a set of positive and negative nodes. If the GNN being explained is ℓ layers deep, then the embedding of a node is a function of its ℓ-hop neighborhood only [44]. Hence, we provide a summary of the ℓ-hop neighborhoods of each node in the positive and negative samples. The summary of a node includes: (1) the attributes of the node, (2) the normalized frequency distribution of GNN predicted class labels for all nodes in each hop from 1 to ℓ, (3) the average L 1 distance per attribute between the exemplar and all nodes at each hop level from 1 to ℓ. While GNN class distribution signals the degree of homophily-bias learned by the GNN, the average attribute-wise distance offers insight into feature relevance in the embedding space. Specifically, if the average distances to positive and negative nodes are similar for a given attribute, it suggests that the feature contributes little to the embedding representation. Note that, since we only provide distribution-level information, the context size remains independent of the graph's density. This property is crucial for ensuring that large graphs do not exceed the LLM's context window capacity.
• Code skeleton: The skeleton of the python code that the LLM is supposed to fill in. The skeleton contains the function definition that takes as input the summary of a node and the output specification, which should return a boolean value.
• Feedback: This includes the Python code and the associated natural language rule from the last iteration and the summary of all nodes where the latest rule failed to match the GNN prediction.
this section cite: ['b43']

Section: Experiments
In this section, we benchmark GNNXEMPLAR and establish:
• Limitations of Existing Explainers: Current explainers, primarily designed for graph classification on small graphs, fail to generalize effectively to node classification in large-scale graphs.
• High-Fidelity explanations: GNNXEMPLAR bridges this gap by consistently achieving high fidelity across both homophilic and heterophilic graph benchmarks.
• User Preference for Textual Explanations: Our user survey involving 60 participants shows a statistically significant preference for textual explanations over graph-based visualizations.
this section cite: []

Section: Experimental Setup
The details of our hardware and software platform, hyper-parameters, LLM engine, training details of the black-box GNN and their accuracies are discussed in App. A.1 and App. A.2. Our codebase is shared at https://github.com/idea-iitd/GnnXemplar.git.
Datasets: Table 1 presents the 8 benchmark datasets we use. Wherever available, we adopt the standard train/validation/test splits from PyTorch Geometric or the original data releases, preserving class balance. We train a GAT for TAGCora and GCN for the rest.
Baselines: As discussed in § 1, there are no existing explainers designed to handle node classification on large graphs. Hence, we adapt state-of-the-art explainers originally developed for graph classification. To enable a fair comparison, we reframe the node classification task as a graph classification problem by extracting the ℓ-hop subgraph around each target node and assigning the node's label to the entire subgraph; ℓ denotes the number of layers in the GNN. We compare against the following baselines: • GNNInterpreter [42]: A generative model trained to synthesize class-representative graphs using reinforcement learning.
• GCNeuron [45]: A neuron-level explainer that identifies high-level subgraph concepts that activate neurons within a GNN.
• GLGExplainer [2]: The only prior global logical explainer that fits a Boolean formula to GNN outputs by clustering local explanation subgraphs (e.g., PGExplainer [24]) around prototypes and learning a formula over those prototypes using ELEN [4].
Since both GNNInterpreter and GCNeuron identify representative subgraphs without generating explicit Boolean rules, we construct a rule by taking a logical OR over all identified subgraphs for each class label. We do not compare with GraphTrail [1] since it only considers graphs with discrete node labels. MAGE [50] is also omitted since it is specific to molecular graphs.
this section cite: ['b41', 'b44', 'b1', 'b23', 'b3', 'b0', 'b49']

Section: Metrics:
We evaluate each explainer by applying its generated Boolean formula to the test set of each dataset. The alignment between the formula and the GNN's predictions is quantified using the following metrics: Fidelity (the proportion of test nodes where the formula's output matches the GNN's prediction), Precision, Recall, and F1-score.
this section cite: []

Section: Fidelity
Table 2 reports the fidelity of each explainer on node classification tasks. Precision, recall and Fscore for the same experiments are reported in Tables 6, 7 and 8 in the Appendix. Several key observations emerge. (1) GNNXEMPLAR consistently achieves the highest fidelity, establishing a new benchmark for global explanation in node classification. (2) Explainers for graph classification demonstrate brittle performance when adapted to node classification, confirming that exact subgraph repetition is rare in real-world graphs. (3) motif-based explainers reliant on subgraph isomorphism fails to scale on large graphs. We elaborate below.
NAs: The NA cases occur when an explanation is generated but cannot be applied or evaluated on any specific node (or neighborhood subgraph) instance. This limitation arises from the reliance of GNNInterpreter on subgraph-based rules. Its only mode of application is through subgraph isomorphism, which severely constrains its generalizability. Subgraph isomorphism can only function when node features are either absent or purely discrete (e.g., scalars). With continuous or highdimensional node features, determining whether two nodes, or their neighborhoods, are "identical" becomes ill-defined, rendering the explanation unusable in most real-world settings.
NFs: NF stands for "No Formula Generated," which is encountered in GLGExplainer. GLGExplainer first trains a surrogate model to mimic the black-box GNN and then attempts to distill this surrogate into interpretable boolean logic. However, we find that in several datasets, either the surrogate model fails to approximate the GNN, or the boolean distillation step breaks down. In both scenarios, GLGExplainer outputs an empty string. This failure indicates that GLGExplainer is brittle when faced with noisy, irregular, or highly entangled graph patterns.
Inferior Baseline Performance. Even when the baselines produce actionable rules, all three yield significantly lower fidelity. This is primarily because they rely on identifying common subgraph patterns per class, an assumption that rarely holds in complex, real-world graphs. In contrast, GN-NXEMPLAR is grounded in exemplar theory and computes distributional distances to strategically selected exemplars in the GNN embedding space. This approach avoids subgraph isomorphism while remaining faithfully aligned with the GNN's predictive behavior. Scalability: Both GLGEX-PLAINER and GCNEURON frequently encounter out-of-memory (OOM) errors on large or dense graphs, with GLGEXPLAINER failing because of its reliance on subgraph-isomorphism and GC-NEURON due to exhaustive neuron-activation pattern enumeration. In contrast, GNNXEMPLAR is explicitly designed to avoid subgraph isomorphism and scales gracefully.
this section cite: []

Section: Ablation Study
To quantify the contributions of our two key innovations, exemplar selection through Rev-k-NN and iterative self-refinement using LLM, we conduct two controlled ablations.
Rev-k-NN vs. Random Exemplar Selection: Here, we replace our Rev-k-NN based exemplar selection with an equal-sized set of randomly sampled nodes (exemplars) per class. This change yields a noticeable fidelity decrease, as random prototypes fail to capture the dense, semantically coherent neighborhoods that Rev-k-NN provides (see Fig. 3-Left subplot).
Self-Refinement vs. Zero-Shot: In the zero-shot variant, we bypass the structure self-refining pipeline discussed in § 3.4 and directly ask the LLM to predict the rule (signature) in one go. Ablation results in Fig. 3 (Right subplot) demonstrate that this zero-shot approach yields noticeably lower fidelity and higher variance on every dataset. This reveals that the full self-refinement pipeline systematically identifies misclassified instances and adjusts the rules accordingly, producing significantly more accurate and robust explanations with reduced variance across runs.
this section cite: []

Section: Impact of Parameters and GNN architectures
App. A.7 studies the impact of parameters k in Rev-k-NN, sample size in Rev-k-NN approximation, and the training-set sample size in self-refine. App. A.8 evaluates robustness to GNN architectures.
this section cite: []

Section: Case Study: Visual Analysis of Rules
To demonstrate the interpretability and domain-alignment of our global explanations, we show representative rule sets from TAGCora (homophilous) (Fig. 14) and BAShapes (heterphilous) (Fig. 15).
TAGCora (Fig. 14) In the TAGCora citation network, topic (class) assignments are driven by both textual features and neighborhood context. For instance, Neural Networks papers are identified by the presence of domain-specific terms like "backpropagation" and "activation functions". Furthermore, the rule also discovers homophillic associations with other "Neural Network" papers among 1 and 2-hop neighbors. The rule further comments that the presence of ICA and HMM words in the abstracts of the neighbors further reinforces the class assignment.
BAShapes (Fig. 15) On the synthetic BAShapes graph, membership in each geometric role is determined solely by local connectivity patterns. GNNXEMPLAR is successfully able to identify these purely structural and heterophilic associations, such as nodes belonging to the class "not in a house" have at least six "non-house" neighbors and minimal ties to other roles.
this section cite: []

Section: Citeseer

this section cite: []

Section: Application: Diagnostic Power in Failure Cases
In Questions, active users exhibit homophilous behavior-they tend to connect to other active users. Inactive users, in contrast, behave heterophilously by connecting primarily to active users. Interestingly, our explanation rule for the inactive class does not reflect this expected heterophily. Instead, it points toward a homophilous pattern even for inactive users. At first glance, this seems contradictory: how can an explanation be so seemingly incorrect and still achieve high fidelity (see Table 2)?
To investigate, we visualized 2-hop neighborhoods of inactive nodes: once with ground-truth labels, and once with GNN predictions (see example in Fig. 5). The results were revealing: the GNN predicts even the neighbors of inactive nodes as inactive, indicating that it has learned an incorrect homophilous pattern for the inactive class as well. This behavior is likely driven by the dataset's class imbalance -most nodes are active, so homophily becomes a statistically advantageous shortcut for the model.
Far from being flawed, the explanation is insightful-it does not mimic the ground truth, but rather points to the root cause of why the inactive class is often misclassified with a low precision of 0.68: the GNN has failed to capture the intended heterophilous behavior.
this section cite: []

Section: Human Evaluation: User Survey
In this work, we advocate a shift from conventional subgraph-based visualizations to natural language explanations. Our motivation stems from the hypothesis that presenting dense graph neighborhoods with high-dimensional node attributes in full detail can overwhelm human cognitive capacity. To test this, we conducted a user study with 60 participants, each responding to 5 A/B test questions, resulting in 300 total comparisons. Each question presented the same explanation for a model prediction in both textual and subgraph form, and participants were asked to indicate their preferred modality. See App. B for survey design details and example screenshots.
We analyzed the results using the Binomial test [37] and McNemar's test [27]. The Binomial test, applied to the aggregate responses (200/300), evaluates whether one modality was preferred significantly more often overall. This yielded a highly significant result (p-value < 0.0001), indicating that text-based explanations were favored by participants at the population level. Table 3 presents the per-question binomial test results, showing that this preference was consistent and statistically significant in all except Q3. While Q3 does not reach statistical significance, this was by design: we deliberately selected an explanation where the subgraph visualization was small and simple. The goal was to verify that participants were not blindly preferring the text modality due to any prior bias.
To assess within-participant consistency, we additionally conducted McNemar's test using the 'exact Binomial' method. Out of 60 participants, 45 chose text more often, while only 15 preferred subgraph more often. This difference was also statistically significant (p-value = 0.0001), confirming that the preference for text was not only strong but consistent across individuals.
this section cite: ['b36', 'b26']

Section: Conclusions, Limitations and Future Works
In this work, we presented GNNXEMPLAR, a novel framework for global explanation of node classification in GNNs. Unlike prior explainers that rely on brittle motif discovery and struggle with scale, GNNXEMPLAR explains through exemplar nodes, which serve as semantic anchors for model behavior. To generate human-understandable explanations, we leveraged LLMs to distill the defining characteristics of each exemplar and its neighborhood into concise natural language rules. Our empirical evaluations demonstrate that GNNXEMPLAR outperforms existing methods in fidelity, scalability, and user interpretability, with a user study confirming strong preference for textual explanations.
Limitations and Future Works: GNNXEMPLAR operates in a setting with access only to the GNN's embedding space, limiting visibility into how feature-topology interactions influence internal activations. As a result, fine-grained mechanistic understanding of the model's decision process remains out of reach. We plan to study this aspect through the lens of mechanistic interpretability.
this section cite: []

Section: References
Ref_id:b0 Title: Graphtrail: Translating gnn predictions into human-interpretable logical rules Year: (2024)
Ref_id:b1 Title: Global explainability of GNNs via logic combination of learned concepts Year: (2023)
Ref_id:b2 Title: Explainability techniques for graph convolutional networks Year: (2019)
Ref_id:b3 Title: Entropy-based logic explanations of neural networks Year: (2022)
Ref_id:b4 Title:  Year: (2009)
Ref_id:b5 Title: How do large language models understand graph patterns? a benchmark for graph pattern comprehension Year: (2025)
Ref_id:b6 Title: Graphsvx: Shapley value explanations for graph neural networks Year: (2021)
Ref_id:b7 Title: Degree: Decomposition based explanation for graph neural networks Year: (2022)
Ref_id:b8 Title: Z orro: Valid, sparse, and stable explanations in graph neural networks Year: (2022)
Ref_id:b9 Title: Citeseer: An automatic citation indexing system Year: (1998)
Ref_id:b10 Title: Persona identification in e-commerce with scarce labels and in-context graph learning Year: (2025)
Ref_id:b11 Title: Bonsai: Gradient-free graph condensation for node classification Year: (2025)
Ref_id:b12 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b13 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b14 Title: Graphlime: Local interpretable model explanations for graph neural networks Year: (2022)
Ref_id:b15 Title: Global counterfactual explainer for graph neural networks Year: (2023)
Ref_id:b16 Title: Gcfexplainer: Global counterfactual explainer for graph neural networks Year: (2025-08)
Ref_id:b17 Title: GNNX-BENCH: Unravelling the utility of perturbation-based GNN explainers through in-depth benchmarking Year: (2024)
Ref_id:b18 Title: Dag matters! gflownets enhanced explainer for graph neural networks Year: (2023)
Ref_id:b19 Title: Generative causal explanations for graph neural networks Year: (2021)
Ref_id:b20 Title: One for all: Towards training one graph model for all classification tasks Year: (2024)
Ref_id:b21 Title: Improving fraud detection via hierarchical attention-based graph neural network Year: (2022)
Ref_id:b22 Title: GOAt: Explaining graph neural networks via graph output attribution Year: (2024)
Ref_id:b23 Title: Parameterized explainer for graph neural network Year: (2020)
Ref_id:b24 Title: On data-aware global explainability of graph neural networks Year: (2023)
Ref_id:b25 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b26 Title: Note on the sampling error of the difference between correlated proportions or percentages Year: (1947)
Ref_id:b27 Title: Context theory of classification learning Year: (1978)
Ref_id:b28 Title: Wiki-cs: A wikipedia-based benchmark for graph neural networks Year: (2020)
Ref_id:b29 Title: Graphreach: Position-aware graph neural network using reachability estimations Year: (2021)
Ref_id:b30 Title: Distill n'explain: explaining graph neural networks using simple surrogates Year: (2023)
Ref_id:b31 Title: A critical look at the evaluation of gnns under heterophily: Are we really making progress? Year: (2023)
Ref_id:b32 Title: Explainability methods for graph convolutional neural networks Year: (2019)
Ref_id:b33 Title: Interpreting graph neural networks for nlp with differentiable edge masking Year: (2021)
Ref_id:b34 Title: Higher-order explanations of graph neural networks via relevant walks Year: (2021)
Ref_id:b35 Title: Reinforcement learning enhanced explainer for graph neural networks Year: (2021-12)
Ref_id:b36 Title: Handbook of parametric and nonparametric statistical procedures Year: (2003)
Ref_id:b37 Title: Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning Year: (2022)
Ref_id:b38 Title: InduCE: Inductive counterfactual explanations for graph neural networks Year: (2024)
Ref_id:b39 Title: Pgm-explainer: Probabilistic graphical model explanations for graph neural networks Year: (2020)
Ref_id:b40 Title: Xiangnan He, and Tat-Seng Chua. Towards multigrained explainability for graph neural networks Year: (2021)
Ref_id:b41 Title: GNNInterpreter: A probabilistic generative model-level explanation for graph neural networks Year: (2023)
Ref_id:b42 Title: Task-agnostic graph explanations Year: (2022)
Ref_id:b43 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b44 Title: Global concept-based interpretability for graph neural networks via neuron analysis Year: ()
Ref_id:b45 Title: Global concept-based interpretability for graph neural networks via neuron analysis Year: (2023)
Ref_id:b46 Title: Graph convolutional neural networks for web-scale recommender systems Year: (2018)
Ref_id:b47 Title: Gnnexplainer: Generating explanations for graph neural networks Year: (2019)
Ref_id:b48 Title: Large language models meet graph neural networks: A perspective of graph mining Year: (2025)
Ref_id:b49 Title: Model-level graph neural networks explanations via motif-based graph generation Year: (2025)
Ref_id:b50 Title: Xgnn: Towards model-level explanations of graph neural networks Year: (2020)
Ref_id:b51 Title: On explainability of graph neural networks via subgraph explorations Year: (2021)
Ref_id:b52 Title: Explaining graph neural networks with structure-aware cooperative games Year: (2022)
Ref_id:b53 Title: Relex: A model-agnostic relational model explainer Year: (2021)
Ref_id:b54 Title: Causal graph discovery with retrieval-augmented generation based large language models Year: (2024)
