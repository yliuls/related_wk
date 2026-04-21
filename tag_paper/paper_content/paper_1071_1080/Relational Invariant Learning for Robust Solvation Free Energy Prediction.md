Title: Relational Invariant Learning for Robust Solvation Free Energy Prediction
Abstract: Predicting the solvation free energy of molecules using graph neural networks holds significant potential for advancing drug discovery and the design of novel materials. While previous methods have demonstrated success on independent and identically distributed (IID) datasets, their performance in out-of-distribution (OOD) scenarios remains largely unexplored. We propose a novel Relational Invariant Learning framework (RILOOD) to enhance OOD generalization in solvation free energy prediction. RILOOD comprises three key components: (i) a mixup-based conditional modeling module that integrates diverse environments, (ii) a novel multi-granularity refinement strategy that extends beyond core substructures to enable context-aware representation learning for capturing multilevel interactions, and (iii) an invariant learning mechanism that identifies robust patterns generalizable to unseen environments. Extensive experiments demonstrate that RILOOD significantly outperforms state-of-the-art methods across various distribution shifts, highlighting its effectiveness in improving solvation free energy prediction under diverse conditions.

Section: Introduction
Predicting the solvation free energy of molecules is crucial, as most chemical and pharmaceutical processes occur in solution, making it highly significant for downstream industries (Chung et al., 2022;Varghese & Mushrif, 2019). This task, often referred to as Solute-Solvent Interaction in Molecular Relational Learning (MRL) (Lim & Jung, 2019;Subramanian et al., 2020;Panwar et al., 2021;Low et al., 2022;Zhang et al., 2022;Lee et al., 2023a;b), focuses on understanding and modeling the interactions between solutes and solvents, conceptualizing these interactions as Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). solvation properties of molecules. More importantly, it extends traditional molecular property prediction frameworks by explicitly incorporating solvent molecules as input features, thereby improving prediction accuracy and enhancing chemical interpretability.
Despite significant advancements in MRL, most existing methods operate under the assumption that training and test data are independent and identically distributed (IID). However, real-world molecular systems exhibit diverse characteristics and uneven data distributions across solvents, making this assumption unrealistic in practical applications. Out-of-Distribution (OOD) scenarios arise when test data differs substantially from training data, as illustrated by the example in Fig. 1. This work focuses on exploring the OOD generalization of molecular solvation properties across different environments within the MRL.
To address distribution shifts, several approaches have been proposed, including invariant learning (Wu et al., 2022a), feature disentanglement (Liu et al., 2021), and data augmentation (Sui et al., 2024;Jia et al., 2024). Among these, invariant learning for OOD generalization (Krueger et al., 2021) has garnered significant attention due to its ability to extract robust features that remain stable across different environments, even under distribution shifts. In molecular modeling, molecular invariant learning is commonly employed to address distribution shifts by identifying core substructures that exhibit strong correlations with molecular properties. However, existing methods encounter notable limitations. For example, Lee et al. (2023a) leverage privileged substructures as causal correlations in MRL. However, they do not account for the solvent-dependent nature of solute properties and the complex coupling effects that govern molecular behavior (Cramer & Truhlar, 2008). Similarly, Lee et al. (2023b) apply back-door adjustment to mitigate spurious correlations but fails to account for solvent effects, thereby neglecting intricate solute-solvent interactions that are essential for accurately characterizing solute properties. As a result, these approaches lead to an incomplete understanding of solute behavior, increased susceptibility to spurious correlations, and poor generalization to unseen environments.
Although atomic interactions have been extensively modeled and have shown success in MRL, a precise under- standing of the solute-solvent interaction remains elusive. Achieving both accuracy and explainability presents significant challenges, particularly in the following areas: (1) the need to accurately model solute-solvent interactions across chemically diverse solvent environments. (2) the inherent complexity of multilevel molecular interactions, which hinders the extraction of invariant features and the construction of robust, generalizable representations. To address these issues, it is essential to develop models that can accurately represent multilevel molecular interactions, effectively capturing the complex one-to-many relationships between solutes and properties.
Based on the aforementioned analysis, in this work, we propose a novel Relational Invariant Learning framework for Out-of-Distribution Generalization (RILOOD) in MRL. Unlike traditional methods, our framework explicitly captures invariant relationships in molecular pairs and achieves a more generalized representation of solute-solvent interactions. Specifically, we first employ a Graph Neural Network (GNN) to encode molecular structures, followed by a crossattention module to map atom-level interactions. We then incorporate mixup-enhanced Conditional Variational Modeling to facilitate cross-environment invariance, leveraging a multi-granularity context-aware interaction mechanism and environment diversity inference. This enables learning of interaction invariance (Xie et al., 2024), allowing the discov-ery of fundamental molecular relationships in a chemically interpretable latent space. Our main contributions can be summarized as follows:
• We formally formulate the out-of-distribution (OOD) generalization problem in Molecular Relational Learning (MRL), establishing a rigorous foundation for studying model robustness across diverse chemical environments.
• We propose RILOOD, a relational invariant learning framework for solvation free energy prediction, featuring three key components: a mixup-based conditional modeling module, a multi-granularity refinement strategy, and an invariant learning mechanism.
• We conduct extensive experiments across multiple distribution shifts, demonstrating that RILOOD consistently outperforms state-of-the-art methods, significantly advancing OOD generalization in molecular property prediction.
this section cite: ['b3', 'b35', 'b17', 'b32', 'b26', 'b20', 'b45', 'b19', 'b33', 'b10', 'b13', 'b4', 'b42']

Section: Related Works

this section cite: []

Section: Molecular Relational Learning
Molecular Relational Learning (Lim & Jung, 2019;Pathak et al., 2020;Subramanian et al., 2020;Lee et al., 2023a;b), which aims to study the relationship between molecules, can be divided into molecular interaction prediction and Drug-Drug Interaction prediction. Molecular interaction prediction, i.e., solvent-based molecular property prediction, includes solvent free energy prediction, solubility prediction, chromophore absorption prediction, and so on. Unlike traditional molecular property prediction, the model need predict the properties exhibited by the same molecule exposed to multisolvent. Recent works (Ramani & Karmakar, 2024;Du et al., 2024) leverage the merged graph to encode atomic interaction and further improve interpretability and reducing redundancy.
this section cite: ['b17', 'b27', 'b32', 'b29', 'b7']

Section: Out of Distribution Generalization
Generalizing well-trained models to unseen environments with different data distributions remains a key challenge in machine learning. To address OOD generalization, three main approaches are typically employed: invariant learning (Li et al., 2022), causal inference (Dawid, 2000), and disentangled learning (Mo et al., 2023). Invariant learning aims to extract stable features across distribution shifts, but ZIN (Lin et al., 2022) argues that identifying invariance in Euclidean data is impossible without environment labels, proposing auxiliary information as a solution. Causal inference approaches utilize Structural Causal Models (SCM) (Chen et al., 2022;Lu et al., 2021) and Independent Causal Mechanisms (ICM) (Peters et al., 2017;Gui et al., 2024) to filter spurious correlations and enhance robust feature discovery. Disentangled learning separates features into invariant factors, which generalize across distributions, and spurious factors, which exhibit unstable correlations. While effective, it relies on strong prior assumptions and carefully curated datasets. These diverse strategies collectively tackle OOD generalization by distinguishing stable predictive patterns from environment-dependent variations, yet significant challenges remain in accurately identifying and effectively leveraging invariant features.
this section cite: ['b16', 'b5', 'b24', 'b18', 'b2', 'b21', 'b28', 'b9']

Section: Invariant Learning in Molecular Relational Learning
Research on invariant learning in molecular representation learning remains sparse. One approach identifies core substructures using the graph information bottleneck to extract minimal task-relevant information (Lee et al., 2023a). Another method leverages causal intervention to learn causal substructures and mitigate distribution shifts (Lee et al., 2023b). In OOD settings, generalization is typically evaluated by partitioning datasets into scenarios like "unseen solvent" or "unseen domain", where test sets exhibit specific biases. However, many studies remain confined to intra-domain frameworks, failing to capture real-world complexities. Despite successes in graph-based invariant learning (Wu et al., 2022a;Yang et al., 2022;Li et al., 2022), two key challenges persist: (1) Environmental labels for graphs are difficult to obtain, often relying on handcrafted rules that provide insufficient causal structure. (2) Invariant patterns and spurious correlations are entangled with shortcut features, complicating the identification of stable representations. Addressing these challenges is crucial for improving the robustness and generalization of MRL models across diverse molecular environments.
this section cite: ['b44', 'b16']

Section: Preliminaries
We define the uppercase letters (e.g., G) as random variables, and the blackboard typefaces (e.g., G) denote the sample spaces. Let G = (V, E) ∈ G denote a graph, where V = {v 1 , v 2 , ..., v n } is the set of nodes and E ∈ V × V is the set of edges.
this section cite: []

Section: Molecular Relational Learning.
The goal of MRL task is to predict the target label Y given the associated input molecular pairs (G 1 , G 2 ). It can be formulated as modeling the conditional distribution p(Y|G 1 , G 2 ).
Notations. Given a dataset D = {((
G i 1 , G i 2 ), Y i )} N i=1 , where G 1 ∈ G 1 is solute molecule, and G 2 ∈ G 2 is solvent molecule, each molecular pair
is associated with a target label Y. N is the total number of samples. The objective is to train a model to predict Y based on the input (G 1 , G 2 ). The model should effectively learn the relationships between the input features and the target variable, leveraging the information from both G 1 and G 2 to accurately predict Y. The model's performance will be evaluated based on the RMSE of the predicted output Ŷ in comparison to the ground truth labels Y.
this section cite: []

Section: Molecular Representation.
We implement our method based on (Pathak et al., 2020), which is a message passing architecture devised for the solute and solvent molecule interaction. Given a pair of molecules G 1 = (V 1 , E 1 ) and G 2 = (V 2 , E 2 ). We first obtain the node representation of each molecule as follows:
h 1 = GNN(V 1 , E 1 ), h 2 = GNN(V 2 , E 2 ).
To capture inter-molecular interactions at the atomic level, the interaction map is constructed as following: I = h 1 • h T 2 , where • is matrix multiplication, I ∈ R N1×N2 . Here, N 1 and N 2 denote the number of atoms in molecule G 1 and G 2 , respectively. We obtained a representation h1 ∈ R N1×D of the solvent's interaction on the solute and a representation h2 ∈ R N2×D of the solute's interaction on the solvent through a shared interaction map according to the following equations: h1 = I • h 2 , h2 = I T • h 1 . H 1 is generated by concatenating two representations h1 and h 1 , i.e., H 1 = concat[h 1 , h1 ]. The overall graph representation is obtained using a readout layer R solute (H 1 ), which set the READOUT function as Set2Set (Vinyals et al., 2015).
this section cite: ['b27', 'b38']

Section: OOD Generalization.
In this work, we mainly focus on OOD generalization in graph-level prediction tasks. Our aim is to train the model with limited labels to infer the domain distribution from unseen data in D te .
Problem formulation. Given a molecular pairs dataset,
D = {((G i 1 , G i 2 ), Y i ) N tr+te i=1
} collect from multiple environments E, which were considered as drawn independently from an identical distribution p e , i.e.,
D ID = {(G 1 , G 2 ) ∈ D | G 1 ∈ G ID G 2 ∈ G ID }.
The training and test datasets are denoted as
D tr = {((G i 1 , G i 2 ), Y i )} N tr i=1 and D te = {((G i 1 , G i 2 ), Y i )} N te i=1 . Our goal is to find an opti- mal predictor Φ: (G 1 , G 2 ) -
→ Y that performs well on all environments. Formally, the learning objectives can be formulated as:
min Φ max e∈E E ((G i 1 ,G i 2 ),Y i )∼p((G1,G2),Y|e) ℓ Φ G i 1 , G i 2 , Y i (1) Definition 3.1. (Data generation process) The OOD distribution can be sampled according to D OOD = {(G 1 , G 2 ) ∈ D | (G 1 ∈ G OOD G 2 ∈ G OOD ) (G 1 ∈ G OOD G 2 ∈ G ID ) (G 1 ∈ G ID G 2 ∈ G OOD )}. The data generation process is as follows: Let E de- note all possible environments, supp(N tr ) ⊂ supp(E), sampled train data from p((G 1 , G 2 ), Y). Out of Distribu- tion indicate that p e ((G 1 , G 2 ), Y) ̸ = p ′ e ((G 1 , G 2 ), Y), i.e., D train = {((G i 1 , G i 2 ), Y i ) N tr i=1 | e ⊂ supp(N tr )}, D test = {((G i 1 , G i 2 ), Y i ) N te i=1 | e ′ ∈ supp(E)\supp(N tr )}.
Details of the OOD dataset splitting are provided in the Appendix B.1.
this section cite: []

Section: Methodology
In this section, we introduce a Relational Invariant Learning framework designed to address Out-of-Distribution generalization (RILOOD) in solvation free energy prediction. An overview of the proposed method is provided in Fig. 2. We detail the motivations and technical aspects of the three key components in RILOOD: Mixup-enhanced Conditional Variational Modeling (Section 4.2), Multi-granularity Context-Aware Refinement (Section 4.3), and Invariant Relational Learning Mechanism (Section 4.4).
this section cite: []

Section: The Overall Framework.
In molecular relation learning, substructure identification methods are widely employed. However, these approaches often fail to account for the variability in a solute's behavior across different solvents, as solute-solvent interactions can differ significantly.
To overcome this limitation, invariant learning aims to identify stable features or patterns that remain consistent across diverse environments. This approach reduces prediction errors and minimizes dependence on environmental variations. A predictor performing well across multiple, varied environments is more likely to generalize robustly to unseen distributions. Our primary objective is to develop a model that is robust to domain shifts, ensuring the mapping from molecule pairs to labels remains stable irrespective of environmental changes. Assumption 4.1. Given a molecular pair (G 1 , G 2 ), each pair is associated with R surrounding environments. We assume the existence of invariant interaction patterns that facilitate generalizable OOD predictions across all environments.
The optimal predictor Φ(•) should satisfy the properties of Invariance and Sufficiency, as detailed in Appendix A.3.
Specifically, we further decompose Φ(•) into two key components as
Φ(•) = g • f (G 1 , G 2 ): (a) A Conditional
Variational Autoencoder (CVAE) f , which models the prior distribution of the solute representation H 1 ∼ p e (z|e) across different environments. Here, z is a low-dimensional, continuous representation of the solute in the latent space. (b) A multi-granularity context-aware learner g, which refines relational features by mapping (H 1 , H 2 ) to a context-aware representation H c , i.e., g : (H 1 , H 2 ) -→ H c . Building on Eq. 1, we reformulate the OOD generalization problem for molecular pairs as:
min g max e∈E E (G i 1 ,G i 2 ,Y i )∼p(G1,G2,Y|e) ℓ g • f G i 1 , G i 2 , Y i
(2) where e denotes the support environments, and ℓ(•, •) represents the loss function.
this section cite: []

Section: Mixup-enhanced Conditional Variational Modeling
Empirically, acquiring explicit environmental labels for solute-solvent pairs is often impractical, presenting a major obstacle to learning solute representations that generalize across diverse solvent conditions. This challenge is further compounded by limited data coverage, which may fail to capture distributional shifts induced by variations in solvation environments. Fundamentally, the difficulty in obtaining environmental labels arises from the fact that solvation energy is governed by complex, multi-scale interactions between solutes and solvents, including non-covalent intermolecular forces and functional group-specific dependencies. In the absence of explicit labels, these underlying physicochemical mechanisms are difficult to accurately model or disentangle. To address this challenge, ZIN (Lin et al., 2022) introduces a method for inferring latent environmental partitions using auxiliary information. Inspired by this approach, we propose conditioning on auxiliary information to implicitly capture solute representations across the environment. adopt mixup, which generates interpolated representations to enhance robustness across environments. We propose Mixup-enhanced CVAE (MCVAE), a module designed to model molecular distributions using paired solvent information and infer the latent distribution q ϕ (z| H1 , e) across diverse environments. Furthermore, we introduce uncertainty constraints to regularize the latent space, thereby improving stability and generalization across varying solvent conditions.
this section cite: ['b18']

Section: Mixup Enhanced. To further improve generalization, we
We assume that solvents belong to R discrete categories, denoted as E = {e r } R r=1 , where each solvent type e r is represented as a R-dimensional one-hot vector e r ∈ {0, 1} R , with its r-th dimension set to 1. To enhance generalization, we employ mixup augmentation to interpolate between environmental conditions, allowing the model to learn continuous latent representations rather than relying solely on discrete labels. This encourages better adaptation to unseen domains.
Given the molecular representations H 1 and H 2 for molecules G 1 and G 2 , respectively, we construct enhanced samples using the following mixup formulation:
H1 = λ • H 1 + (1 -λ) • H 2 , e = λ • e 1 + (1 -λ) • e 2 (3)
where H1 denotes the mixed molecular representation and e signifies the interpolated solvent condition. The mixing coefficient λ, is sampled from a Beta distribution, specifically λ ∼ Beta(α, α), to ensure a smooth interpolation between different environments. This modeling strategy can be readily extended to other contexts, such as scaffold-based modeling. The derivations for the upper and lower bounds of this mixed representation are detailed in Appendix A.1.
For the regression task, we incorporate an uncertainty constraint to mitigate noise introduced by the mixup technique.
Here, σ(•) 2 is the uncertainty variance, and we constrain it. Consequently, the regression loss can be reformulated as follows:
L reg = 1 N N i=1 1 σ(H i 1 ) 2 ∥y i -Φ(G i 1 , G i 2 )∥ 2 + log σ(H i 1 ) 2(4)
this section cite: []

Section: Variational Inference for Mixup Representations.
To model the conditional log-likelihood log p( H1 | e), we introduce variational inference, reformulating the objective as a variational lower bound by approximating the posterior distribution q(z | H1 , e):
max θ,ϕ E H1∼D E q ϕ (z| H1,e) log p θ ( H1 | z, e) , s.t. D KL q ϕ (z | H1 , e) ∥ p θ (z | H1 ) < δ(5)
where δ is a threshold, and the KL divergence constraint ensures that the approximate posterior remains close to the prior, preventing latent space collapse and enhancing generalization across diverse solvent conditions.
In practice, as the number of solvent categories grows, independent modeling becomes computationally prohibitive. To enhance scalability, a learnable function could be employed to assign soft environment labels to solvents, rather than modeling each solvent separately. However, given the limited solvent diversity in our dataset (e.g., only five solvents in QM9Solv dataset), we refrain from adopting this approach to ensure a fair comparison.
Environment Inference via MCVAE Optimization. By optimizing MCVAE, we aim to infer the solute distribution within a latent environment, offering a novel strategy for learning environment-aware molecular representations.
Specifically, we minimize the divergence between the approximate posterior distribution q ϕ (z | H1 , e) of the latent variable z and the true posterior probability p ϕ ( H1 | e, z) for a specific environment e. Leveraging the rich prior knowledge embedded in conditional encoders, our training objective comprises two main components: (1) Encourage the R-group latent distribution produced by the encoder to approximate a standard normal distribution as closely as possible. (2) Sample z from the conditional distribution q ϕ (z | e), and ensure that the reconstructed solute molecular features closely resemble the original features.
L MCVAE (θ, ϕ; H1 , e) = -KL q ϕ (z | H1 , e)∥p θ (z | H1 ) + 1 N N i=1 1 σ( H1 ) 2 ∥z -H1 ∥ 2 + log σ( H1 ) 2(6)
where z = g ϕ H1 , e, ϵ , with ϵ ∼ N (0, I), and I is the identity matrix. Detailed proof is in Appendix A.2.
this section cite: []

Section: Multi-granularity Context-Aware Refinement.
Although the solvent type serves as a useful proxy for environmental context, directly relying on it can lead to predictive shortcuts. To address this, we propose a joint optimization framework that simultaneously learns: (i) an environment modeling function from auxiliary information, (ii) an interaction-aware feature extractor, and (iii) mutual information constraints. The goal is to encode a context-aware molecular representation that dynamically captures solute behavior under specific conditions while accurately modeling the context-dependent relevance of solute molecules.
Previous studies have predominantly focused on atomiclevel interactions within molecules (Pathak et al., 2020). However, since solvents interact globally with solutes, a multi-granularity interaction strategy is better suited to capture their complex effects. Notably, solute-solvent interactions are non-covalent and are often neglected in conventional modeling approaches. To address this limitation, we employ a self-attention mechanism to dynamically identify intermolecular interactions, allowing representations to adapt more effectively across diverse environments.
this section cite: ['b27']

Section: Multi-granularity Interactions with Context.
To effectively model solute-solvent interactions, we propose a Multigranularity Context-Aware Refinement (MCAR) strategy, designed to capture hierarchical interaction patterns across multiple levels. The MCAR mechanism is implemented in two key steps: (1) Context learning: Both coarse-grained molecular-level contexts and fine-grained feature-level contexts are simultaneously captured to jointly learn comprehensive contextual information. (2) Pattern refinement: Invariant interaction patterns are refined through matrix multiplication between the coarse-grained and fine-grained feature representations.
We begin by constructing an initial embedding E that aggregates relevant molecular features: E = concat[z, H 2 ], where z and H 2 represent specific molecular features. To model global interactions, we project E into query (Q), key (K), and value (V) matrices using learnable linear transformations:
Q, K, V = EW Q , EW K , EW V(7)
where W Q , W K , W V are trainable projection matrices. A scaled dot-product self-attention mechanism is then applied to these matrices to capture long-range dependencies across molecular entities:
O c = Attention(Q, K, V) = W c • Softmax QK T √ d k V (8)
where W c is learnable transformations that enhance attention expressiveness. In parallel, local intra-molecular interactions are captured via a non-linear transformation:
O f = PReLU(W L E + b L )(9)
where W L and b L are learnable parameters for local feature transformation. The PReLU is activation function to enhance the expressiveness of the local representations. Finally, the context-aware representation is obtained by performing a Hadamard product between the global and local interaction features:
H c = O c • O f (10
)
where • denotes the Hadamard product, enabling the integration of multi-scale contextual information into a unified and expressive feature representation.
To improve feature extraction, we maximize mutual information to retain essential features while reducing redundancy and noise. Specifically, we optimize the mutual information between the solute representation H 1 and the context-aware feature H c . The solute feature H 1 , may be influenced by spurious correlations that hinder generalization. In contrast, the context-aware feature H c captures invariant and meaningful correlations, leading to more robust representations across diverse environments. We therefore denote H c as H inv to emphasize its role in learning invariant features. To this end, we formulate the optimization objective as follows:
max fc,w I H inv ; Y , s.t. H inv ∈ arg max Hinv=w(H1),| Hinv|≤H1 I H inv ; H 1 | Y(11)
Finally, contrastive learning provides a practical solution for the approximation, the learning objective is defined as:
L M I = - 1 M M i=1 log exp(sim( Ĥi inv , H i 1 )) exp(sim( Ĥi inv , H i 1 )) + M j=1,j̸ =i exp(sim( Ĥj inv , H j 1 ))(12
) 4.4. Invariant Relational Learning Mechanism Optimization Objective. Eq. 2 clarifies the training objective of OOD generalization. However, directly optimizing Eq. 2 is not impracticable. Instead, we formulate a joint optimization framework:
L = L inv + αL MCVAE + βL MI (13
)
where α and β are weight hyperparameters for L MCVAE and L MI , respectively. The term L inv represents the prediction loss, measuring the discrepancy between the model's output and the ground truth. For regression tasks, Eq. 4 can be used instead of L inv .
Proposition 4.2. Given the auxiliary environment e, our goal is to build a model p θ ( H1 | e, z) that learns the feature H1 ∈ R conditioned on e. Optimizing Eq. 6 ensures that z exhibits sufficient predictive power, thereby allowing the model to satisfy the Sufficient condition in Assumption 4.1. Furthermore, minimizing Eq. 13 encourages the model to satisfy the Invariance condition in Assumption 4.1.
this section cite: []

Section: Experiments
In this section, we conduct extensive experiments to answer the research questions:
• RQ1: How to evaluate the effectiveness of the model in OOD scenarios?
• RQ2: How effective is RILOOD in discovering invariant features and improving generalization?
this section cite: []

Section: Experimental Settings
Datasets. We use six datasets to evaluate our method. Specifically, the Minnesota Solvation Database (MN-Solv) (Marenich et al., 2012), QM9Solv (Ward et al., 2021), CompSolv (Moine et al., 2017), CombiSolv (Vermeire & Green, 2021), MolMerger (Ramani & Karmakar, 2024), and Abraham (Grubbs et al., 2010). The detailed statistics and descriptions are given in Appendix B. More experiments are provided in Appendix C.2.
Baselines. For a comprehensive comparison, we adopt two types of baselines: (1) There is no interaction layer between molecular encoders. We use three commonly used GNN models, including GIN (Xu et al., 2018), GCN (Kipf & Welling, 2016), GAT (Veličković et al., 2017), to obtain molecular embedding through concatenation and then enter the prediction layer; (2) there is an interaction layer between molecular encoders, including ERM (Vapnik, 2013), Group-DRO (Sagawa et al., 2019), MixUp (Zhang et al., 2017a), MolMerger (Ramani & Karmakar, 2024), CIGIN (Pathak et al., 2020), CGIB (Lee et al., 2023a), CMRL (Lee et al., 2023b).
Metrics. We choose widely-used metrics in previous works, the performance of the molecular interaction prediction task is evaluated in terms of RMSE (Pathak et al., 2020). Lower error indicate better prediction performance. AUROC (Lee et al., 2023b) for DDI prediction.
this section cite: ['b22', 'b39', 'b25', 'b37', 'b29', 'b8', 'b43', 'b11', 'b36', 'b34', 'b31', 'b29', 'b27', 'b27']

Section: Main Results (RQ1)
Real-world Dataset. To assess the generalization performance of our method, we conducted comprehensive experiments on three datasets, demonstrating the effectiveness of the proposed approach. To further explore distribution shifts across diverse environments, we evaluated model performance under two distinct settings: Solvent and Scaffold.
The overall results, summarized in Tab. 1, lead to the following key observations:
Our method consistently outperforms baseline models, achieving superior results across all datasets. Traditional approaches exhibit limitations, as they primarily rely on substructure-based invariance, which often introduces spurious correlations in MRL. The notable improvement observed in RILOOD stems from its ability to capture multigranularity interactions and identify invariant patterns, enabling the model to effectively adapt to domain shifts. Further discussions on extending this method to the I.I.D. setting are provided in Appendix C.2, with additional results in Tab. 4.
this section cite: []

Section: Synthetic Dataset.
To evaluate model robustness under distribution shifts, we apply dataset-specific shift strategies, introducing spurious features to construct synthetic datasets. Following (Li et al., 2022;Wu et al., 2022b), spurious correlations are injected by controlling the variant distribution. Further details are provided in Appendix C.2. Specifically, we manually introduce spurious relationships of varying degrees between environment e and the label Y in the training set, setting degree d = {0.25, 0.33, 0.5, 0.75}. The results, presented in Fig. 3 (a), indicate that as d increases, performance generally improves due to a greater degree of distribution shift. However, our proposed method exhibits the highest stability, effectively mitigating the effects of spurious correlations.
this section cite: ['b16']

Section: Generalization on Graph Classification.
To evaluate the applicability of our method to molecular pair data and classification tasks, we conducted experiments on the DDI dataset.  As illustrated in Fig. 3(b), RILOOD consistently outperforms existing approaches under OOD conditions. This performance gain can be attributed to RILOOD's enhanced generalization capability, which facilitates effective knowledge transfer from known molecular interactions to structurally similar compounds and previously unseen scaffolds. Such transferability improves the model's robustness to distributional shifts, thereby ensuring adaptability across diverse molecular structures.
this section cite: []

Section: In-depth Analysis (RQ2)
To assess the contribution of each module, we conduct an ablation study by removing specific components: Multigranularity Context-Aware Refinement (MCAR) trained on the downstream task (M); mutual information loss L MI (Mi); conditional distribution modeling loss L MCVAE (MC); and MCAR removal but all loss is used (w/o MCAR). The complete model is jointly trained using Eq. 13 (Ours). The results are summarized in Tab. 2. From Tab. 2, we observe the following: (1) Incorporating MCAR improves baseline performance, highlighting the importance of context-aware interactions in enhancing model robustness. (2) While conditional modeling significantly affects performance, the individual contributions of L MCVAE and L MI are smaller compared to joint training. (3) Removing MCAR leads to a performance drop; however, the model still surpasses the baseline due to the co-optimization of all losses.
Feature Visualization. To evaluate the effectiveness of MCAR, we used t-SNE to visualize molecular interactions in the best performing model and compare them to baselines. As shown in Fig. 4, (1) The solute in the test set originated from different distributions than the training set, demonstrating the distribution shift; (2) MCAR enhances feature diversity, improving molecule interaction modeling; and (3) MCAR captures domain-invariant features, boosting generalization to unseen domains. These results confirm our  method's robustness against distribution shifts.
this section cite: []

Section: Conclusion
In this work, we propose a Relational Invariant Learning framework to solve out-of-distribution in solvation free energy prediction. Three tailored modules are jointly optimized to train the model and learn the representation of invariant molecules in diverse environments. Mixup enhanced molecular representations are used for variational modeling of diverse environments, further capturing invariant interaction patterns through multi-granularity context-aware refinement strategy. Extensive experiments and theoretical analysis prove the superiority of our method.
this section cite: []

Section: References
Ref_id:b0 Title: The properties of known drugs. 1. molecular frameworks Year: (1996)
Ref_id:b1 Title: Machine learning with physicochemical relationships: solubility prediction in organic solvents and water Year: (2020)
Ref_id:b2 Title: Learning causally invariant representations for out-of-distribution generalization on graphs Year: (2022)
Ref_id:b3 Title: Group contribution and machine learning approaches to predict abraham solute parameters, solvation free energy, and solvation enthalpy Year: (2022)
Ref_id:b4 Title: A universal approach to solvation modeling Year: (2008)
Ref_id:b5 Title: Causal inference without counterfactuals Year: (2000)
Ref_id:b6 Title: Esol: estimating aqueous solubility directly from molecular structure Year: (2004)
Ref_id:b7 Title: Mmgnn: A molecular merged graph neural network for explainable solvation free energy prediction Year: (2024)
Ref_id:b8 Title: Mathematical correlations for describing solute transfer into functionalized alkane solvents containing hydroxyl, ether, ester or ketone solvents Year: (2010)
Ref_id:b9 Title: Joint learning of label and environment causal independence for graph out-of-distribution generalization Year: (2024)
Ref_id:b10 Title: Graph invariant learning with subgraph co-mixup for out-of-distribution generalization Year: (2024)
Ref_id:b11 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b12 Title: Solubility dataset of compounds in organic solvents and water in a wide range of temperatures Year: (2023)
Ref_id:b13 Title: Outof-distribution generalization via risk extrapolation (rex) Year: (2021)
Ref_id:b14 Title: Conditional graph information bottleneck for molecular relational learning Year: (2023)
Ref_id:b15 Title: Shiftrobust molecular relational learning with causal substructure Year: (2023)
Ref_id:b16 Title: Learning invariant graph representations for out-of-distribution generalization Year: (2022)
Ref_id:b17 Title: Delfos: deep learning model for prediction of solvation free energies in generic organic solvents Year: (2019)
Ref_id:b18 Title: Zin: When and how to learn invariance without environment partition? Year: (2022)
Ref_id:b19 Title: Learning causal semantic representation for out-of-distribution prediction Year: (2021)
Ref_id:b20 Title: Explainable solvation free energy prediction combining graph neural networks with chemical intuition Year: (2022)
Ref_id:b21 Title: Invariant causal representation learning for out-ofdistribution generalization Year: (2021)
Ref_id:b22 Title: Minnesota solvation databaseversion Year: (2012)
Ref_id:b23 Title: Biosnap datasets: Stanford biomedical network dataset collection Year: (2018)
Ref_id:b24 Title: Disentangled multiplex graph representation learning Year: (2023)
Ref_id:b25 Title: Estimation of solvation quantities from experimental thermodynamic data: Development of the comprehensive compsol databank for pure and mixed solutes Year: (2017)
Ref_id:b26 Title: Comprehensive modelling of pharmaceutical solvation energy in different solvents Year: (2021)
Ref_id:b27 Title: Chemically interpretable graph interaction network for prediction of pharmacokinetic properties of drug-like molecules Year: (2020)
Ref_id:b28 Title: Elements of causal inference: foundations and learning algorithms Year: (2017)
Ref_id:b29 Title: Graph neural networks for predicting solubility in diverse solvents using molmerger incorporating solute-solvent interactions Year: (2024)
Ref_id:b30 Title: Deep learning improves prediction of drug-drug and drug-food interactions Year: (2018)
Ref_id:b31 Title: Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization Year: (2019)
Ref_id:b32 Title: Multisolvent models for solvation free energy predictions using 3d-rism hydration thermodynamic descriptors Year: (2020)
Ref_id:b33 Title: Unleashing the power of graph data augmentation on covariate distribution shift Year: (2024)
Ref_id:b34 Title: The nature of statistical learning theory Year: (2013)
Ref_id:b35 Title: Origins of complex solvent effects on chemical reactivity and computational tools to investigate them: a review Year: (2019)
Ref_id:b36 Title: Graph attention networks Year: (2017)
Ref_id:b37 Title: Transfer learning for solvation free energies: From quantum chemistry to experiments Year: (2021)
Ref_id:b38 Title: Order matters: Sequence to sequence for sets Year: (2015)
Ref_id:b39 Title: Graphbased approaches for predicting solvation energy in multiple solvents: open datasets and machine learning models Year: (2021)
Ref_id:b40 Title: Handling distribution shifts on graphs: An invariance perspective Year: (2022)
Ref_id:b41 Title: Discovering invariant rationales for graph neural networks Year: (2022)
Ref_id:b42 Title: Multilevel attention network with semi-supervised domain adaptation for drug-target prediction Year: (2024)
Ref_id:b43 Title: How powerful are graph neural networks? arXiv preprint Year: (2018)
Ref_id:b44 Title: Learning substructure invariance for out-of-distribution molecular representations Year: (2022)
Ref_id:b45 Title: Accurate prediction of aqueous free solvation energies using 3d atomic featurebased graph neural network with transfer learning Year: (2022)
Ref_id:b46 Title: mixup: Beyond empirical risk minimization Year: (2017)
Ref_id:b47 Title: Predicting potential drug-drug interactions by integrating chemical, biological, phenotypic and network data Year: (2017)
