Title: FLEXOLMO: Open Language Models for Flexible Data Use
Abstract: We introduce FLEXOLMO, a new class of language models (LMs) that supports (1) distributed training without data sharing, where different model parameters are independently trained on closed datasets, and (2) data-flexible inference, where these parameters along with their associated data can be flexibly included or excluded from model inferences with no further training. FLEXOLMO employs a mixture-of-experts (MoE) architecture where each expert is trained independently on closed datasets and later integrated through a new domain-informed routing without any joint training. FLEXOLMO is trained on FLEXMIX, a corpus we curate comprising publicly available datasets alongside seven domain-specific sets, representing realistic approximations of closed sets. We evaluate models with up to 37 billion parameters (20 billion active) on 31 diverse downstream tasks. We show that a general expert trained on public data can be effectively combined with independently trained experts from other data owners, leading to an average 41% relative improvement while allowing users to opt out of certain data based on data licensing or permission requirements. Our approach also outperforms prior model merging methods by 10.1% on average and surpasses the standard MoE trained without data restrictions using the same training FLOPs. Altogether, this research presents a solution for both data owners and researchers in regulated industries with sensitive or protected data. FLEXOLMO enables benefiting from closed data while respecting data owners' preferences by keeping their data local and supporting fine-grained control of data access during inference.

Section: Introduction
Pretraining language models (LMs) typically requires centralized access to all data during training and does not have any mechanism to track or control the influence of specific data points on model parameters. Model developers must therefore make a one-time decision on which data sources to Figure 1: An overview of FLEXOLMO. Data owners can contribute without sharing the data by training their own expert modules (FFNs and router embeddings) with a shared public model as an anchor point. At inference, these modules are integrated into a MoE model via a novel router embedding concatenation. This design enables flexible inclusion or exclusion of experts and strict opt-out guarantees, e.g., Github data can be excluded at no cost (blurred) during inference.
include, with limited ability to remove the effect of certain data after training [1,2,3]. Moreover, this centralized approach precludes the use of closed data that data owners cannot share with model developers for confidentiality, regulatory, or other reasons. Although solutions have been proposed to allow training without sharing the data, such as federated learning [4,5], their practical adoption remains limited due to performance degradation and the high cost of synchronized training [6,7].
We introduce FLEXOLMO, a new class of LMs that enables distributed training on locally maintained datasets while enabling flexible opt-in and opt-out during inference. FLEXOLMO (Figure 1) employs a mixture-of-experts (MoE) architecture [8,9], where each expert is trained independently on closed datasets and later integrated into an MoE. This design allows data owners to contribute asynchronously without sharing their data, while also enabling continual updates with new data and providing strong guarantees for data opt-out during inference. Our approach can be seen as an instance of model merging [10], which merges different models into a unified one [11,12]. However, our model is designed to address the unique challenges in our problem setup-combining models pre-trained on completely disjoint datasets with different distributions-which makes prior model merging techniques like ensembling output probabilities [11] or merging model weights [12] suboptimal.
A key challenge in training FLEXOLMO is ensuring the merging of independently trained experts without joint training. We introduce a training algorithm where each data owner independently trains an expert module using the frozen public model as a shared anchor (Figure 1). This approach teaches independently trained experts to coordinate with the same public model and, by extension, with each other. Additionally, the router, a module that determines which experts process each token, typically requires joint training. We address this by assigning each expert a router embedding, initialized from its domain embedding using an off-the-shelf embedder [13] and further finetuned on its corresponding data during individual expert training. These embeddings are then concatenated to form the router during merging, removing the need for joint training.
To validate FLEXOLMO, we curate a data mixture called FLEXMIX, which includes a public training set along with seven domain-specific sets (e.g., news, educational text, and Reddit). These domains are chosen to simulate scenarios where high-quality data that can benefit LM training is not publicly available.
We train FLEXOLMO first on public data, then extend it by merging expert modules trained independently on our simulated closed sets. While continued pretraining on these sets improves some downstream tasks, it suffers from catastrophic forgetting and inconsistent performance. In contrast, FLEXOLMO improves upon the public model by 41% and also outperforms prior merging techniques such as model soup and ensembling by 10.1% across 31 downstream tasks. We observe the largest improvements on tasks related to closed sets. Notably, even on benchmarks where no individual closed set improved performance over the public model, combining multiple experts yielded significant gains, demonstrating synergies among independently trained modules. Our qualitative analysis demonstrates that sparse expert activation across layers through the MoE architecture is key to these gains. Our qualitative analysis shows that the MoE architecture's ability to selectively activate different experts per layer per token is crucial to these gains by combining the strengths of each specialized expert. We hope our work enables research with a broader range of closed datasets for LM training, particularly for organizations interested in collaborating on scientific research through the new features that FlexOlmo provides.
2 Background & Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b10', 'b11', 'b12']

Section: Background: Data Restrictions
The standard LM training practice requires model developers to aggregate all data centrally and make a one-time decision on which data source to include and exclude. But many real-world data come with sharing and usage restrictions and necessitates (1) model training without data pooling and (2) model inference that can flexibly select different data sources based on use case and access privileges.
this section cite: []

Section: Data Sharing Constraints
Organizations in regulated industries require LMs that can leverage their closed datasets while maintaining strict data privacy and access controls. Healthcare institutions, financial firms, and other entities possess valuable domain-specific data but cannot share it externally due to HIPAA, GDPR [14,15], data sovereignty laws [16], and intellectual property (IP) protections. These organizations need training paradigms that enable AI improvement on their sensitive data while ensuring such sensitive data never leaves certain environments and can be removed from the model after training, e.g., when data usage rights expire. In such settings, modular training approaches, where individual experts are trained independently and asynchronously on locally maintained data, are essential.
this section cite: ['b13', 'b14', 'b15']

Section: Data Use Constraints
The inclusion of certain data depends on specific use cases and end users. Privileged access: User-facing applications often involve closed data restricted to specific, authorized users [17]. For example, GitHub Copilot must tailor code suggestions to reflect internal repositories based on an engineer's role and access rights [18]. Copyright and data consent: Legal and ethical considerations on training data for AI are evolving and uncertain [19,20,21,22,23,24], and often depend on the data's intended use, e.g., licenses may prohibit commercial use or limit certain query types [25,26]. Model control: Training data often include sensitive content [27,28,29] which may be beneficial in certain contexts but harmful in others. For instance, one may want to activate the use of toxic content for toxicity classification in a research setting, but deactivate it in applications presented to a general audience.
this section cite: ['b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28']

Section: Related Work
Federated Learning Federated Learning (FL) trains a single model over distributed datasets by synchronously aggregating client updates [5,4,30]. FL methods range from classical approaches that iteratively aggregate parameter updates from local clients [5,31] to parameter-efficient techniques which have been adapted for LMs [32,33,34]. FL can guarantee data privacy using techniques such as homomorphic encryption [35] and differential privacy (DP) [36]. However, FL has seen limited adoption in LM training due to the high cost of synchronization and performance degradation [6,7], and remains susceptible to privacy attacks due to inter-client communication [37,38]. Similar to [39], our approach avoids data sharing but differs fundamentally by supporting independent, asynchronous training without costly inter-client communication, and allowing real-time opt-in and opt-out. Like FL, our model allows data owners to optionally apply DP training locally for privacy guarantees. Because DP is orthogonal to our architecture, each contributor can independently choose whether to apply it, providing flexibility without compromising the overall design.
this section cite: ['b4', 'b3', 'b29', 'b4', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b5', 'b6', 'b36', 'b37', 'b38']

Section: Model Merging
Our work builds on recent efforts [40,10] that advocate for developing machine learning models like open-source softwares, where sub-parts of the model can be trained independently and subsequently merged into unified systems. This can be achieved through various methods, including weight merging, output ensembling, and expert routing. Model soup-merging model weights trained on different datasets from the same initialization-can boost performance [41,12,42,43,44], especially with weighted combinations [45,46,47,48,49,50,51]. Weighted output ensembling (e.g., BTM [52,53]) is also effective when models are trained on distinct datasets initialized from the same seed model. These approaches can be applied to our setting, where each expert is independently trained starting from the same public model then merged into a unified one. Our experiments ( §5) show that these methods are less effective, primarily because they lack learned connections between different modules, which constrains the expressivity of the resulting models.
An alternative line of work focuses on expert routing methods, such as DEMix [54], BTX [55] and its extensions [56,57,58], which merge dense, independently trained experts into a mixture-of-experts (MoE) framework. We draw inspiration from this work, as we also integrate independently trained models into a MoE. However, these methods require joint training on a union of all datasets used in expert training after merging. By contrast, FLEXOLMO removes the need for joint data access to enable training on locally maintained datasets. Our work is also related to ModuleFormer [59], which induces sparse modularity from uncurated data using novel load balancing and concentration losses.
Related efforts in parameter-efficient training have explored merging LoRA adapter weights trained on separate datasets [60,61,62,63], particularly to reduce communication overhead in collaborative settings and support fine-grained data access control and opt-out use cases [64,65]. Unlike these methods, which focus on merging lightweight adapters, our approach merges full expert models into a standard MoE architecture.
Mixture-of-Experts (MoE) MoE models [8,9,66], consisting of many small feedforward networks called experts, have gained popularity for their training and inference efficiency. Our work leverages the MoE architecture; however, our motivation and training method are fundamentally different as our primarily goal is to support modularity rather than efficiency.
this section cite: ['b39', 'b9', 'b40', 'b11', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b7', 'b8', 'b65']

Section: FLEXOLMO: LMs with Flexible Data Use

this section cite: []

Section: Problem Setup
Let M pub be a model trained on a publicly available dataset D pub , and D = {D 1 , D 2 , ..., D n } represent a collection of locally maintained datasets with separate owners. Our objective is a single model M final , which is constructed via composing M pub and a set of modules {M 1 , M 2 , . . . , M n }, where each M i is independently trained by the owner of D i , who also has access to M pub .
This model satisfies two requirements: (1) training M final does not require anyone to have joint access to the full dataset collection D, as each M i is trained independently by the owner of dataset D i ; (2) removing any module M i from M final guarantees complete removal of its associated data D i .
The key modeling challenges are: (1) to develop an algorithm that creates M i using D i and M pub , and (2) to design the merging algorithm that combines M pub , M 1 , M 2 , . . . , M n into M final .
this section cite: ['b0']

Section: Model Architecture
FLEXOLMO follows the standard MoE architecture: it replaces the feedforward network (FFN) in each transformer block with a router and n small FFNs called expert modules {M pub , M 1 , ..., M n }. Note that we omit the layer index for each expert in our notation for simplicity. Given a processed input token embedding x ∈ R h , the MoE module computes output representation y:
y = i∈Topk(r(x)) softmax(r(x) i )M i (x),
where the router function r computes the expert probabilities from x. Unlike standard MoEs where experts are trained jointly, our experts are trained asynchronously on distinct datasets {D 1 , ..., D n }.
this section cite: []

Section: Training Algorithm
Standard MoEs train all experts and the router jointly on all data. In contrast, FLEXOLMO trains experts independently by teaching them to coordinate ( §3.3.1) and merges them at inference using a domain-informed router ( §3.3.2). Optional router tuning can further improve performance ( §3.3.3).
this section cite: []

Section: Training Experts to Coordinate
A straightforward way to train each expert would be to directly continue to train each expert M i on its own data D i [52]. We found that this method causes the experts to diverge too much from one another and from the original seed model, which makes merging after isolated training difficult.
To prevent such divergence, we train experts independently while teaching them to coordinate (Figure 1). We use M pub as an anchor that teaches experts to coordinate with M pub and, by extension, with each other. Specifically, during training, for dataset D i , we construct a MoE model with two expert modules-both initialized from the same FFNs from M pub . During training, we freeze M pub expert and the shared attention layer, while the other expert (M i ) is trained on D i . As each data owner updates only their own FFNs while keeping all other parameters (those inherited from M pub such as attention layer) frozen, the learned FFNs are designed to naturally coordinate with each other later during merging at inference time. Importantly, with this approach, a router is learned so that each expert can be integrated into a MoE architecture without additional training (details in §3.3.2).
this section cite: ['b51']

Section: Domain-Informed Router
The router plays a critical role in MoE: the router function r maps an input vector x to a distribution over expert modules, including the public model as one of the experts: r(x) = W r x, W r ∈ R (n+1)×h In typical MoEs, W r is trained end-to-end alongside all expert modules, using access to the full training dataset. Instead, we decompose W r into individual expert-specific router embeddings, where each row r i represents the router embedding for expert M i , learned only from D i :
W r =     r pub r 1 . . . r n     , where r i = 1 |S i | d k ∈Si E(d k ) ∈ R h , S i ⊂ D i .
These router embeddings can be initialized by averaging domain-specific embeddings of samples from each D i , obtained by encoding subsets of data using an off-the-shelf embedder E [67] that maps a document into an h-dimensional vector. This method is motivated by prior model merging work that leverages domain embeddings for routing [68,42,69,70,71,72,73].
During coordinated training of experts ( §3.3.1), we learn the router embeddings in pairs: [r pub , r i ] The public embedding r pub remains frozen across all experts, while r i is finetuned separately alongside the parameters of M i . At inference time, merging the expert embeddings into the complete router matrix W r directly integrates all expert modules into one unified MoE. Furthermore, experts can be flexibly added or removed by simply adding or removing their corresponding router embedding.
Adding a Bias Term Unlike standard router learning that is learned among all experts jointly, coordinated training of experts only learns pairwise routing decisions between one expert and the public model. This means the model never directly compares experts M 1 and M 2 during training, potentially limiting generalization during inference. To alleviate this issue, we add a negative bias term b i for each independent trained expert {M 1 , M 2 , . . . , M n }. We select expert M i when:
r i • x + b i > r pub • x ∀i ∈ {1, 2, ..., n}
Otherwise default to M pub . This helps the later merging process, where each expert competes not just with the public model but with all other experts. Further details and justifications are provided in §D.
this section cite: ['b66', 'b67', 'b41', 'b68', 'b69', 'b70', 'b71', 'b72']

Section: Optional Router Training on Proxy Data
With our proposed model design, expert modules can be merged without any additional training. However, if data owners are willing to identify proxy samples within the public dataset M pub that resemble their closed data, we can optionally perform a lightweight router tuning step after merging, using only public data from D pub . Specifically, we assume each data owner selects a small proxy set Di ⊆ D pub , where | Di | ≪ 0.01 × |D i |, chosen to approximate the distribution of their closed dataset D i . While Di is too small to train expert modules, it still provides useful signals for improving router quality. To construct Di , we train a binary classifier to distinguish D i from D pub and select public samples with the highest predicted likelihood of belonging to D i . After merging, we tune the router embeddings r 1 , • • • , r n , r pub on the combined set D1 , • • • , Dn , and D pub , sampled uniformly.
this section cite: []

Section: Experimental Setup

this section cite: []

Section: Training Data: FLEXMIX
Our corpus comprises a single Public Mix and seven closed sets, either real or simulated, which are designed to be disjoint from each other. Figure 5 in §B provides the statistics.
• Public Mix represents general web text based on Common Crawl (CC) 1 . Specifically, we took the Baseline version of DCLM [74], excluding news and creative writing content (described below). This represents a public dataset that can be used without restrictions.
• News includes news content from DCLM-Baseline, obtained by applying the classifier from [75] and selecting documents classified as News Articles. While included in CC when downloaded, many of the original sources are subject to closed access [20].
• Creative Writing includes creative content from DCLM-Baseline, obtained by applying the classifier from [75] and selecting documents classified as Creative Writing.
• Code includes code repositories from Starcoder [76,77] with additional quality filtering as in [78].
• Academic includes open-access academic papers obtained from [79]; these are papers from [80,81] but re-processed using olmOCR [79] for cleaner plain text.
• Educational Text includes educational text from digitized PDFs, converted to plain text using olmOCR [79].
• Math includes math-relevant content, including web pages about or using math and math problem sets, obtained by combining Dolmino Math Mix [78] and FineMath4+ [82].
• Reddit contains posts and comments originally sourced and released by Dolma [83], further filtered and processed to improve quality (details in Appendix B). As of this writing, this Reddit data is no longer unrestrictedly downloadable due to Reddit's 2023 policy change. 2These seven sets are designed to represent datasets with at least one of the following characteristics:
(1) historically closed and not publicly available; (2) previously publicly available but now closed; or (3) domains with scarce high-quality public data.
this section cite: ['b73', 'b74', 'b19', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80', 'b78', 'b78', 'b77', 'b81', 'b82']

Section: Evaluation
We evaluate our models and baselines on a large and diverse collection of well-established benchmarks, consisting of 31 tasks across 10 categories, broadly grouped into (1) general-purpose LM benchmarks and (2) domain-specific evaluations. More details are provided in §C.
this section cite: []

Section: General-purpose Evaluation
We report results on (1) MC9, nine multiple-choice datasets including ARC-Easy [84], ARC-Challenge [84], BoolQ [85], CSQA [86], HellaSwag [87], Open-BookQA [88], PIQA [89], SocialIQa [90], and WinoGrande [91], (2) GEN5, five generative tasks including CoQA [92], SQuAD [93], Natural Questions [94], TriviaQA [95], and DROP [96], as well as (3) MMLU [97], (4) MMLU-Pro [98], (5) AGIEval [99] consisting of 20 tasks from college admission tasks, and (6) BBH [100] consisting of 23 challenging BIG-Bench tasks.
this section cite: ['b83', 'b83', 'b84', 'b85', 'b86', 'b87', 'b88', 'b89', 'b90', 'b91', 'b92', 'b93', 'b94', 'b95', 'b96', 'b97']

Section: Domain-specific Evaluation
While general-purpose evaluation benchmarks already include some math assessment, we further evaluate math ability using (6) Math2, which encompasses two specialized math benchmarks: GSM8K [101] and MATH [102]. To evaluate coding capabilities, we use (7) Code4, 4 coding benchmarks including MBPP [103], MBPPPLUS [104], HUMANEVAL [105], and HUMANEVALPLUS [104]. To measure scientific literature understanding, we report on (8) SciRIFF5: comprising 5 subtasks from SciRIFF [106]. Finally, we include (9) NewsG: news generation and (10) PoemG: poem generation tasks, both evaluated using an LM judge.
this section cite: ['b100', 'b101', 'b102', 'b103', 'b104', 'b103', 'b105']

Section: Baselines
We compare our method against several baselines, either taken directly from prior work or minimally adapted to our problem setting. All baselines, except for 'Unrestricted training,' train a set of dense models independently by continuing pretraining from the public model M pub on each simulated closed set, without architectural changes, and merge them using model merging techniques.
Prompt-based Routing We use an LM-based domain classifier via prompting to route each query to the most suitable model, which is then used exclusively. We use Llama-3.1-8B-Instruct [107] and OLMo-2-1124-7B-Instruct [78] as classifiers. More details are in §A.1.
Model Soup We apply both average and weighted parameter averaging across all models, following [12]. The weights are derived by applying a softmax over the log-likelihoods of each model on the test example input.
this section cite: ['b106', 'b77', 'b11']

Section: Branch-Train-Merge (BTM)
We follow BTM [11], which ensembles models by computing a weighted average of their output probabilities. Weights are obtained via a softmax over the loglikelihoods of the test example input of each model. As in the original BTM, ensembling can be restricted to the top-k models by zeroing the weights of all other models and renormalizing. See §A.1 for full details.
BTX We follow BTX [55], which upcycles an MoE from independently trained dense models. It copies the dense model parameters to the corresponding experts in MoE while averaging non-expert parameters such as attention layers for merging. The original BTX requires training all model parameters on combined datasets after merging. To approximate it as closely as possible while adhering to our setting, we perform this post-merge training on the public set only.
Unrestricted MoE To assess how closely our method approaches the benefits of full data access while preserving data separation, we construct an upper-bound reference model: a sparse MoE initialized from the public-only dense model and trained on the combined dataset, including all closed sets and Public Mix. As MoE training incurs roughly 2× the FLOPs of our approach for the same data size, we report both compute-controlled (1× FLOPs, 0.5× data) and data-controlled (2× FLOPs, 1× data) comparisons.
this section cite: ['b10', 'b54']

Section: Training Setup
For the public model M pub , we use a dense model with 7 billion parameters following the OLMo 2 architecture [78]. This model contains 32 layers with hidden dimension 4,096 and is trained on our public mix for 1 trillion tokens. Following [78], we use a learning rate of 0.0009 and the AdamW optimizer with parameters β 1 = 0.9 and β 2 = 0.95 and a cosine learning rate scheduler. The public model is pretrained using 512 H100 GPUs with a global batch size of 4 million tokens for three days.
Each data owner then takes this checkpoint and performs continued-pretraining for 50 billion tokens on their own data (totaling 400B tokens across all experts). For the optional router training, we use 5 billion tokens in total. The final FLEXOLMO, trained on 8 sets, has 37 billion total parameters with 20 billion active (4 active experts out of 8). More details can be found in §A.2.
this section cite: ['b77', 'b77']

Section: Results and Analysis
We conduct ablation studies and compare against a comprehensive set of baselines at a small scale with four experts-Public mix, math, educational text, and code (Table 1). We then evaluate our final model on the full setup including the Public mix and all seven simulated closed sets (Table 2).
Finally, we present an in-depth analysis to illustrate the behavior and effectiveness of FLEXOLMO.
this section cite: []

Section: Main Results
Individual experts excel at their specialized tasks As shown in Table 1, experts trained on each domain-specific set demonstrate strong performance in their specialized domains: the Math expert achieves the highest scores on math tasks, while the Code expert performs the best on coding benchmarks. However, these experts exhibit considerable performance degradation when evaluated on tasks outside their domains. Notably, the Code expert performs poorly on general benchmarks.
FLEXOLMO outperforms individual experts FLEXOLMO outperforms individual experts in most cases. It improves upon the model trained solely on public data, achieving an average 41% relative gain. Largest improvements appear on benchmarks where closed data significantly boosts Table 2: Evaluation of FLEXOLMO trained on eight sets (public mix and seven simulated closed sets) on 31 tasks across 10 categories, tested with 1,000 samples per subtask. "no RT" indicates no optional router training on proxy data ( §3.3.3). MC9 GEN5 MMLU MMLU Pro AGIEval BBH Math2 NewsG PoemG SciRIFF5 Code4 Avg. Prev. Public model 68.7 58.8 55.9 26.2 39.9 35.7 8.2 76.0 47.8 48.1 1.1 42.4 Individual experts Math 62.5 44.3 50.6 24.1 42.0 45.6 53.1 42.6 28.0 50.7 15.8 41.8 Code 40.5 39.4 29.5 14.5 27.4 38.1 6.0 45.1 28.2 48.0 21.0 30.7 Educational Text 64.3 52.1 56.5 27.0 39.7 40.3 13.6 57.6 51.8 51.7 3.0 41.6 News 46.5 48.6 36.4 15.2 25.7 30.9 2.5 77.7 26.9 47.0 0.0 32.5 Creative Writing 42.7 43.9 31.5 11.6 23.3 27.6 1.7 56.9 67.5 42.4 0.0 31.7 Academic 41.0 45.2 33.8 14.8 24.1 32.4 6.5 51.8 23.0 52.0 0.0 29.5 Reddit 64.7 36.5 56.1 25.5 35.5 19.7 2.5 54.1 8.6 32.7 1.7 30.7 Combined model BTM (top-2) 68.7 57.7 59.4 28.3 43.2 44.3 23.1 73.6 54.4 46.3 24.0 47.6 FLEXOLMO (no RT) 69.2 53.2 58.8 34.0 43.4 42.1 52.1 78.2 60.1 54.4 18.6 51.3 FLEXOLMO 70.8 59.8 60.4 30.9 45.1 46.4 48.5 80.7 62.2 54.3 17.2 52.4
individual expert performance, e.g., 35.6 → 47.1 on BBH, 8.1 → 50.7 on math, and 1.0 → 17.3 on coding. Notably, FLEXOLMO even matches or exceeds the performance of specialized experts on their respective tasks (e.g., on BBH and Math2).
this section cite: []

Section: FLEXOLMO achieves more effective merging than baselines
We also compare FLEXOLMO with baseline merging methods ( §4.3). All baselines outperform the model trained on Public Mix only. However, their performance is inconsistent: model soup and BTX are generally weak, 3 while prompt-based routing is highly unstable: it performs well when the classifier selects the correct expert, but degrades sharply when it does not. Among the baselines, BTM yields the best performance. Nonetheless, FLEXOLMO outperforms all prior model merging methods, beating the best baseline BTM by 10.1% relative on average. We attribute this to the MoE-based design of our model, which selectively activates different experts per layer, effectively combining the complementary strengths of each specialized model (see further analysis in §5.2).  Comparison to the unrestricted MoE Compared to the unrestricted MoE trained without considering data restrictions ( §4.3), FLEXOLMO outperforms the FLOP-controlled setting (1× FLOPs, 0.5× Data). It slightly underperforms the data-controlled model (2× FLOPs, 1× Data). This indicates that FLEXOLMO enables training without direct access to the data (requiring only model sharing) and flexible opt-in and opt-out functions while retaining strong performance.
Ablations on FLEXOLMO We further evaluate FLEXOLMO by removing different components introduced in §3.3: learning to coordinate, router initialization, and the bias term. Our results show that each component plays an important role, with the removal of any one leading to performance drop. In particular, we observe that randomly initializing router embeddings leads to the final learned router embeddings being very similar to each other, making the later merging of multiple experts harder. Furthermore, we confirm that FLEXOLMO benefits from additional router training ( §3.3.3) and using external embedders for router initialization, compared to using the public model's hidden states as the initialization (Table 3).
Final FLEXOLMO in the full setup Finally, we evaluate FLEXOLMO in the full eight-expert setup and compare it against the public-only model, individual experts trained on closed datasets, and BTM (top-2), the strongest baseline from Table 1. This was done by simply adding four additional experts, benefiting from FLEXOLMO's flexibility in easily adding new datasets. Consistent with earlier findings, FLEXOLMO outperforms all individual models (the public baseline and individual experts), demonstrating the synergistic effect of combining independently trained modules. Compared to the strongest baseline BTM, it achieves a 10% relative improvement on average (Table 2). FLEXOLMO excels on benchmarks where specialized experts perform well (BBH, Math2, NewsG, PoemG, SciRIFF5, Code4), matching or surpassing the experts, and also shows strong results on tasks where no single dataset suffices (e.g., MC9, Gen5, MMLU, MMLU Pro, AGI Eval).
this section cite: []

Section: Model Behavior Analysis
Routing patterns Figure 2 visualizes the router's token distribution across experts for various domain inputs. The router tends to activate the corresponding domain expert (e.g., math inputs activate the math expert), demonstrating its ability to identify the most relevant module. We also observe frequent activation of the public expert, likely due to our coordinated training strategy, where each expert is designed to complement the public expert. Also, different combinations of experts are activated at different layers. This highlights the model's layer-specific specialization and its greater expressivity than approaches that route inputs to a single expert (e.g., prompt-based routing).
this section cite: []

Section: Number of active experts
We further analyze how the number of active experts affects downstream task performance. As shown in Figure 3, performance consistently improves as more experts are activated, up to four experts, after which it plateaus. This suggests that the final model can operate efficiently as a sparse model by activating only four experts per input during inference.
Data opt-out FLEXOLMO offers a straightforward mechanism for opting out of specific datasets by removing the corresponding expert module at inference time. In Figure 4, we evaluate the model's performance after excluding the news expert. As expected, performance drops on in-domain tasks such as news generation. However, on unrelated tasks, performance remains largely unaffected.
this section cite: []

Section: Data Extraction Analysis
FLEXOLMO enables data owners to contribute to the model without sharing their data by instead sharing model weights trained locally on their data. A natural and important concern is whether their data can be extracted from these shared weights [108,109,110]. This risk is particularly relevant when training data includes private or confidential information.
We empirically assess this risk by implementing training data extraction following prior work [108]. Specifically, we sample 10,000 documents from the math data. 4 From each document, we extract a 32-token prefix and use it to generate 256-token continuation using top-k sampling (k = 50), top-p sampling (p = 0.95), and a temperature of 1.0. We sample 10 times per prefix, and if any of the generated outputs achieves a normalized Levenshtein similarity of 0.9 or higher with the original document, we consider that document to be extracted. To validate our implementation, we apply it to a model overfitted on the dataset (trained for 100 epochs), and observe a 60% extraction rate.
Our results are as follows: (1) A public model that has not seen any math data yields an extraction rate of 0.1%. (2) A dense model trained on the math dataset (i.e., math expert) yields 1.6%. (3) FLEXOLMO with the math expert included yields 0.7%.
These results lead to the following conclusions. First, in practice, it is difficult to extract a substantial portion of the training data, which is in line with previous findings [108]. However, if a model includes any weights trained on the data, nonzero (though small) fraction of the data may be extractable. If data owners are comfortable with this minimal leakage, as long as a meaningful fraction of the data remains not extractable, we believe that FLEXOLMO, in its current form, is a viable solution. If the owners' data includes any private or sensitive information, we recommend training experts using differentially private (DP) learning methods before contributing them to the model, which provides formal privacy guarantees. Applying DP is orthogonal to our architecture, and different data owners can make independent decisions, providing flexibility without compromising the overall design.
this section cite: ['b107', 'b108', 'b109', 'b107', 'b107']

Section: Conclusion
We introduce FLEXOLMO, a new class of LMs that solves real-world data constraint challenges with (1) modular, distributed training, where different model parameters are independently trained on disjoint and locally maintained datasets, and (2) data-flexible inference, where data can be selectively included at inference-time, with guarantees. We show that FLEXOLMO significantly outperforms competitive baselines, while providing the benefits of distributed training and flexible inference. Name # Tokens (B) Public Mix 2.37 × 10 3 News 158.0 Creative Writing 201.9 Math 20.3 StarCoder 83.0 Academic 58.6 Educational text 102.2 Reddit 9.9
this section cite: []

Section: References
Ref_id:b0 Title: TOFU: A task of fictitious unlearning for LLMs Year: (2024)
Ref_id:b1 Title: Machine unlearning sixway evaluation for language models Year: (2024)
Ref_id:b2 Title: Machine unlearning doesn't do what you think Year: (2024)
Ref_id:b3 Title: Federated learning: Strategies for improving communication efficiency Year: (2016)
Ref_id:b4 Title: Communication-Efficient Learning of Deep Networks from Decentralized Data Year: (2017-04)
Ref_id:b5 Title: Federated learning with differential privacy: Algorithms and performance analysis Year: (2019)
Ref_id:b6 Title: Trading off privacy, utility, and efficiency in federated learning Year: (2022)
Ref_id:b7 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b8 Title: Open mixture-ofexperts language models Year: (2024)
Ref_id:b9 Title: A survey on model moerging: Recycling and routing among specialized experts for collaborative learning Year: (2024)
Ref_id:b10 Title: Branch-train-merge: Embarrassingly parallel training of expert language models Year: (2022)
Ref_id:b11 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b12 Title:  Year: (2024)
Ref_id:b13 Title: Project aurora: the power of data, technology and collaboration to combat money laundering across institutions and borders Year: ()
Ref_id:b14 Title:  Year: (2021)
Ref_id:b15 Title: Data sovereignty: A review Year: (2021)
Ref_id:b16 Title: Databricks lakeguard: Supporting fine-grained access control and multi-user capabilities for apache spark workloads Year: (2025)
Ref_id:b17 Title: Productivity assessment of neural code completion Year: (2022)
Ref_id:b18 Title: Foundation models and fair use Year: (2023)
Ref_id:b19 Title: Consent in crisis: The rapid decline of the ai data commons Year: (2024)
Ref_id:b20 Title: Art and the science of generative ai Year: (2023)
Ref_id:b21 Title: The foundation model transparency index Year: (2023)
Ref_id:b22 Title: Foundation models and copyright questions Year: (2023)
Ref_id:b23 Title: A large-scale audit of dataset licensing and attribution in ai Year: (2024)
Ref_id:b24 Title: On the opportunities and risks of foundation models Year: (2021)
Ref_id:b25 Title: The data provenance initiative: A large scale audit of dataset licensing & attribution in ai Year: (2023)
Ref_id:b26 Title: Into the laion's den: Investigating hate in multimodal datasets Year: (2023)
Ref_id:b27 Title: Documenting large webtext corpora: A case study on the colossal clean crawled corpus Year: (2021)
Ref_id:b28 Title: Detecting pretraining data from large language models Year: ()
Ref_id:b29 Title: Advances and open problems in federated learning Year: (2021)
Ref_id:b30 Title: Federated optimization in heterogeneous networks Year: (2020)
Ref_id:b31 Title: Fedlora: When personalized federated learning meets low-rank adaptation Year: (2024)
Ref_id:b32 Title: Fedprompt: Communication-efficient and privacy-preserving prompt tuning in federated learning Year: (2023)
Ref_id:b33 Title: Federated lora with sparse communication Year: (2024)
Ref_id:b34 Title: Public-key cryptosystems based on composite degree residuosity classes Year: (1999)
Ref_id:b35 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b36 Title: Attack of the tails: Yes, you really can backdoor federated learning Year: (2020)
Ref_id:b37 Title: Exploring adversarial attacks in federated learning for medical imaging Year: (2024)
Ref_id:b38 Title: Federated learning with buffered asynchronous aggregation Year: (2022)
Ref_id:b39 Title: Building machine learning models like open source software Year: (2023)
Ref_id:b40 Title: Heterogeneous swarms: Jointly optimizing model roles and weights for multi-llm systems Year: (2025)
Ref_id:b41 Title: Adaptersoup: Weight averaging to improve generalization of pretrained language models Year: (2023)
Ref_id:b42 Title: Collective model intelligence requires compatible specialization Year: (2024)
Ref_id:b43 Title: Model swarms: Collaborative search to adapt llm experts via swarm intelligence Year: (2024)
Ref_id:b44 Title: Tiesmerging: Resolving interference when merging models Year: (2023)
Ref_id:b45 Title: Merging models with fisher-weighted averaging Year: (2022)
Ref_id:b46 Title: Editing models with task arithmetic Year: (2023)
Ref_id:b47 Title: Evolutionary optimization of model merging recipes Year: (2024)
Ref_id:b48 Title: Adamerging: Adaptive model merging for multi-task learning Year: (2024)
Ref_id:b49 Title: Merge to learn: Efficiently adding skills to language models with model merging Year: (2024)
Ref_id:b50 Title: When one llm drools, multi-llm collaboration rules Year: (2025)
Ref_id:b51 Title: Branch-train-merge: Embarrassingly parallel training of expert language models Year: (2022)
Ref_id:b52 Title: Scaling expert language models with unsupervised domain discovery Year: (2023)
Ref_id:b53 Title: Demix layers: Disentangling domains for modular language modeling Year: (2022)
Ref_id:b54 Title: Branch-train-mix: Mixing expert llms into a mixture-of-experts llm Year: (2024)
Ref_id:b55 Title: Scalable multi-domain adaptation of language models using modular experts Year: (2024)
Ref_id:b56 Title: Bts: Harmonizing specialized experts into a generalist llm Year: (2025)
Ref_id:b57 Title: Bam! just like that: Simple and efficient parameter upcycling for mixture of experts Year: (2024)
Ref_id:b58 Title: Moduleformer: Modularity emerges from mixture-of-experts Year: (2023)
Ref_id:b59 Title: Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning Year: (2023)
Ref_id:b60 Title: Sparse mixture of low rank adaptation Year: (2023)
Ref_id:b61 Title: LoRAMoE: Alleviating world knowledge forgetting in large language models via MoE-style plugin Year: (2024-08)
Ref_id:b62 Title: Adapterfusion: Non-destructive task composition for transfer learning Year: (2020)
Ref_id:b63 Title: Adapterswap: Continuous training of llms with data removal and access-control guarantees Year: (2024)
Ref_id:b64 Title: Exact unlearning of finetuning data via model merging at scale Year: (2025)
Ref_id:b65 Title: Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024)
Ref_id:b66 Title: Generative representational instruction tuning Year: (2025)
Ref_id:b67 Title: Nexus: Specialization meets adaptability for efficiently training mixture of experts Year: (2024)
Ref_id:b68 Title: Loraretriever: Input-aware lora retrieval and composition for mixed tasks in the wild Year: (2024)
Ref_id:b69 Title: Exploring the benefits of training expert language models over instruction tuning Year: (2023)
Ref_id:b70 Title: Dam: Dynamic adapter merging for continual video qa learning Year: (2025)
Ref_id:b71 Title: Routerretriever: Routing over a mixture of expert embedding models Year: (2024)
Ref_id:b72 Title: Token-level adaptation of lora adapters for downstream task generalization Year: (2023)
Ref_id:b73 Title:  Year: (2025)
Ref_id:b74 Title: Organize the web: Constructing domains enhances pre-training data curation Year: (2025)
Ref_id:b75 Title:  Year: ()
Ref_id:b76 Title: Leandro von Werra, and Harm de Vries. The stack: 3 tb of permissively licensed source code Year: (2022)
Ref_id:b77 Title:  Year: (2025)
Ref_id:b78 Title: Unlocking trillions of tokens in pdfs with vision language models Year: (2025)
Ref_id:b79 Title: peS2o (Pretraining Efficiently on S2ORC) Dataset. Technical report, Allen Institute for AI, 2023 Year: ()
Ref_id:b80 Title: S2ORC: The semantic scholar open research corpus Year: (2020-07)
Ref_id:b81 Title: Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big -data-centric training of a small language model Year: (2025)
Ref_id:b82 Title:  Year: (2024)
Ref_id:b83 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b84 Title: Boolq: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b85 Title: Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph Year: (2018)
Ref_id:b86 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b87 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b88 Title: Reasoning about physical commonsense in natural language Year: (2019)
Ref_id:b89 Title: Socialiqa: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b90 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2019)
Ref_id:b91 Title: Coqa: A conversational question answering challenge Year: (2019)
Ref_id:b92 Title: Squad: 100,000+ questions for machine comprehension of text Year: (2016)
Ref_id:b93 Title: Natural questions: A benchmark for question answering research Year: (2019)
Ref_id:b94 Title: Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017)
Ref_id:b95 Title: Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs Year: (2019)
Ref_id:b96 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b97 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b98 Title: Agieval: A human-centric benchmark for evaluating foundation models Year: (2023)
Ref_id:b99 Title: Challenging big-bench tasks and whether chain-of-thought can solve them Year: (2022)
Ref_id:b100 Title:  Year: (2021)
Ref_id:b101 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b102 Title: Program synthesis with large language models Year: (2021)
Ref_id:b103 Title: Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation Year: (2023)
Ref_id:b104 Title:  Year: (2021)
Ref_id:b105 Title: Sciriff: A resource to enhance language model instructionfollowing over scientific literature Year: (2024)
Ref_id:b106 Title:  Year: (2024)
Ref_id:b107 Title: Extracting training data from large language models Year: (2021)
Ref_id:b108 Title: Extracting training data from diffusion models Year: (2023)
Ref_id:b109 Title: Extracting memorized pieces of (copyrighted) books from open-weight language models Year: (2025)
Ref_id:b110 Title: Arctic-embed: Scalable, efficient, and accurate text embedding models Year: (2024)
Ref_id:b111 Title: The pushshift reddit dataset Year: (2020)
Ref_id:b112 Title: Accelerating the science of language models Year: (2024-08)
Ref_id:b113 Title: Does your data spark joy? performance gains from domain upsampling at the end of training Year: (2024)
Ref_id:b114 Title: The narrativeqa reading comprehension challenge Year: (2017)
Ref_id:b115 Title: Olmes: A standard for language model evaluations Year: (2025)
Ref_id:b116 Title: Solving quantitative reasoning problems with language models Year: (2022)
Ref_id:b117 Title: Muse: Machine unlearning six-way evaluation for language models Year: (2024)
