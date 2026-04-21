Title: UNCOVER UNDERLYING CORRESPONDENCE FOR ROBUST MULTI-VIEW CLUSTERING
Abstract: Multi-view clustering (MVC) aims to group unlabeled data into semantically meaningful clusters by leveraging cross-view consistency. However, real-world datasets collected from the web often suffer from noisy correspondence (NC), which breaks the consistency prior and results in unreliable alignments. In this paper, we identify two critical forms of NC that particularly harm clustering: i) category-level mismatch, where semantically consistent samples from the same class are mistakenly treated as negatives; and ii) sample-level mismatch, where collected cross-view pairs are misaligned and some samples may even lack any valid counterpart. To address these challenges, we propose Cor-reGen, a generative framework that formulates noisy correspondence learning in MVC as maximum likelihood estimation over underlying cross-view correspondences. The objective is elegantly solved via an Expectation-Maximization algorithm: in the E-step, soft correspondence distributions are inferred across views, capturing class-level relations while adaptively down-weighting noisy or unalignable samples through GMM-guided marginals; in the M-step, the embedding network is updated to maximize the expected log-likelihood. Extensive experiments on both synthetic and real-world noisy datasets demonstrate that our method significantly improves clustering robustness. The code is available at https://github.com/XLearning-SCU/2026-ICLR-CorreGen.

Section: INTRODUCTION
Describing the same object from multiple perspectives (Yan et al., 2021) or modalities (Sharma et al., 2018), multi-view data have become increasingly prevalent in real-world applications. To exploit such data, contrastive multi-view clustering (MVC) has emerged as a powerful unsupervised paradigm (Qin et al., 2025b;Wang et al., 2025a). Relying on the consistency prior that views from the same instance should be semantically aligned, contrastive MVC pulls positive pairs (i.e., views of the same instance) closer while pushing negative pairs (i.e., views from different instances) apart in the embedding space. Through this process, it could learn a shared embedding space across views and group unlabeled samples into semantically meaningful clusters.
However, this prior is often difficult to satisfy. In practice, multi-view datasets are commonly constructed by crawling paired data from the web, such as images with their associated alt text (Wang et al., 2015). This automatic process inevitably introduces the noisy correspondence (NC) problem (Huang et al., 2021), where cross-view pairs are incorrectly matched. Such noise undermines the cross-view consistency prior and severely distorts the semantic structure of the learned embedding space.
In this paper, we identify two major types of NC that are particularly harmful to clustering: i) Category-level mismatch, where views from different modalities but belonging to the same class are mistakenly treated as negatives by contrastive MVC methods, despite their underlying semantic consistency; ii) Sample-level mismatch, which manifests in two scenarios: alignable mispairs, where A ripe red apple hanging on a tree branch.
Small red-green apples hanging on a tree.
A pet sitting inside on the green grass.
A bird !@Øü on the gro brigh #¿ ùe eyes, aeç~! a sample is wrongly paired with a mismatched view despite having a correct counterpart elsewhere; and unalignable samples, where no valid counterpart exists due to corruption, noise, or poor data quality. Such issues are especially prevalent in web-collected data, where the pairwise noise ratio can exceed 20% (Sharma et al., 2018;Wang et al., 2015). Critically, manually verifying or cleaning these correspondences is prohibitively expensive, underscoring the need for robust multi-view clustering methods. To address NC, recent works (Qu et al., 2025) mainly adopt either pairwise reweighting or realignment strategies, as illustrated in Fig. 1. However, both approaches overlook category-level semantics and unalignable samples, leading to suboptimal results in clustering.
Recognizing these limitations, we shift from the existing discriminative contrastive objective to a generative one. Specifically, we formulate noisy correspondence learning in MVC as a maximum likelihood estimation objective of the underlying joint distribution, in which the counterparts across views are modeled as unobserved latent variables. Unlike previous methods (Wang et al., 2025b;Qin et al., 2025a) that focus on verifying whether given positive or negative pairs are correctly aligned, our formulation uncovers the underlying correspondences without heavily relying on predefined (potentially noisy) pairs. By maximizing the overall log-likelihood, we capture the semantic structure in a principled and probabilistic manner.
To effectively optimize the proposed objective, we develop an Expectation-Maximization (EM) based algorithm CorreGen. In the E-step, the goal is to infer a latent correspondence distribution across views. We first estimate the marginal likelihood of each sample by fitting a Gaussian Mixture Model in the embedding space. Intuitively, this estimation assigns higher probabilities to samples that lie in large and coherent clusters, while noisy or unalignable samples receive lower probabilities. These marginals serve as constraints to solve an optimal transport formulation, yielding a soft many-to-many assignment that captures category-level relationships across views. In the M-step, the estimated correspondences are used to maximize the expected log-likelihood, updating the embedding network such that semantically consistent pairs are assigned higher likelihoods. Iterating between the two steps gradually uncovers reliable correspondences and refines robust cluster representations. In summary, the contribution of our work can be summarized as follows:
• We identify and formalize two types of noisy correspondence in MVC: category-level mismatch and sample-level mismatch, where both are prevalent in real-world multi-view datasets and harmful to clustering.
• We propose CorreGen, a novel generative framework that models latent cross-view correspondences through maximum likelihood estimation, solved elegantly via an EM algorithm. Furthermore, we prove that the standard InfoNCE is a special case of our formulation under specific assumptions.
• We introduce a principled E-step solution that jointly models category-level correspondences and suppresses sample-level noise by leveraging GMM-guided marginals. Extensive experiments on both synthetic and real-world noisy datasets validate the effectiveness of our approach. Notably, our method achieves 10% accuracy improvements on the challenging UMPC-Food101 dataset (Wang et al., 2015).
this section cite: ['b36', 'b25', 'b29', 'b12', 'b25', 'b29', 'b24', 'b29']

Section: RELATED WORK
Robust Multi-view Clustering aims to handle imperfections that commonly occur in real-world datasets. These imperfections can be broadly categorized into two types: i) Incomplete Multi-view Problem (IMP) arises when some views are missing, resulting in incomplete cross-view information. To mitigate this issue, recent methods adopt various completion-based strategies such as anchor learning (Liu et al., 2024), subspace learning (Zhang et al., 2024), or diffusion models (Zhang et al., 2025). These approaches aim to impute the missing views and recover complete multi-view representations; ii) Partially view-aligned Problem (PVP) occurs when the correspondences across views are misaligned. For example, in multi-camera surveillance, images of the same person from different cameras may be temporally asynchronous (Huang et al., 2020). To address this, He et al. (2024) introduce a variational contrastive learning framework to realign unpaired data, while Yan et al. (2025) design a multi-stage strategy that iteratively updates cross-view correspondences for unpaired data.
Although both PVP and NC address erroneous cross-view correspondences, the NC problem studied in this paper differs in two significant aspects. First, misalignments in NC are unobserved, with no manually verified labels or alignment indicators available (Lee et al., 2018). Second, NC encompasses not only instance-level mismatches, but also category-level misalignments and even unalignable samples that lack valid counterparts across views.
Noisy Correspondence Learning was first introduced in cross-modal retrieval (Huang et al., 2021), where mismatched image-text pairs are mistakenly treated as true positives. Recently, this problem has garnered increasing attention across a range of domains, including video reasoning (Lin et al., 2024), graph matching (Lin et al., 2023), person re-identification (Yang et al., 2022a) and multiview clustering (Sun et al., 2024;2025). Existing solutions can be broadly categorized into two groups: i) Reweighting-based methods (Yang et al., 2024) aim to reduce the impact of mismatched pairs by assigning them lower weights during training. For example, Huang et al. (2021) adjust the margins in triplet contrastive loss to account for false positives; ii) Realignment-based methods (Lin et al., 2024) attempt to reassign each sample to a more plausible counterpart across views, thereby mitigating alignment errors.
Although existing methods achieve promising results, they mainly refine given positive pairs while overlooking potential category-level correspondences, leading to suboptimal clustering performance. Different from these discriminative approaches, we propose a generative objective for noisy correspondence learning in MVC, which assigns higher likelihoods to semantically consistent samples and uncovers latent correspondences. Notably, our optimization does not rely heavily on off-the-shelf pairs, thereby mitigating the noisy correspondence problem from a new perspective.
this section cite: ['b19', 'b43', 'b44', 'b11', 'b9', 'b35', 'b14', 'b12', 'b18', 'b17', 'b26', 'b40', 'b12', 'b18']

Section: METHOD
In this section, we first introduce the problem setting and formalize correspondence learning in multi-view clustering (MVC) as a generative maximum likelihood estimation problem in Sec.
3.1. To optimize this objective, we propose CorreGen, an EM-based framework in Sec. 3.2, and detail its two steps in Sec. 3.2.1 and Sec. 3.2.2. 3.1 PROBLEM DEFINITION Given a multi-view dataset {(x (1) i , . . . , x
i )} N i=1 with N instances observed from V views, the goal of MVC is to learn an encoder f θ that maps each view x (v) i into a shared embedding space, i.e., z
(v) i = f θ (x (v) i ).
Ideally, the distribution of these embeddings should form C well-separated semantic clusters, such that traditional clustering algorithms (e.g., K-means (McQueen, 1967)) can easily distinguish them.
To achieve this goal, recent contrastive MVC methods (Yang et al., 2023) pull positive pairs (i.e., views of the same instance) closer while pushing negative pairs (i.e., views from different instances) apart in the embedding space. Formally, for any pair of views (v 1 , v 2 ) with v 1 ̸ = v 2 , the positive and negative sets are defined as
P + v1,v2 = N i=1 { (x (v1) i , x(v2)
i , t 12 ii = 1)}, P - v1,v2 = N i=1 N j=1,j̸ =i { (x (v1) i , x(v2)
j , t 12 ij = 0)}, (1) where t 12 ij ∈ {0, 1} is an indicator variable that equals 1 if x (v1) i and x (v2) j
belong to the same instance, and 0 otherwise. Nevertheless, contrastive MVC essentially formulates an instance-level discriminative task (Wu et al., 2018), which overlooks the intrinsic cluster structure of data. As a result, real-world multi-view datasets are particularly vulnerable to the noisy correspondence problem, where the assumed cross-view alignment fails to hold. For clarity, we formalize its two manifestations, namely category-level mismatch and sample-level mismatch, as defined below.
Definition 1 (Category-level mismatch).
Consider a cross-view pair (x (v1) i , x (v2) j , t 12 ij ), where t 12 ij ∈ {0, 1} denotes whether the pair is treated as positive or negative. Let c (v1) i and c (v2) j be the oracle class labels of x (v1) i and x (v2) j , respectively. A category-level mismatch occurs if c (v1) i = c (v2) j but t 12 ij = 0, i.e., samples from the same semantic class are incorrectly assigned as a negative pair.
In other words, category-level mismatch occurs when semantically related instances are mistakenly treated as negatives. Ideally, all cross-view pairs of samples from the same class should be regarded as positives with t 12 ij = 1, rather than only those from the same instance.
Definition 2 (Sample-level mismatch). Consider a cross-view pair (x
(v1) i , x(v2)
i , t 12 ii ), where c (v1) i and c (v2) i denote the oracle class labels of x (v1) i and x (v2) i , respectively. A sample-level mismatch occurs if either i) c (v1) i ̸ = c (v2) i , or ii) at least one of c (v1) i or c (v2) i
does not correspond to any valid class. In both cases, the pair cannot be regarded as a valid positive correspondence.
Specifically, sample-level mismatch admits two scenarios: i) alignable mispaired: although the constructed pair is incorrect, the sample
x (v1) i still has a valid counterpart x (v2) k
in the other view. This case often co-occurs with category-level mismatch; ii) unalignable mispaired: there is no valid counterpart, e.g., the sample x (v1) i might be corrupted or purely noisy data.
These two types of complex noisy correspondence motivate a more fundamental question: can we reduce the reliance on pre-defined pairs and instead directly model the intrinsic relationships that couple different views? Building on this intuition, we adopt a generative formulation that maximizes the marginal log-likelihood of the observed multi-view data (Bengio et al., 2013):
θ * = arg max θ V v=1 N i=1 log p(x (v) i ; θ).(2)
In multi-view clustering, each sample in one view may be associated with multiple counterparts in another view. Since these associations are unknown a priori, we treat them as latent variables. By aggregating over all unordered view pairs (v i , v j ), the objective can be reformulated as:
θ * = arg max θ V v1 N i V v2 log N j p(x (v1) i , x (v2) j ; θ).(3)
Maximizing this marginal likelihood implicitly encourages the model to learn a meaningful joint distribution p(x
(v1) i , x(v2) j
; θ). In particular, to maximize the inner summation over j, the parameters θ must assign higher joint probability to semantically consistent pairs, thereby revealing the underlying cross-view correspondences in a probabilistic sense.
Compared with discriminative objectives, this generative formulation offers two key advantages: i) it alleviates the heavy reliance on pre-defined positive and negative pairs, making it naturally robust to sample-level unmatchable cases; ii) it captures many-to-many probabilistic correspondences across views, which better reflects the complex coupling of real-world multi-view data and mitigates category-level mismatch. However, the nested summation in Eq. ( 3) makes direct optimization intractable. To address this, we cast the objective into the Expectation-Maximization (EM) framework and present the theoretical derivation in the next section.
this section cite: ['b21', 'b41', 'b33', 'b0']

Section: CORRESPONDENCE GENERATION VIA EXPECTATION-MAXIMIZATION
To simplify the derivation of the joint log-likelihood defined in Eq. ( 3), we first consider a subset of the objective involving only two views:
θ * = arg max θ N i=1 log N j=1 p(x (v1) i , x (v2) j ; θ).(4)
Directly optimizing Eq. ( 4) is intractable due to the nested log-sum over latent variables. To address this, we introduce an auxiliary distribution Q(x
(v2) j ) over x (v2) j such that N j=1 Q(x (v2) j
) = 1. This allows us to derive a lower bound:
N i=1 log N j=1 p(x (v1) i , x (v2) j ; θ) = N i=1 log N j=1 Q(x (v2) j ) p(x (v1) i , x (v2) j ; θ) Q(x (v2) j ) ,(5)
≥ N i=1 N j=1 Q(x (v2) j ) log p(x (v1) i , x(v2)
j ; θ) Q(x (v2) j ) ,(6)
where the inequality follows from Jensen's inequality. The bound becomes tight when Q(x
(v2) j ) = p(x (v2) j ; x (v1) i
, θ), i.e., when the auxiliary distribution matches the posterior under the current parameters θ ( t). Substituting this choice of Q into the bound gives:
θ * = arg max θ N i=1 N j=1 Q(x (v2) j ) log p(x (v1) i , x (v2) j ; θ) - N i=1 N j=1 Q(x (v2) j ) log Q(x (v2) j ) (7) = arg max θ N i=1 N j=1 p(x (v2) j ; x (v1) i , θ (t) ) log p(x (v1) i , x (v2) j ; θ),(8)
where the entropy term -
N i N j Q(x (v2) j ) log Q(x (v2) j
) is omitted since it is independent of θ.
In the E-step, we estimate the posterior distribution p(x
(v2) j ; x (v1) i , θ (t)
), which provides a soft assignment of correspondences between samples across views. In the M-step, we maximize the weighted log-likelihood in Eq. ( 8), updating the parameters θ guided by the correspondences inferred in the E-step. By aggregating over all views, the above derivation naturally generalizes to multiple views. Fig. 2 shows an overview of the above EM process and the details of the two steps will be discussed in the next section.
this section cite: []

Section: E-STEP: ESTIMATING UNDERLYING CORRESPONDENCES
In the E-step, we estimate the posterior distribution of latent correspondences p(x
(v2) j ; x (v1) i , θ (t) ) under the current parameters θ (t) : p(x (v2) j ; x (v1) i , θ (t) ) = p(x (v1) i , x (v2) j ; θ (t) ) p(x (v1) i ; θ (t) ) ,(9)
which naturally decomposes the estimation into two parts, namely, the marginal distribution of individual views and the joint distribution across views.
this section cite: []

Section: GMM Guide

this section cite: []

Section: Underlying Correspondence Estimation

this section cite: []

Section: Virtual Sample
Robust Correspondence Learning
this section cite: []

Section: E-step M-step

this section cite: []

Section: Guide
Figure 2: Overview of the CorreGen framework which operates via an EM procedure: the E-step infers the underlying correspondence distribution using GMM-guided marginals and a virtual sample mechanism to handle noise; the M-step subsequently utilizes these estimated soft correspondences to guide the robust representation learning.
First, we estimate the joint distribution between views v 1 and v 2 , represented as a matrix P ∈ R N ×N + where each entry t) ). A good estimate of P should not only satisfy the marginal constraints but also capture the semantic dependency between the two views. Inspired by recent works that utilize Optimal Transport (OT) to model complex cross-view relationships for clustering (Deng et al., 2025;Fu et al., 2025), we formulate the estimation of the optimal joint distribution as an OT problem. We introduce a correlation function s(z
P ij = p(x (v1) i , x (v2) j ; θ (
(v1) i , z (v2) j
) (e.g. cosine similarity) to measure the semantic correlations of a sample pair under the current parameters θ (t) , with z
(v) i = f θ (t) (x (v) i ).
Then the expected correlation is defined as
E P [s] = N i=1 N j=1 P ij s(z (v1) i , z (v2) j ).(10)
We then seek the optimal joint distribution by maximizing this expectation:
P * = arg max P ∈Π(p (v 1 ) ,p (v 2 ) ) E P [s] s.t Π(p (v1) , p (v2) ) = P ∈ R N ×N + P 1 N = p (v1) , P ⊤ 1 N = p (v2) ,(11)
where p (vi) is the marginal distribution vector for view v i , i.e., p t) ). This formulation ensures that the estimated joint distribution preserves the marginal constraints while assigning higher probability mass to semantically correlated pairs. However, due to the sample-level unalignable problem, there may exist outliers whose joint probabilities with all other samples should ideally be close to zero.
i = p(x (vi) i ; θ ((vi)
this section cite: ['b3', 'b6']

Section: Virtual Sample for Partial Alignment.
To handle the outliers and obtain a more realistic joint distribution, we first introduce a virtual sample for each view to represent the outliers. Let ρ denote the potential noise ratio, which corresponds to the marginal probability mass of the virtual sample. We then augment the joint distribution to P ∈ R (N +1)×(N +1) + , ensuring that the total probability mass assigned to outliers equals ρ. Formally, P satisfies
P 1 N +1 = [p (v1) ; ρ], P ⊤ 1 N +1 = [p (v2) ; ρ],(12)
which enables the model to absorb unalignable or noisy samples into the virtual probability mass.
Recall from Eq. ( 9) that estimating the posterior probabilities requires both the joint distribution p(x t) ) and the marginal distribution p(x t) ). In the expectation formulation Eq. ( 11), these marginals act as constraints on the feasible set of couplings Π(p (v1) , p (v2) ), which essentially determines how many valid counterparts each sample can align with. Under category-level mismatch, the number of valid counterparts is not uniform but depends on the size and structure of the sample's semantic class. Therefore, the marginal distribution should naturally reflect this variability: samples from larger clusters or closer to cluster centers are assigned higher alignment mass, while outliers receive lower probabilities.
(v1) i , x (v2) j ; θ (
(v1) i ; θ (
this section cite: []

Section: GMM-guided Marginal Estimation.
We assume that each sample is generated from a latent semantic cluster, which can be approximated by an anisotropic Gaussian distribution x (v) i ∼ N (µ c , Σ c ). Accordingly, we fit the embedding space of each view with a Gaussian Mixture Model (GMM) and compute the posterior responsibility of each cluster for every sample. The marginal probability is then estimated as
p(x (v) i ; θ (t) ) = m di -1 m -1 • N c N ,(13)
d i = exp -ϵ (z (v) i -µ c ) ⊤ Σ -1 c (z (v) i -µ c ) ,(14)
where N c is the number of samples assigned to cluster c by GMM, ϵ and m are shaping parameters. Concretely, we first compute the Mahalanobis distance in Eq. ( 14) between each sample and its cluster center, and map the result through an exponential kernel to obtain an assignment confidence d i . This confidence is further passed through a curve-shaping function m d i -1 m-1 , which amplifies the contrast between high-and low-confidence samples: samples closer to the cluster center receive disproportionately higher weights, while distant ones are smoothly down-weighted rather than suppressed abruptly. Finally, the re-scaled confidence is combined with the cluster proportion N c /N to yield the final probability to fill the marginal distribution in Eq. ( 11). In practice, we set ϵ = 0.1 and m = 10, and apply a momentum update to stabilize training. Proposition 1. Eq. ( 11) with virtual sample can be solved by an efficient scaling algorithm if adding an entropy regularization λH( P ), where λ is a regularization factor. Specifically, we derive the optimal augmented joint distribution P * through the following iterations:
P * = Diag(u) exp( S/λ)Diag(v), with iteration update u ← p(v1) /(exp( S/λ)v), v ← p(v2) /(exp( S⊤ /λ)u).(15)
where u ∈ R N +1 + , v ∈ R N +1 + are two scaling vectors, p(vi) = [p (vi) ; ρ]. The extended correlation matrix S ∈ R (N +1)×(N +1) is constructed as:
S = S 0 N ×1 0 1×N A , (16
)
where
S ij = s(z (v1) i , z (v2) j
) and A is a constant. The optimal joint distribution estimation P * is obtained by discarding the last row and column of P * , i.e., P * = P * 1:N,1:N . The proof is provided in Appendix A.
this section cite: []

Section: M-STEP: ROBUST CORRESPONDENCE LEARNING
In the M-step, we maximize the overall log-likelihood of the observed data based on the estimated posterior distribution. To make Eq. ( 8) tractable, we approximate the joint distribution p(x
(v1) i , x (v2) j
; θ) by normalizing the similarity scores of embeddings in the latent space
p(x (v1) i , x (v2) j ; θ) = exp(s(z (v1) i , z (v2) j )/τ ) N m=1 N n=1 exp(s(z (v1) m , z (v2) n )/τ ) ,(17)
where
z (v) i = f θ (x (v) i ) denotes the embedding of x (v) i
and τ is a temperature parameter. According to Eq. ( 9), we compute the posterior using the optimal joint distribution P * and marginals p (v1)  obtained in the E-step, defined as
Q ij = P * ij /p (v1) i
. Substituting this parameterization into Eq. ( 8), the M-step objective becomes
θ * = arg max θ N i=1 N j=1 Q ij log exp(s(z (v1) i , z (v2) j )/τ ) N m=1 N n=1 exp(s(z (v1) m , z (v2) n )/τ ) ,(18)
where s(•, •) denotes a correlation function. Unlike contrastive objectives that rely on manually defined positive/negative pairs, this formulation leverages the soft correspondences P * inferred in the E-step, thereby mitigating the negative effects of noisy correspondence and enabling more robust representation learning. Importantly, we find that the widely used InfoNCE loss can be unified into our framework as a special case as stated below. Proposition 2. If the marginal distribution p(x (v) i ; θ) is uniform and the posterior probability degenerates to p(x (v2) i ; x (v1) i , θ) = 1 (i.e., only paired cross-view samples are treated as valid positives), then Eq. (8) reduces to the standard InfoNCE contrastive objective:
θ * = arg max θ N i=1 log exp(s(z (v1) i , z (v2) i )/τ ) N n=1 exp(s(z (v1) i , z (v2) n )/τ ) . (19
)
The proof is in Appendix B.
this section cite: []

Section: EXPERIMENTS
In this section, we conduct extensive experiments to evaluate the effectiveness of our method in addressing both category-level and sample-level noisy correspondence. Our study is guided by the following research questions: Q1: Does our method outperform existing robust MVC approaches under noisy correspondence (Section 4.2)? Q2: Can our method reliably uncover underlying categorylevel correspondences across views (Section 4.3)? Q3: How does performance vary under different levels of mismatch (Appendix D)? Q4: How sensitive is our method to hyperparameter choices (Appendix E)? Q5: Are the proposed components crucial for the improvements (Appendix F)?
this section cite: []

Section: EXPERIMENTAL SETUP
Datasets. We evaluate our method on four widely used datasets: Scene15 (Fei-Fei & Perona, 2005), Caltech101 (Li et al., 2015), LandUse21 (Yang & Newsam, 2010), and UMPC-Food101 (Wang et al., 2015). Notably, UMPC-Food101 contains images from 101 food categories paired with recipes crawled from the web, which inevitably introduces substantial irrelevant or noisy information. Representative examples of such noisy image-text pairs are provided in Appendix I.
Baselines. We compare CorreGen against seven state-of-the-art MVC methods, including DCP (Lin et al., 2022), SURE (Yang et al., 2022b), GCFAgg (Yan et al., 2023), CGCN (Wang et al., 2024), DIVIDE (Lu et al., 2024), CANDY (Guo et al., 2024), and ROLL (Sun et al., 2025). For fair comparison, we apply a view realignment strategy to the learned representations following prior studies (Guo et al., 2024;Sun et al., 2025), where realignment is consistently performed within batches of 512 to ensure fair evaluation.
Implementation Details. CorreGen introduces a generative objective for MVC that can be seamlessly integrated into existing contrastive frameworks. We implement it on top of DIVIDE (Lu et al., 2024) as the base model. More details are provided in Appendix C.
this section cite: ['b5', 'b15', 'b42', 'b29', 'b16', 'b34', 'b30', 'b20', 'b7', 'b27', 'b7', 'b27', 'b20']

Section: PERFORMANCE COMPARISON (Q1)
Since MVC is an unsupervised task, category-level correspondences depend on the underlying class sizes and distributions, making category-level mismatch an intrinsic challenge rather than one that can be explicitly specified. Therefore, in this section, we focus on evaluating model performance under different sample-level mismatch settings, which include two cases: i) alignable mispairs, caused by instance-level permutations across views; and ii) unalignable mispairs, caused by noisy or corrupted samples. We control these two factors using the Mismatch Ratio (MR) and Corruption Ratio (CR), with detailed construction described in Appendix C.
Table 1 reports results under different MR. Our method consistently achieves the best performance, benefiting from its generative objective and robust correspondence discovery, which remain effective even with a few aligned pairs. Table 2 further evaluates scenarios with both alignable and unalignable mismatches. While all baselines degrade severely as MR and CR increase, our method maintains strong performance by jointly leveraging GMM-based marginals to down-weight noisy samples and virtual samples to absorb unalignable ones, mitigating the influence of low-quality pairs.
this section cite: []

Section: POSTERIOR DISTRIBUTION VISUALIZATION (Q2)
We next investigate whether CorreGen can uncover the latent correspondences across views. On Caltech101 with MR=0.2 and CR=0.0, we sample a mini-batch and estimate their posterior distributions at different training stages, comparing them with the true category-level ground truth.
As shown in Fig. 3, the category-level correlations are weak in the early training phase. By midtraining, the estimated posterior distributions already resemble the ground truth, and the gap further narrows in the later stages. These results demonstrate that CorreGen progressively uncovers the latent class-level correspondences, thereby effectively alleviating category-level mismatches.
this section cite: []

Section: CONCLUSION
In this paper, we propose a novel generative framework for multi-view clustering under the noisy correspondence challenge. Unlike existing discriminative approaches that rely heavily on off-theshelf pairwise alignments, our method models cross-view dependencies by maximizing the joint likelihood of observed data, thereby uncovering latent correspondences in a principled manner. Extensive experiments across multiple datasets demonstrate that our approach not only achieves supe- rior clustering performance but also exhibits strong robustness to sample-level and category-level mismatches. In the future, we plan to extend this framework to unpaired multi-modal learning and apply it to cross-modal retrieval tasks with large-scale noisy data.
this section cite: []

Section: References
Ref_id:b0 Title: Representation learning: A review and new perspectives Year: (2013)
Ref_id:b1 Title: Partial optimal tranport with applications on positive-unlabeled learning Year: (2020)
Ref_id:b2 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b3 Title: THESAURUS: contrastive graph clustering by swapping fused gromov-wasserstein couplings Year: (2025)
Ref_id:b4 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b5 Title: A bayesian hierarchical model for learning natural scene categories Year: (2005)
Ref_id:b6 Title: Learn from global rather than local: Consistent context-aware representation learning for multi-view graph clustering Year: (2025)
Ref_id:b7 Title: Robust contrastive multi-view clustering against dual noisy correspondence Year: (2024)
Ref_id:b8 Title: Trusted multi-view classification Year: (2021)
Ref_id:b9 Title: Robust variational contrastive learning for partially view-unaligned clustering Year: (2024)
Ref_id:b10 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b11 Title: Partially view-aligned clustering Year: (2020)
Ref_id:b12 Title: Learning with noisy correspondence for cross-modal matching Year: (2021)
Ref_id:b13 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b14 Title: Cleannet: Transfer learning for scalable image classifier training with label noise Year: (2018)
Ref_id:b15 Title: Large-scale multi-view spectral clustering via bipartite graph Year: (2015)
Ref_id:b16 Title: Dual contrastive prediction for incomplete multi-view representation learning Year: (2022)
Ref_id:b17 Title: Graph matching with bi-level noisy correspondence Year: (2023)
Ref_id:b18 Title: Multi-granularity correspondence learning from long-term noisy videos Year: (2024)
Ref_id:b19 Title: Alleviate anchor-shift: Explore blind spots with cross-view reconstruction for incomplete multi-view clustering Year: (2024)
Ref_id:b20 Title: Decoupled contrastive multi-view clustering with high-order random walks Year: (2024)
Ref_id:b21 Title: Some methods of classification and analysis of multivariate observations Year: (1967)
Ref_id:b22 Title: Margin-aware noise-robust contrastive learning for partially view-aligned problem Year: (2025)
Ref_id:b23 Title: A survey on representation learning for multi-view data Year: (2025)
Ref_id:b24 Title: Anchor-guided sample-and-feature incremental alignment framework for multi-view clustering Year: (2025)
Ref_id:b25 Title: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning Year: (2018)
Ref_id:b26 Title: Robust multi-view clustering with noisy correspondence Year: (2024)
Ref_id:b27 Title: Roll: Robust noisy pseudo-label learning for multi-view clustering with noisy correspondence Year: (2025)
Ref_id:b28 Title: Contrastive multi-view subspace clustering via tensor transformers autoencoder Year: (2025)
Ref_id:b29 Title: Recipe recognition with large multimodal food dataset Year: (2015)
Ref_id:b30 Title: Partially view-aligned representation learning via cross-view graph contrastive network Year: (2024)
Ref_id:b31 Title: Noisy correspondence rectification via asymmetric similarity learning Year: (2025)
Ref_id:b32 Title: Visual transformers: Token-based image representation and processing for computer vision Year: (2020)
Ref_id:b33 Title: Unsupervised feature learning via nonparametric instance discrimination Year: (2018)
Ref_id:b34 Title: Gcfagg: Global and cross-view feature aggregation for multi-view clustering Year: (2023)
Ref_id:b35 Title: Partially multi-view clustering via re-alignment Year: (2025)
Ref_id:b36 Title: Deep multi-view learning methods: A review Year: (2021)
Ref_id:b37 Title: Partially view-aligned representation learning with noise-robust contrastive loss Year: (2021-06)
Ref_id:b38 Title: Learning with twin noisy labels for visible-infrared person re-identification Year: (2022)
Ref_id:b39 Title: Robust multi-view clustering with incomplete information Year: (2022)
Ref_id:b40 Title: Robust object re-identification with coupled noisy labels Year: (2024)
Ref_id:b41 Title: Liming Fang, and En Zhu. Cluster-guided contrastive graph clustering network Year: (2023)
Ref_id:b42 Title: Bag-of-visual-words and spatial extensions for land-use classification Year: (2010)
Ref_id:b43 Title: Unified and tensorized incomplete multiview kernel subspace clustering Year: (2024)
Ref_id:b44 Title: Incomplete multi-view clustering via diffusion contrastive generation Year: (2025)
