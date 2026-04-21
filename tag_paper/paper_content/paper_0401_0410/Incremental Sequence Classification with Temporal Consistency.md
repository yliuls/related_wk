Title: Incremental Sequence Classification with Temporal Consistency
Abstract: We address the problem of incremental sequence classification, where predictions are updated as new elements in the sequence are revealed. Drawing on temporaldifference learning from reinforcement learning, we identify a temporal-consistency condition that successive predictions should satisfy. We leverage this condition to develop a novel loss function for training incremental sequence classifiers. Through a concrete example, we demonstrate that optimizing this loss can offer substantial gains in data efficiency. We apply our method to text classification tasks and show that it improves predictive accuracy over competing approaches on several benchmark datasets. We further evaluate our approach on the task of verifying large language model generations for correctness in grade-school math problems. Our results show that models trained with our method are better able to distinguish promising generations from unpromising ones after observing only a few tokens.

Section: Introduction
Learning to classify a sequence x = (x 1 , . . . , x T ) into one of K classes is a fundamental problem in machine learning [44,16]. In this work, we focus on an incremental variant in which the sequence is revealed element by element, and a predictive model must provide predictions at every timestep. As a concrete example, consider a movie review, represented as a sequence of tokens.
A x1 touching x2 movie x3 . x4 It x5 is x6 full x7 of x8 emotions x9 and x10 wonderful x11 acting x12 . x13
We ask the question: Can we predict the sentiment of this review using only the first two or three tokens? More generally, can we learn a predictive model that accurately classifies any prefix x ≤t , consisting of the first t ≤ T elements of a sequence? This problem naturally arises in domains where there is a cost associated to waiting for the full sequence; In healthcare or finance, for example, the cost might be time or opportunity [45,34]. Recently, sequence classifiers have also been deployed as verifiers to improve applications of large language models (LLMs), where generating full sequences incurs a non-trivial computational cost [8,39,31].
A key property of the incremental classification problem is that any calibrated predictive model should be temporally-consistent. That is, the predictive class distribution given a prefix x ≤t should equal the expected class distribution given the extended prefix x ≤t+1 , where the expectation is taken over x t+1 . We take advantage of this temporal-consistency property to develop a novel loss function for training incremental classifiers (Section 2). Our approach shares many parallels with temporal-difference (TD) learning [37], an important class of methods in reinforcement learning (RL) [38]. Exploiting these parallels, we study a simple sequence model and demonstrate that optimizing our loss function can lead to substantial data-efficiency gains over the standard maximum-likelihood approach (Section 3).
We apply our method to text classification using decoder-only transformers (Section 4). We first evaluate it on four benchmark datasets and find that models trained with our temporal-consistency loss achieve higher predictive accuracy-both on prefixes and on full sequences. When fine-tuning models of the OPT family [49], the improvement in predictive performance is comparable to increasing the size of the base model by a factor 10. We then explore the emerging application of verifying LLM generations. On GSM8K math word problems [8], our method yields verifiers that accurately distinguish correct from incorrect generations after observing only a few tokens. We illustrate how this enables a more favorable trade-off between answer accuracy and computational cost.
Our approach is inspired by celebrated methods in RL and is rigorously grounded in theory. It is simple to implement, adds negligible computational overhead during training and inference, and improves predictive performance across all the datasets we evaluated. As such, we believe it will be valuable to machine-learning practitioners.
this section cite: ['b43', 'b15', 'b44', 'b33', 'b7', 'b38', 'b30', 'b36', 'b37', 'b48', 'b7']

Section: Related work
Early sequence classification [44,45,27] addresses a problem that is distinct but complementary to ours: determining when enough information has been observed to make a reliable prediction. In contrast, we focus exclusively on what to predict at each prefix. Our goal is to maximize classification accuracy across all timesteps, without addressing the decision-making aspect. We note that incremental classifiers trained with our method could serve as components in early classification architectures [34,4].
TD learning [37,38] leverages a temporal-consistency condition, closely related to ours, to learn value functions in RL. Classical TD learning is concerned with modeling the reward-to-go, a scalar quantity. Our work can be understood as extending the core idea underpinning TD learning to the multiclass classification setting. Distributional RL [29,2,9] shares some algorithmic features, but does not address classification. Cheikhi and Russo [7] recently analyzed the data-efficiency benefits of TD learning, and we build on their results in Section 3. Beyond scalar values, ideas from TD learning have also been applied to survival analysis [26,40] and early event prediction [48].
Incremental classifiers have recently been applied to verify LLM generations, either during posttraining [39,21] or at inference time [8,47,31], where they are referred to as token-level verifiers [8] or outcome-supervised reward models [39,21]. Verification is typically framed as a binary classification problem, with verifiers trained using a cross-entropy loss against the final observed outcome, a baseline we refer to as direct cross-entropy (DCE). Some prior work use soft targets, as we do, but these targets are still derived solely from observed outcomes [41,20]. A notable exception is Mudgal et al. [31], who frame verification as a regression problem and learn a token-level value function using TD learning. However, they rely on a squared loss, which is ill-suited to binary outcomes [18]. Our work bridges this gap by developing a TD-style approach tailored to classification.
this section cite: ['b43', 'b44', 'b26', 'b33', 'b3', 'b36', 'b37', 'b28', 'b1', 'b8', 'b6', 'b25', 'b39', 'b47', 'b38', 'b20', 'b7', 'b46', 'b30', 'b7', 'b38', 'b20', 'b40', 'b19', 'b30', 'b17']

Section: Direct and temporally-consistent estimators
In order to introduce our approach to learning incremental sequence classifiers, we shift our perspective slightly. Instead of a sequence of arbitrary elements, which we have denoted by x = (x 1 , . . . , x T ) in the introduction, we now consider a sequence of Markov states s = (s 1 , . . . , s T ). That is, we assume that the ground-truth distribution p(s, y) of states and class label satisfies
p(s t+1 | s t , . . . , s 1 ) = p(s t+1 | s t ), p(y | s t , . . . , s 1 ) = p(y | s t ),(1)
for any t < T , and for any t ≤ T and any y, respectively. This is not a restrictive assumption: By slight abuse of notation, one can write s t = x ≤t and p(s t+1 | s t ) = p(x t+1 | x ≤t ), and trivially satisfy these two Markov properties. The Markov-chain perspective will be useful to relate our developments to temporal-difference learning, and to provide intuition into the statistical benefits of our method in Section 3.
Notation and problem setup Let [M ] denote the set of consecutive integers 1, . . . , M . We are given a dataset of N labelled sequences D = {(s n , y n ) : n ∈ [N ]}, where y n ∈ Direct cross-entropy A natural approach for learning p θ (y | s t ) is to maximize the likelihood of the observed samples from p(y | s t ) in the dataset D, or equivalently, to minimize the cross-entropy relative to these samples. Given a labeled sequence (s, y), we define the following loss function, which penalizes deviations from the label y simultaneously across all t ≤ T .
ℓ DCE (θ; s, y) = - T t=1 log p θ (y | s t ) = T t=1 H δ y p θ (• | s t ) ,(2)
where H[p∥q] = -K k=1 p k log q k is the cross-entropy function. Given the full dataset D, we find
θ ⋆ DCE ← arg min θ n ℓ DCE (θ; s n , y n ).(3)
We call this approach direct cross-entropy (DCE), because the target distribution is the observed one-hot class label δ y . In LLM verification applications, this is the predominant approach to training token-level verifiers [8,39,21,41,20].
this section cite: ['b7', 'b38', 'b20', 'b40', 'b19']

Section: A family of temporally-consistent estimators
Intuitively, a drawback of the direct approach is that, for early states s t (with t ≪ T ), the training signal provided by observed label y is noisy. Indeed, the prediction p θ (y | s t ) needs to account for two sources of uncertainty, a) the uncertainty about how the remainder of the sequence s t+1 , . . . , s T will unfold, and b) the uncertainty about the label y given the last state s T . We make progress by observing that
p(y | s t ) = E p(st+1|st) [p(y | s t+1 )],(4)
for all y and all t < T , an identity that follows from (1). This identity captures a notion of temporal consistency: It states that the class distribution at step t is equal to the class distribution at step t + 1 on average. Driven by this observation, we propose the following loss function, which for t < T penalizes the temporal inconsistency relative to a reference model parametrized by θ ′ :
ℓ TC (θ; θ ′ , s, y) = H δ y p θ (• | s T ) + T -1 t=1 H p θ ′ (• | s t+1 ) p θ (• | s t ) .(5)
Comparing ( 2) and ( 5) carefully, we can think of this temporal-consistency (TC) loss as replacing the hard targets δ y by soft targets p θ ′ (• | s t+1 ) capturing the predictive distribution at the next state.
Given the full dataset D, we start with random parameters θ (0) , and iteratively solve
θ (i+1) ← arg min θ n ℓ TC (θ; θ (i) , s n , y n ).(6)
In Section 3, we study a tractable setting and show that this iteration converges to a fixed point θ ⋆ TC , and that the estimator is consistent, i.e., that p θ ⋆ TC (y | s t ) → p(y | s t ) as the dataset size N → ∞. We can extend the identity (4) to multiple steps, capturing the temporal consistency of class distributions across longer time spans (c.f. Appendix A.1). We use this to formulate a generalized temporal-consistency loss,
ℓ TC-λ (θ; θ ′ , s, y) = T t=1 H z t p θ (• | s t ) , z t = λ T -t δ y + (1 -λ) T -t k=1 λ k-1 p θ ′ • | s t+k , (7
)
where λ ∈ [0, 1] is a hyperparameter. In this loss function, the target z t providing the training signal is a weighted average of the predictive distributions k steps ahead, with exponentially decreasing weights. The larger λ is, the larger the influence of distant states is. 1 In fact, TC-λ generalizes both TC (5) and DCE (2). Setting λ = 0 recovers TC, while λ = 1 yields DCE. Throughout the paper, we refer to any model trained with the TC-λ loss with λ < 1 as temporally consistent.
this section cite: ['b1']

Section: Connections to temporal-difference learning
Our approach is inspired by temporal-difference (TD) learning, a key idea in reinforcement learning [37,38,28]. Quoting Sutton's seminal paper, "whereas conventional prediction-learning methods assign credit by means of the difference between predicted and actual outcomes, the new methods assign credit by means of the difference between temporally successive predictions." Recasting our developments into the language of reinforcement learning, we can think of the probabilistic classifier we learn as a state-value function, capturing the eventual final outcome of a trajectory at any given intermediate state. The temporal-consistency condition (4) is a form of Bellman equation, relating the value function across successive states. Similarly to TD learning, our approach uses the temporal inconsistency of predictions across successive states as the learning signal. Our generalized loss function (7) is inspired by the TD(λ) family of algorithms.
A key difference is that we consider categorical outcomes and a cross-entropy loss, as opposed to the scalar outcomes and squared loss usually employed in TD learning algorithms. In Section 4, we compare our approach to classical value-estimation methods from RL, by treating our K-way classification problem as K separate value estimation problems.
this section cite: ['b36', 'b37', 'b27', 'b6']

Section: Convergence, consistency and data efficiency
In this section, we analyze the DCE and TC estimators in problems with a finite number of states. We consider tabular models, where a separate probability is learned for each state-class pair. This tractable setting enables a theoretical comparison of the properties of DCE and TC. Specifically, we show that a) the TC optimization procedure in (6) converges and b) yields a consistent estimator of the true class probabilities, and that c) the TC estimator is more data-efficient than DCE. Complete proofs of all propositions are provided in Appendix A.2. Our perspective in this section is partly inspired by dynamic programming and its application to stochastic shortest path problems, as presented in Bertsekas and Tsitsiklis [3, Sec. 2.2].
In the finite-state case, we formalize the problem of multiclass sequence classification as that of finding the absorption probabilities of a Markov chain on M transient states and K absorbing states [32]. The transient states correspond to M distinct values each element of the sequence s can take, and the absorbing states correspond to K classes. The Markov chain is fully characterized by the pair (Q, R), where the M × M matrix Q describes the transition probabilities between every pair of transient states, and the M × K matrix R describes the transition probabilities from transient to absorbing states. We assume that it is possible to reach a terminal state from any transient state within a finite number of steps. That is, for any transient state m, there is a t ≥ 0 such that [Q t R] mk > 0 for some k. Our goal is to estimate the absorption probabilities p ⋆ mk = p(y = k | s t = m) from data, organized into the M × K matrix P ⋆ . Before addressing the estimation problem, note that if the ground-truth transition matrices Q and R are known, P ⋆ can be computed by starting from an initial guess P 0 and refining it iteratively as
P i+1 = QP i + R.(8)
Proposition 1. For any M × K row-stochastic P 0 , the fixed-point iteration (8) converges to P ⋆ .
Now consider the setting where, instead of access to Q and R, we are given a dataset D = {(s n , y n ) : n ∈ [N ]} of trajectories sampled from the Markov chain. Each trajectory is composed of a transient sequence s n and terminates in an absorbing state y n . We explore two approaches to estimating the absorption probabilities.
this section cite: ['b31']

Section: Direct estimation
We can estimate the absorption probabilities directly, as the empirical fraction of sequences passing through m ending in k. Denoting by T (s) the length of sequence s and letting D ′ = ∪ (s,y)∈D ∪ t≤T (s) {(s t , y)} be the union of all pairs of transient state and eventual absorbing state in D, we have
pdir mk = E (s,y)∼D ′ [1 {y=k} | s = m],
where (s, y) ∼ D ′ denotes uniform sampling on D ′ . We collect these estimates into the matrix P dir . Indirect estimation Alternatively, we can first estimate the matrices Q and R by using the empirical one-hop transition counts. Letting A = ∪ (s,y)∈D ∪ t<T (s) {(s t , s t+1 )} and B = ∪ (s,y)∈D {(s T (s) , y)} be the union of all transitions between successive transient states and between transient and absorbing states, respectively, and T = A ∪ B, we write
s 1 1 s 2 1 s W 1 s 1 2 s 2 2 s W 2 s 1 T s 2 T s W T 0 1 • • • • • • • • • • • • • • • • • • • • • • • • • • • • • • • • •
qmm ′ = E (s,s ′ )∼T [1 {s ′ =m ′ } | s = m], rmk = E (s,s ′ )∼T [1 {s ′ =k} | s = m].(9)
We can then plug the estimates Q and R into the fixed-point iteration (8), and iterate until convergence. We denote the resulting estimate by P ind . By Proposition 1, this fixed point corresponds to the exact absorption probability matrix of the empirical Markov chain ( Q, R), but it is only an approximation of the ground-truth the absorption probabilities.
The following proposition relates the TC loss (5) and its iterative optimization procedure (6), introduced in Section 2, to the indirect estimator P ind we have just derived.
Proposition 2. Let p θ (y = k | s t = m) . = θ mk . Then, the TC iterative optimization procedure (5) is equivalent to the fixed-point iteration (8) on the empirical Markov chain ( Q, R) defined by (9).
It follows that, in the tabular case, the TC estimator (6) is guaranteed to converge. Furthermore, under mild assumptions on the data-generating distributionfoot_1 , the TC estimator is consistent, i.e.,
P ind → P ⋆ as N → ∞.
this section cite: ['b7', 'b8']

Section: Similarly, we can relate the optimization of the DCE loss (2) to the direct estimator P dir (see Proposition 4 in Appendix A).

this section cite: []

Section: Statistical benefits
In general, DCE and TC will not result in the same estimate, i.e., P ind ̸ = P dir , raising the question: Which estimator is better? In related work, Cheikhi and Russo [7] study direct and indirect estimators for the state-value function of a Markov reward process. They find that, depending on the structure of the Markov reward process, the indirect estimator can be significantly more data-efficient than (and is always at least as efficient as) the direct estimator.
In Figure 1, we revisit one of their examples and adapt it to our classification setting. We consider an absorbing Markov chain with T layers of W distinct states each. For any t < T , the probability of transitioning from a state at layer t to any other state at layer t + 1 is 1/W . From any state at layer T , we transition to one of two absorbing states, 0 and 1, with probability 1/2 each. Given this, it is easy to verify that p ⋆ mk = 1/2 for any transient state m and any absorbing state k ∈ {0, 1}. We collect N trajectories of length T + 1, sampling the first state uniformly at random, and subsequent states as described previously. The next proposition shows that, for any state in the first layer, the expected squared error of the indirect estimator is W times smaller than that of the direct estimator.
Proposition 3 (Adapted from [7]). For any T ≥ 1 and any state m in the first layer,
E (p ind mk -p ⋆ mk ) 2 E (p dir mk -p ⋆ mk ) 2 N →∞ ----→ 1/W.
We validate this result empirically in Figure 1 (right), where we report on numerical simulations using N = 20W . In this Markov chain, the indirect estimator is increasingly more data-efficient as W increases. Intuitively, the indirect approach acts as a form of regularization, by requiring the solution to satisfy the temporal-consistency property (4), which arises from the problem's Markov structure.
The indirect approach benefits from a form of data pooling: By minimizing the inconsistency across successive states instead of regressing the target outcome directly, we take advantage of information from other trajectories that originated from different states and crossed paths.
Beyond the tabular setting In the remainder of this paper, we move beyond the tabular setting studied above and fine-tune parametric large language models on very large state spaces. The theoretical results developed in this section no longer strictly apply, yet we believe that the underlying intuition remains valuable for interpreting our method's performance on complex, real-world tasks.
In text classification, for example, many distinct word sequences can result in a similar semantic state predictive of the target label. Optimizing for consistency across successive states can therefore improve data efficiency, similarly to the tabular toy example. With temporal consistency, the model implicitly exploits information from sequences that begin differently but reach similar intermediate semantic states, the same data-pooling phenomenon underpinning the gains of the indirect method in Figure 1.
this section cite: ['b6', 'b6']

Section: Empirical evaluation
In this section, we apply our methodology to text classification with decoder-only transformers. First, in Section 4.1, we evaluate multiple different approaches to training incremental classifiers. We compare the predictive performance of models on four well-known text classification benchmarks. Then, in Section 4.2, we consider a concrete application to verifying LLM generations. We show that an accurate token-level correctness classifier enables solving grade-school math problems computationally more efficiently.
this section cite: []

Section: Model architecture & training
Decoder-only (i.e., causal) transformers [23,33] are particularly well-suited to incremental sequence classification. As the tth output of a decoder on an input sequence x depends only on the prefix x ≤t , we can compute predictions at every prefix efficiently, with a single forward inference pass. Throughout this section, we model class probabilities as
p θ (• | x ≤t ) = softmax(Ah t + b),(10)
where h t ∈ R D is the hidden vector at the last layer of a transformer at position t, and A ∈ R K×D and b ∈ R K are the parameters of a classification head. We start with a pre-trained language model, and jointly optimize (A, b) as well as all the parameters of the transformer, which we collectively refer to as θ. We make two minor practical adjustments with respect to the optimization procedure outlined in Section 2. First, we update the parameters using stochastic gradient updates. Second, we average the loss over all prefixes of each sequence, instead of summing them. This means that every sequence contributes to the loss equally, irrespective of its length. Appendix B describes the precise training procedure and documents how we select hyperparameters.
this section cite: ['b22', 'b32']

Section: Text classification datasets
We consider four text classification datasets, spanning tasks such as movie review sentiment prediction (IMDB [25]) and topic classification (OHSUMED [30], NEWSGROUPS [19], AG-NEWS [10]). The number of classes K ranges from 2 to 23. We provide summary statistics for all datasets in Appendix B.2, including the number of training and test samples and the distribution of document length. In addition to the standard setting, where the goal is to predict the class label given the full sequence x, we are interested in evaluating the performance of classifiers in the incremental setting, where we need to make a prediction after observing only a prefix x ≤t consisting of the first t tokens.
We fine-tune pre-trained language models from the OPT family [49]. Unless otherwise noted, we report results on the 125 M-parameter version of the family, with hidden size D = 768. Our primary Table 1: Predictive performance of incremental text classifiers on four datasets. We report the classification accuracy on 4-token and 16-token prefixes, and on full sequences (all tokens). We highlight the best and second-best performing models.
OHSUMED NEWSGROUPS IMDB AG-NEWS 4 16 all 4 16 all 4 16 all 4 16 all Most frequent 16.0 16.0 16.0 5.3 5.3 5.3 50.0 50.0 50.0 25.0 25.0 25.0 GPT 4o 31.5 54.0 57.5 7.5 11.0 80.4 58.0 67.0 94.3 77.4 87.4 88.3 Filtering 22.1 46.9 46.2 10.4 19.8 72.0 64.1 73.4 92.1 77.9 86.6 87.2 Last token 16.7 45.0 80.6 6.5 9.4 87.9 56.6 68.4 94.7 54.4 78.3 94.8 Specialist 24.5 59.8 80.6 23.6 47.3 87.9 59.8 73.3 94.7 77.8 91.8 94.8 Direct ℓ2 loss 30.3 65.0 80.4 19.7 29.9 88.3 63.6 74.7 94.3 80.0 92.6 94.7 LSTD(λ) 32.7 64.9 78.0 26.2 36.7 87.8 64.6 75.4 94.7 81.1 92.8 94.9 DCE 30.5 65.5 81.1 27.7 40.1 89.0 63.5 74.7 94.4 80.0 92.6 94.8 TC-λ (ours) 33.7 68.3 81.8 33.4 44.7 88.5 64.7 75.7 94.9 81.4 93.0 95.0
focus is on comparing models trained by using the DCE (2) and TC-λ (7) loss functions. Both of these approaches train a single model to classify prefixes of any length (including the full sequence).
In addition, we also consider the following baselines and competing methods.
Most frequent A naive baseline that always predicts the most frequent class in the training split.
this section cite: ['b24', 'b29', 'b18', 'b9', 'b48']

Section: GPT-4o
We design a simple prompt asking GPT-4o (version 2024-08-06) to classify a text, by giving the list of classes, one example for each class, and the text itself. We access GPT-4o through OpenAI's commercial API. Details are provided in Appendix B.2.
Filtering We fine-tune a class-conditional language model p θ (x | y) on the training data, by using a standard next-token prediction task. At test time, we use Bayes' rule to reverse the conditional probability as p(y | x) = Z -1 p θ (x | y)p(y), where p(y) is a prior distribution and Z = y p θ (x | y)p(y).
Last token Similarly to the DCE loss (2), we minimize the cross-entropy relative to the observed label. But unlike the DCE loss, we include only a single cross-entropy term per sequence, corresponding to the prediction based on the hidden vector h T at the last token (i.e., capturing the full sequence). To the best of our knowledge, this is the most widespread approach to text classification with decoder-only transformers [43,12,17].
Specialist This approach is similar to the last token method, but improves upon it by training a distinct, specialized model for each prefix length. Instead of full sequences, each model is trained exclusively on prefixes of the corresponding length.
Direct ℓ 2 loss This approach is similar to DCE but removes the softmax and replaces the crossentropy loss with a squared loss. This is analogous to standard offline Monte Carlo methods for value function estimation in RL [38,Chap. 5].
this section cite: ['b42', 'b11', 'b16', 'b37']

Section: LSTD(λ)
The offline temporal-difference value estimation method of Bradtke and Barto [5], recently applied to language modelling in Mudgal et al. [31]. This approach is similar to TC-λ, without the softmax and with a squared loss instead of the cross-entropy loss.
For TC-λ and LSTD(λ), we treat λ as a hyperparameter. Values for this and for all other hyperparameters are presented in Appendix B.2. The time it takes to train a TC-λ model is virtually indistinguishable (within 1%) from that used to train a DCE model, confirming that the overhead required to compute the soft targets {z t } in (7) is negligible compared to the cost of LLM forward and backward passes.
In Table 1, we report the predictive accuracy of classifiers on hold-out sequences, given prefixes of 4 tokens, 16 tokens, and full sequences. For DCE and TC-λ and the corresponding squared-loss methods, we report additional metrics and more prefix lengths in the appendix. As expected, the accuracy of every classifier increases as more tokens are available. However, even with only 4 or 16 tokens (often representing a small fraction of the full sequence), some approaches reach a non-trivial accuracy. We observe that TC-λ outperforms DCE and other approaches in all but two cases, where it ranks second. This supports the insights from Section 3.1: incorporating temporal-consistency into the loss function improves predictive performance, even on real-world sequences and with large parametric models. Gains are more substantial for short prefixes, but-perhaps surprisingly-optimizing for temporal-consistency also improves full-sequence classification accuracy. Relatedly, we observe that training incremental classifiers is beneficial even when the goal is only to classify full sequences. Indeed, both DCE and TC-λ outperform the last-token approach in this setting. This was noticed by Cobbe et al. [8], who suggest that including prefixes in the loss provides a useful auxiliary signal. Our findings reinforce this observation. Finally, we note that the filtering method generally underperforms approaches using a classification head, consistent with conventional wisdom that, for classification problems, discriminative models can be more effective than generative models [1].
Cross-entropy vs. squared loss When comparing DCE and TC-λ, which use a cross-entropy loss, to their squared-loss counterparts (direct ℓ 2 loss and LSTD(λ), respectively), we observe the following. Temporal consistency tends to improves performance regardless of the loss function, especially on partial sequences. Models trained with a cross-entropy loss perform significantly better than those trained with a squared loss on datasets with many classes (OHSUMED and NEWSGROUPS).
For binary or three-way classification tasks (IMDB and AG-NEWS), both loss functions yield a similar accuracy. However, Figure 6 in the appendix shows that methods optimizing a squared loss produce noticeably less well-calibrated predictive probabilities.
Increasing model size Focusing on the OHSUMED dataset and the DCE and TC-λ methods, we train OPT models with 125 M, 350 M, and 1.3 B parameters. Figure 2 shows the area under the ROC curve (ROC AUC) for predictions after 4, 8, and 16 tokens, as well as for full sequences. These results offer the following perspective on the benefits of temporal consistency: DCE requires a model that is approximately 10× larger to match the performance of TC-λ.
Varying the temporal-consistency parameter In Figure 3 (left), we show the accuracy of TC-λ models trained on OHSUMED with different values of λ. The setting λ = 1, corresponsing to DCE, is never optimal, but the best setting depends on the prefix length. When optimizing for full-sequence accuracy, we observe empirically (across the four datasets) that performance is maximized with an effective lookahead of 5-50 tokens, corresponding values of λ between 0.8 and 0.98.
this section cite: ['b4', 'b30', 'b7', 'b0']

Section: Beyond predictive performance
We examine the effects of optimizing for temporal consistency more closely. Given predictive distributions p t and p t+1 produced after seeing t and t + 1 tokens, respectively, we compute the divergence KL[p t+1 ∥p t ]. Figure 3 (right) shows this divergence, averaged of all successive prefixes of all sequences in the test set. This is the quantity that TC-λ explicitly optimizesfoot_2 , and unsurprisingly DCE models are significantly less consistent. While we view temporal consistency primarily as a means to improve predictive performance, stable and consistent predictions might be valuable in their own right, for example in decision-making systems [34]. E v a n n e e d e d $ 2 0 t o b u y t h e w a t c h . H e a l r e a d y h a d $ 1 2 . S o n o w h e n e e d s $ 2 0 -$ 1 2 = $ 8 . T h e a n s w e r i s 8 . 0.0 0.5 1.0
𝑝 𝜃 ( 𝑦 = 1 | 𝒙 ≤𝑡 ) E v a n h a s $ 1 2 + $ 1 = $ 1 3 . E v a n s t i l l n e e d s $ 2 0 - $ 1 3 = $ 7 .
T h e a n s w e r i s 7 .
1
Figure 4: Predicted probability of correctness for two generations of Qwen2.5-0.5B for the prompt David found $12 on the street. He then gave it to his friend Evan who has $1 and needed to buy a watch worth $20. How much does Evan still need?
this section cite: ['b33']

Section: Language model verification on GSM8K
Next, we consider the problem of using an LLM to solve math word problems. In seminal work, Cobbe et al. [8] propose a simple two-step procedure, consisting of a) sampling N generations from the LLM, and b) scoring them with a verifier model, ultimately selecting the generation with the largest score. This best-of-N approach was shown to significantly increase performance over a single generation, at the expense of a larger computational cost (due to sampling N generations). In this setting, incremental verifiers bring substantial benefits: If the verifier is able to accurately distinguish between correct and incorrect generations early on, we can focus computational resources on extending only the most promising generations. This idea has recently received significant attention [47,39,21,41,31,20]. In this section, we demonstrate that our TC-λ approach holds promise for learning better incremental LLM verifiers.
We study GSM8K, a dataset of grade-school math problems and their solutions [8]. For our experiments, we use Qwen2.5-0.5B, a pre-trained language model with 0.5 B parameters that is known to perform well on GSM8K for its size [46]. We proceed as follows. For each of the 7473 problems in the training set, we sample 32 generations with temperature 0.7. We label each generation based on whether or not the answer extracted from the generation matches the ground-truth solution. For each problem, we keep exactly one correct and one incorrect generation (sampled uniformly at random), resulting in a balanced dataset. We then train incremental verifiers by fine-tuning Qwen2.5-0.5B with a classification head, as in (10), using DCE and TC-λ (details provided in Appendix B.3). Figure 4 illustrates our setup: Given a problem, our model predicts the probability, token after token, that a generation will eventually produce the correct answer.
Predictive performance For this binary classification task, the ROC AUC metric is particularly relevant, as it reflects the probability that the model correctly ranks a randomly-selected correct generation higher than an incorrect one [15]. as a function of the number of generated tokens. We observe that TC-λ significantly outperforms DCE when the number of tokens is small. With only 8 tokens, TC-λ is almost 70% accurate in distinguishing correct from incorrect generations. Consistent with our findings on the text classification experiments, on this binary classification task, models trained with a squared loss perform almost identically to those trained with a cross-entropy loss in terms of ROC AUC. However, as shown in Figure 8 in the appendix, they produce significantly worse predictive probabilities.
this section cite: ['b7', 'b46', 'b38', 'b20', 'b40', 'b30', 'b19', 'b7', 'b45', 'b14']

Section: Application to test-time scaling
We illustrate the benefit of accurate early correctness predictions in a basic test-time scaling scenario, where we trade off answer accuracy against computational cost, measured by the total number of generated tokens. We propose a simple modification to the bestof-N method, which we call boosted best-of-N and which is a simplistic version of the speculative rejection method recently proposed by Sun et al. [36]. Sample 10 tokens for each of 2N independent generations, rank them using the TC-λ incremental verifier, and continue sampling the remaining tokens for the top-N generations until completion. Finally, apply the verifier again to the N completed generations and select the best one. Figure 5 (center & right) shows that our approach compares favorably to vanilla best-of-N and to majority voting [42]. For a given level of accuracy, the boosted approach requires 23%-33% fewer tokens.
this section cite: ['b35', 'b41']

Section: Limitations & future work
We have introduced TC-λ, a loss function for training incremental sequence classifiers that draws on insights from TD learning to improve predictive performance. Our empirical results focus on text classification with transformers, but our approach is architecture-agnostic and applicable to any sequence classification problem. Future work could explore its effectiveness on multimodal applications, such as predicting task success from video frames in robotics [11] and games [13].
Given a limited compute budget, we have prioritized small-scale experiments. This has enabled us to run comprehensive hyperparameter sweeps and to report performance averaged over multiple random seeds, increasing our confidence in the results. Our experiments suggest that TC-λ could benefit models at all scales (Figure 2). However, the effectiveness of temporally-consistent methods with models larger than 1.3 B parameters has not yet been systematically evaluated. Likewise, while our experiments on LLM verification and test-time scaling show promise, further evaluation is needed, particularly in combination with state-of-the-art approaches such as speculative rejection [36].
Finally, a promising direction for future work is applying TC-λ to multi-token prediction [35], which has been shown to improve LLM pre-training [14,22] and accelerate inference [6]. Importantly, calibrated multi-token predictive distributions should satisfy a temporal-consistency condition that is similar to ( 4), making this a natural fit for our approach.
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: ['b10', 'b12', 'b35', 'b34', 'b13', 'b21', 'b5']

Section: Theory Assumptions and Proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Complete proofs for each of the three propositions presented in Section 3 are provided in Appendix A.2. For space reasons, we were unable to provide proof sketches in the main body, but we made an effort to introduce the theoretical results gradually, in such a way that the reader should have an intuitive understanding of why each result holds.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental Result Reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We have attempted to document our experiments thoroughly, providing in the appendix many additional details that we deemed out of scope for the main text, such as specific dataset versions, hyperparameter configurations, the model selection procedure, and more. We intend to complement this information with a comprehensive code release upon publication.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All datasets used in our experiments are publicly available, as are the pretrained language models we build upon (OPT [49] and Qwen2.5 [46]). We intend to release code with instructions to support the reproduction of our main experiments upon publication.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: ['b48', 'b45']

Section: Experimental Setting/Details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: All these details are provided in Appendix B.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: All plots in the paper include error bars reflecting 95% confidence intervals across multiple random initializations, computed as 1.96 × standard error. Table 1 omits error measures for clarity, but the results corresponding to the DCE and TC-λ rows are also shown with error bars in Figure 6.
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

Section: Experiments Compute Resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: We provide this information in Appendix B.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have reviewed the NeurIPS code of ethics, and we believe our work conforms with it.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader Impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [No] Justification: Our paper proposes a general-purpose training method for improving incremental sequence classification, without targeting any specific application domain. While we acknowledge that downstream applications of such models may carry societal implications, we do not identify any impacts (positive or negative) that are specific to the contributions of this work.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
absorption probability p ⋆ m1 . To this end, we collapse the two absorbing states 0 and 1 into a single terminal state denoted by ∅. We instantiate the reward distribution as follows. For any transition between two transient states, the reward is always 0. For any transition between a state at layer T and the absorbing state ∅, the reward is 1 with probability 1/2 and zero otherwise.
We can now recast our problem by using the terminology of Cheikhi and Russo [7,Sec. 7]. For every transient state m we have
V (m) = p ⋆ m1 , V MC (m) = pdir m1 , V TD (m) = pind m1 .
We focus on states in the first and second layer. Given the sampling process we have defined in the main text, any given state in the second layer will appear in N/W trajectories in expectation, and the transition between any pair of first and second layer states will appear in N/W 2 trajectories in expectation. It follows that, for any state m in the first layer and any state m ′ in the second layer, the coupling coefficient and the inverse trajectory pooling coefficient are given by C(m, m ′ ) = 1/W, C(m) = 1/W, respectively. We can then apply Theorem 7.2 in Cheikhi and Russo [7] to obtain the desired result.
The next proposition relates the DCE loss (2) and the optimization problem (3), introduced in Section 2, to the direct estimator P dir presented in Section 3.
Θ ⋆ DCE ∈ arg min Θ n ℓ DCE (Θ, s n , y n ) = arg min Θ (s,y)∈D ′ H[δ y ∥θ s ] = arg min Θ m c m k pdir mk H[δ k ∥θ m ] = arg min Θ m c m H[ pdir m ∥θ m ],
where pdir m = [p dir mk ] ∈ R K , and where the last equality uses the linearity of the cross-entropy with respect to the target distribution. It follows that the cross-entropy is minimized if Θ = P dir .
this section cite: ['b6', 'b6']

Section: B Additional details on experimental evaluation
This section provides additional details on the experiments presented in the main paper. In Section B.1, we describe the precise procedure we employ to train and select models. In Section B.2, we provide more details on the text classificiation experiments of Section 4.1. In Section B.3, we provide more details on the verification experiments of Section 4.2.
this section cite: []

Section: B.1 Training, model selection & metrics
Algorithm 1 presents one step of the training loop for the TC-λ and DCE approaches. Note that DCE is obtained simply by setting λ = 1, as explained in Section 2.1. With respect to the iterative optimization procedure (6), we make two minor practical adjustments. First, we update the parameters using stochastic gradient updates. Second, we average the loss over all prefixes of each sequence, instead of summing them. This means that every sequence contributes to the loss equally, irrespective of its length.
We run our experiments on an a3-highgpu-8g instance on Google Cloud, with 208 vCPUs, 1872 GB of memory, and 8 NVIDIA H100 GPUs. We ensure that every experiment runs on a single Algorithm 1 TC-λ incremental classifier: single training step Require: minibatch B, parameters θ, temporal-consistency parameter λ, learning rate η 1: ℓ(θ) ← 0 ▷ Initialize the loss function. 2: for (x, y) ∈ B do ▷ Iterate over the minibatch.
3: T ← length(x) 4: z T ← δ y 5: for t = T -1, . . . , 1 do 6: z t ← λz t+1 + (1 -λ)p θ (• | x ≤t+1 ) 7: ℓ(θ) ← ℓ(θ) + 1 T T t=1 H[stopgrad(z t )∥p θ (• | x ≤t )] 8: return θη |B| ∇ θ ℓ(θ) ▷ Update the parameters.
GPU. For every dataset, we set the batch size to maximize GPU utilization. We use the AdamW optimizer [24] with a dynamic learning rate. The learning rate starts at zero, increases linearly over a warmup period, then decreases linearly to zero at the end of the optimization process. We vary the following hyperparameters:
• the number of training epochs,
• the maximum learning rate,
• the warmup period (as a ratio of the total number of training steps),
• the weight decay, and
• the temporal-consistency parameter λ (for TC-λ only).
We run experiments on a grid of hyperparameter configurations, and we select the configuration that maximizes the full-sequence predictive accuracy on a small dataset of held-out sequences. Finally, throughout section 4, we report the mean and standard deviation of the performance of the winning hyperparameter configuration on the full test set, across 10 training runs with different random seeds.
We consider three metrics to measure the quality of a model p θ (y | x) on held-out data. The average accuracy is the fraction of examples where arg max y p θ (y | x) matches the ground-truth label y ⋆ (higher is better). The average negative log-likelihood (NLL) is the empirical average of -log p θ (y ⋆ | x) (lower is better). The area under the ROC curve (ROC AUC) captures the model's ability to discriminate between a random positive and negative example; In the multiclass setting, we use the one-vs-rest macro-average version. A higher ROC AUC is better.
this section cite: ['b23']

Section: B.2 Text classification datasets
We evaluate models on the following four well-known text classification benchmarks. Summary statistics and licensing terms for each dataset are provided in Table 2.
OHSUMED The dataset contains abstracts of publications in medical journals [30]. The task is to categorize each abstract into one of 23 themes, corresponding to sub-categories of cardiovascular diseases. We use the archive ohsumed-all-docs.tar.gz available at https://disi.unitn.it/moschitti/corpora.htm.
this section cite: ['b29']

Section: NEWSGROUPS
The dataset contains newsgroup documents from 20 different newsgroups [19]. The task is to identify which newsgroup the document comes from. We use the version of the dataset hosted at https://huggingface.co/datasets/  google-research-datasets/newsgroup.
IMDB The dataset contains movie reviews from the IMDb website [25]. The task consists of identifying the sentiment of the review (positive or negative). We use the version of the dataset hosted at https://huggingface.co/datasets/stanfordnlp/imdb.
this section cite: ['b18', 'b24']

Section: AG-NEWS
The dataset contains short news articles [10]. The task is to identify the topic of each article. We use the version of the dataset hosted at https://huggingface.co/datasets/  fancyzhx/ag_news. The NEWSGROUPS, IMDB and AG-NEWS datasets are provided with separate train and test splits, which we reuse as-is. For OHSUMED, we create our own train and test splits, by partitioning the data uniformly at random.
We fine-tune pre-trained models from the OPT family [49], which are made publicly available under the OPT-175B license 4 . Table 3 provides the hyperparameter configurations for the fine-tuned OPT-125M models whose performance we report in Table 1 in the main text. These hyperparameters are found using the process outlined in Section B.1.
Figure 6 presents detailed results for DCE, TC-λ, and the corresponding squared-loss variants, Direct ℓ 2 loss and LSTD(λ), for prefixes of length 2 i , i = 0, . . . , 9, across three metrics. We report the mean and the 95% confidence interval of 10 independent seeds, but the standard error is too small to be visible on the plot. On all datasets, the TC-λ models outperform the DCE models on almost all metrics at almost all prefix lengths.
Compute resources Each experiment, consisting of training an evaluating a model, takes 5-90 minutes on a single GPU, depending on the dataset, the size of the model, and the number of epochs. We estimate that the total compute used for all the experiments performed in the paper, including the hyperparameter sweeps, amounts to approximately 2000 GPU hours.
this section cite: ['b9', 'b48']

Section: B.2.1 GPT-4o baseline
We present the templates used for prompting GPT-4o in Figure 7. We use the structured outputs API to ensure that the model always returns exactly one valid class label. For cost reasons, we sample of a subset of 200 examples uniformly at random from the test set for each dataset and each prefix length (4, 16, and all tokens), and restrict our evaluation to this subset. Approximately 0.7% of requests trigger the ChatGPT filters (mostly in the IMDB and NEWSGROUPS datasets). We simply omit the corresponding examples from the evaluation.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claims made in the abstract and in the introduction are supported by the remainder of the paper, and they respect the guidelines outlined below.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: In Section 3, when developing a theory for our approach, we call out specific assumptions that are likely not satisfied in real-world applications. However, the insights appear to be robust, as supported by the empirical results presented in Section 4. Section 5 discusses additional limitations of our work.
Guidelines: 11. Safeguards Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: This work does not involve the release of models or datasets that present a high risk of misuse. As such, no specific safeguards have been put in place.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We use publicly available pretrained models and benchmark datasets in our experiments. We have cited the original sources in the paper and have made a reasonable effort to ensure that all assets are used in accordance with their licenses and terms of use.
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

Section: New Assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA]
Justification: The paper does not introduce any new assets directly. We plan to release code to support reproducibility and will ensure it is adequately documented.
Guidelines:
• The answer NA means that the paper does not release new assets.
this section cite: []

Section: A Additional methodological details & proofs
In this appendix, we provide additional details relating to the material presented in Sections 2 and 3.
this section cite: []

Section: A.1 Generalized temporal-consistency condition
The temporal-consistency condition (4) can be extended to k-step transitions as follows. Slightly abusing notation and setting s T +1 ≡ • • • ≡ s T +k ≡ y, and p(y | s t ) = 1 {y=st} for t > T , we have
p(y | s t ) = E p(s t+k |st) [p(y | s t+k )],
for any y, any t and any k. This follows from the Markov properties (1).
this section cite: []

Section: A.2 Theoretical results
This sections provides proofs for Propositions 1, 2, and 3. It also introduces an additional result, Proposition 4, that is the analogue of Proposition 2 for DCE.
For an N -dimensional vector x, we denote the L 1 -norm as
∥x∥ 1 = N n=1 |x i |.
Similarly, for an N × M matrix X, we denote the induced ∞-norm as ∥X∥ ∞ = max N n=1 ∥x n ∥ 1 , where x n is the nth row of X.
Proof of Proposition 1. Denote by a t m the mth row of the matrix Q t R. By assumption, we know that there is a τ ∈ N ≥0 and a ε > 0 such that ∥a τ m ∥ 1 ≥ ε for all m. Let F : R M ×K → R M ×K be the linear operator defined by F (P ) = QP + R. We will show that F τ +1 is a contraction mapping in the induced ∞-norm, that is,
∥F τ +1 (P ) -F τ +1 (P ′ )∥ ∞ ≤ (1 -ε)∥P -P ′ ∥ ∞ ,
for any two M × K row-stochastic matrices P , P ′ . By the Banach fixed-point theorem, it follows that F admits a unique fixed point P ⋆ = lim t→∞ F t (P 0 ) for any initial P 0 .
By construction, the matrix U . = [Q R] is row-stochastic, and thus ∥U ∥ ∞ = 1 and ∥Q∥ ∞ ≤ 1. It follows that
Q τ U = Q τ +1 Q τ R is such that ∥Q τ U ∥ ≤ ∥Q∥ τ ∞ ∥U ∥ ∞ ≤ 1. Since every row of Q τ R has L 1 -norm at least ε, it must be that ∥Q τ +1 ∥ ∞ ≤ 1 -ε. It follows that ∥F τ +1 (P ) -F τ +1 (P ′ )∥ ∞ = ∥Q τ +1 (P -P ′ )∥ ∞ ≤ ∥Q τ +1 ∥ ∞ ∥P -P ′ ∥ ∞ ≤ (1 -ε)∥P -P ′ ∥ ∞ .
Proof of Proposition 2. Consider one step of the TC optimization problem (6), where we denote the tabular model by using the M × K matrix Θ = [θ m ]. Letting c m = (s,s ′ )∈T 1 {s=m} , we have
Θ (i+1) ∈ arg min Θ n ℓ TC (Θ, Θ (i) , s n , y n ) = arg min Θ (s,y)∈B H[δ y ∥θ s ] + (s,s ′ )∈A H[θ (i) s ′ ∥θ s ] = arg min Θ m c m k rmk H[δ k ∥θ m ] + m ′ qmm ′ H[θ (i) m ′ ∥θ m ] = arg min Θ m c m H[r m + q⊤ m Θ (i) ∥θ m ],
where rm = [r mk ] ∈ R K , and qm = [q mm ′ ] ∈ R M , and where the last equality uses the linearity of the cross-entropy with respect to the target distribution. It follows that the cross-entropy is minimized if Θ = QΘ (i) + R, which corresponds exactly to ( 8).
Proof of Proposition 3. The Markov chain of Figure 1 has exactly two absorbing states, and the absorption probabilities sum up to 1. As such, we can focus on the variance of the estimator for p ⋆ m1 . We will construct a Markov reward process whose state-value function V (m) is equivalent to the  6: Detailed results for OPT-125M models trained with DCE, TC-λ, and the corresponding squared-loss variants on the text classification datasets. Accuracy and ROC AUC: higher is better. Average NLL: lower is better. We report means and 95% confidence intervals over 10 runs (too small to be visible).  Figure 8: Predictive negative log-likelihood of incremental Qwen2.5-0.5B verifiers on GSM8K (mean and 95% CI over 10 runs).
this section cite: []

Section: Listing 1: System prompt

this section cite: []

Section: B.3 Language model verification on GSM8K
For the language model verification experiments, we use the Qwen2.5-0.5B pre-trained language model [46], which is publicly available under the Apache 2.0 license. The GSM8K dataset [8] is publicly available under the MIT license. Hyperparameter selection follows the procedure outlined for the text classification experiments in Section B.1, with one small change: instead of selecting the best hyperparameter configuration based on the full-sequence accuracy, we select it based on the full-sequence ROC AUC. Table 4 provides the hyperparameter configurations for the fine-tuned Qwen2.5-0.5B models whose performance we report in Figure 5 in the main text.
In these experiments, there is one important conceptual difference with respect to the text classification experiments of Section 4.1. The sequence we seek to classify consists of the generated tokens only, but the verifier also needs to access the prompt (i.e., the problem statement). We implement this by concatenating the generated response to the prompt. However, when computing the loss, we mask out the terms that correspond to tokens in the prompt.
Predictive NLL Figure 5 (left) in the main text shows that models trained with a squared loss achieve essentially the same ROC AUC as those trained with cross-entropy. Figure 8 further shows that models trained with a cross-entropy loss achieve substantially lower (i.e., better) negative loglikelihood. Thus, when accurate & calibrated predictive uncertainties are important, optimizing the cross-entropy objective performs better in practice.
this section cite: ['b45', 'b7']

Section: References
Ref_id:b0 Title: Bayesian Reasoning and Machine Learning Year: (2012)
Ref_id:b1 Title: A distributional perspective on reinforcement learning Year: (2017-08)
Ref_id:b2 Title: Neuro-Dynamic Programming Year: (1996)
Ref_id:b3 Title: CALIMERA: A new early time series classification method Year: (2023)
Ref_id:b4 Title: Linear least-squares algorithms for temporal difference learning Year: (1996)
Ref_id:b5 Title: Medusa: Simple LLM inference acceleration framework with multiple decoding heads Year: (2024-07)
Ref_id:b6 Title: On the statistical benefits of temporal difference learning Year: (2023-07)
Ref_id:b7 Title: Training verifiers to solve math word problems Year: (2021-10)
Ref_id:b8 Title: Distributional reinforcement learning with quantile regression Year: (2018-02)
Ref_id:b9 Title: Ranking a stream of news Year: (2005-05)
Ref_id:b10 Title: Vision-language models as success detectors Year: (2023-08)
Ref_id:b11 Title: Looking right is sometimes right: Investigating the capabilities of decoder-only LLMs for sequence labeling Year: (2024-08)
Ref_id:b12 Title: Building open-ended embodied agents with internet-scale knowledge Year: (2022-12)
Ref_id:b13 Title: Better & faster large language models via multi-token prediction Year: (2024-07)
Ref_id:b14 Title: The meaning and use of the area under a receiver operating characteristic (ROC) curve Year: (1982)
Ref_id:b15 Title: Deep learning for time series classification: a review Year: (2019)
Ref_id:b16 Title: Language models (mostly) know what they know Year: (2022-07)
Ref_id:b17 Title: Revisiting squared-error and cross-entropy functions for training neural network classifiers Year: (2005)
Ref_id:b18 Title: NewsWeeder: Learning to filter netnews Year: (1995-07)
Ref_id:b19 Title: Token-supervised value models for enhancing mathematical reasoning capabilities of large language models Year: (2025-04)
Ref_id:b20 Title: Let's verify step by step Year: (2024-05)
Ref_id:b21 Title: DeepSeek-V3 technical report Year: (2024-12)
Ref_id:b22 Title: Generating Wikipedia by summarizing long sequences Year: (2018-04)
Ref_id:b23 Title: Decoupled weight decay regularization Year: (2019-05)
Ref_id:b24 Title: Learning word vectors for sentiment analysis Year: (2011-06)
Ref_id:b25 Title: Temporally-consistent survival analysis Year: (2022-12)
Ref_id:b26 Title: Predicting grades Year: (2015)
Ref_id:b27 Title: Playing Atari with deep reinforcement learning Year: (2013-12)
Ref_id:b28 Title: Parametric return density estimation for reinforcement learning Year: (2010-07)
Ref_id:b29 Title: Complex linguistic features for text classification: A comprehensive study Year: (2004-04)
Ref_id:b30 Title: Controlled decoding from language models Year: (2024-07)
Ref_id:b31 Title:  Year: (1997)
Ref_id:b32 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b33 Title: TEASER: Early and accurate time series classification Year: (2020)
Ref_id:b34 Title: Blockwise parallel decoding for deep autoregressive models Year: (2018-12)
Ref_id:b35 Title: Fast best-of-N decoding via speculative rejection Year: (2024-12)
Ref_id:b36 Title: Learning to predict by the methods of temporal differences Year: (1988)
Ref_id:b37 Title: Reinforcement Learning: An Introduction Year: (2018)
Ref_id:b38 Title: Solving math word problems with process-and outcome-based feedback Year: (2022-11)
Ref_id:b39 Title: Deep end-to-end survival analysis with temporal consistency Year: (2024-10)
Ref_id:b40 Title: Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations Year: (2023-12)
Ref_id:b41 Title: Self-consistency improves chain of thought reasoning in language models Year: (2023-07)
Ref_id:b42 Title: HuggingFace's transformers: State-of-the-art natural language processing Year: (2020-10)
Ref_id:b43 Title: A brief survey on sequence classification Year: (2010)
Ref_id:b44 Title: Early classification on time series Year: (2012)
Ref_id:b45 Title: Qwen2.5 technical report Year: (2024-12)
Ref_id:b46 Title: FUDGE: Controlled text generation with future discriminators Year: (2021-06)
Ref_id:b47 Title: Temporal label smoothing for early event prediction Year: (2023-07)
Ref_id:b48 Title: OPT: Open pre-trained transformer language models Year: (2022-05)
