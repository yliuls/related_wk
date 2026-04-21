Title: REASONING WITH SAMPLING: YOUR BASE MODEL IS SMARTER THAN YOU THINK
Abstract: Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilities can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains.

Section: INTRODUCTION
Reinforcement learning (RL) has become the dominant paradigm for enhancing the reasoning capabilities of large language models (LLMs) (Guo et al., 2025;Hu et al., 2025). Equipped with a reward signal that is typically automatically verifiable, popular RL techniques have been successfully applied to posttrain frontier models, leading to sizeable performance gains in domains like math, coding, and science (Hendrycks et al., 2021;Li et al., 2022;Rein et al., 2024).
Despite the widespread empirical success of RL for LLMs, a large body of literature has centered around the following question: are the capabilities that emerge during RL-posttraining fundamentally novel behaviors that are not present in the base models? This is the question of distribution sharpening (He et al., 2025;Shao et al., 2025;Yue et al., 2025): that is, whether the posttrained distribution is simply a "sharper" version of the base model distribution, instead of placing mass on reasoning traces the base model is unlikely to generate.
Several works point towards the difficulty in learning new capabilities with RL-posttraining. He et al. (2025); Song et al. (2025) compare the pass@k (multi-shot) scores of base models with posttrained models, finding that for large k, base models actually outperform while the latter suffer from degraded generation diversity. In such cases, RL appears to redistribute pass@k performance to single-shot performance at the expense of multi-shot reasoning. Yue et al. (2025) also notes that the reasoning traces post-RL are tightly concentrated at high likelihoods/confidences under the base model, seemingly drawing from existing high-likelihood capabilities. We illustrate this point in our own experiments in Figure 4. Regardless, the advantage of RL-posttraining for single-shot reasoning has remained, as of yet, undeniable.
In this paper, we present a surprising result: sampling directly from the base model can achieve single-shot reasoning capabilities on par with those from RL.
We propose a sampling algorithm for base models that leverages additional compute at inference time, achieving single-shot performance that nearly matches RL-posttraining on in-domain reason-
this section cite: ['b9', 'b12', 'b19', 'b10', 'b29', 'b36', 'b10', 'b32', 'b36']

Section: MATH500
HumanEval GPQA ing tasks and can even outperform on out-of-domain reasoning tasks. Furthermore, we observe that generation diversity does not degrade with our sampler; in fact, our pass@k (multi-shot) performance strongly outperforms RL. We benchmark specifically against Group Relative Policy Optimization (GRPO), which is the standard RL algorithm for enhancing LLM reasoning (Shao et al., 2024).
Crucially, our algorithm is training-free, dataset-free, and verifier-free, avoiding some of the inherent weaknesses of RL methods including extensive hyperparameter sweeps to avoid training instabilities, the need to curate a diverse and expansive posttraining dataset, and the lack of guaranteed access to a ground truth verifier/reward signal (Prabhudesai et al., 2025).
Our contributions can be summarized as follows:
i) We introduce the power distribution as a useful sampling target for reasoning tasks. Since it can be explicitly specified with a base LLM, no additional training is required. ii) We further introduce an approximate sampling algorithm for the power distribution using a Markov chain Monte Carlo (MCMC) algorithm that iteratively resamples token subsequences according to their base model likelihoods. iii) We empirically demonstrate the effectiveness of our algorithm over a range of models (Qwen2.5-Math-7B, Qwen2.5-7B, Phi-3.5-mini-instruct) and reasoning tasks (MATH500, HumanEval, GPQA, AlpacaEval 2.0). Our results show that sampling directly from the base model can achieve results on par with GRPO. In fact, for some out-of-domain tasks, our algorithm consistently outperforms the RL baseline. Moreover, over multiple samples, we avoid the collapse in diversity afflicting RL-posttraining, achieving the best of both worlds in terms of single-to-few-shot reasoning capabilities as well as sample diversity.
Our results collectively illustrate that existing base models are much more capable at single-shot reasoning than current sampling methods reveal.
this section cite: ['b30', 'b25']

Section: RELATED WORKS
Reinforcement learning for LLMs. RL has been instrumental in posttraining LLMs. Early on, RL with human feedback (RLHF) (Ouyang et al., 2022) was developed as a technique to align LLMs with human preferences using a trained reward model. Recently, RL with verifiable rewards (RLVR)
has emerged as a powerful new posttraining technique, where many works (Guo et al., 2025;Lambert et al., 2024;Hu et al., 2025;Zeng et al., 2025) discovered that a simple, end-of-generation reward given by an automated verifier could substantially enhance performance on difficult reasoning tasks in mathematics and coding. The Group Relative Policy Optimization (GRPO) algorithm was at the center of these advances (Shao et al., 2024). Building off of this success, many subsequent works have examined using reward signals derived from internal signals such as self-entropy (Zhao et al., 2025), confidence (Prabhudesai et al., 2025), and even random rewards (Shao et al., 2025). Similar to these works, our paper examines base model likelihoods as a mechanism for improving reasoning performance, but crucially, our technique is training-free.
Autoregressive MCMC sampling with LLMs. Prior works have explored integrating classic MCMC techniques with autoregressive sampling. Many settings including red-teaming, promptengineering, and personalized generation can be framed as targeting sampling from the base LLM distribution but tilted towards an external reward function. Zhao et al. (2024) proposes learning intermediate value functions that are used in a Sequential Monte Carlo (SMC) framework (Chopin, 2004), where multiple candidate sequences are maintained and updated according to their expected future reward. Similarly, Faria et al. ( 2024) proposes a Metropolis-Hastings (MH) algorithm, which instead of maintaining multiple candidates performs iterative resampling, again updating according to expected reward. Methodologically, our sampling algorithm is most similar to this latter work, but the crucial difference is that our target sampling distribution is completely specified by the base LLM, avoiding the need for an external reward.
this section cite: ['b24', 'b9', 'b16', 'b12', 'b37', 'b30', 'b40', 'b25', 'b29', 'b39', 'b3']

Section: PRELIMINARIES
Let X be a finite vocabulary of tokens, and let X T denote the set of finite sequences of tokens x 0:T = (x 0 , x 1 , . . . , x T ), where x i ∈ X for all i and T ∈ Z ≥0 is some nonnegative integer. For convenience, for a given t, let x <t = (x 0 , . . . , x t-1 ) and x >t = (x t+1 , . . . , x T ), with similar definitions for x ≤t and x ≥t . In general, x refers to a token sequence x 0:T , where T is implicitly given.
Then an LLM defines a distribution p over token sequences X T by autoregressively learning the conditional token distributions p(x t |x <t ) for all t, giving the joint distribution via the identity
p(x 0:T ) = T t=0 p(x t |x <t ).(1)
To sample a sequence from p, we simply sample from the LLM token by token using the conditional distributions, which by (1) directly samples from the joint distribution. Here p is a mixture of Gaussians, which we plot against p α (α = 4.0).
this section cite: []

Section: MCMC SAMPLING FOR POWER DISTRIBUTIONS
In this section, we introduce our sampling algorithm for base models. Our core intuition is derived from the notion of distribution sharpening posed in Section 1. Sharpening a reference distribution refers to reweighting the distribution so that high likelihood regions are further upweighted while low likelihood regions are downweighted, biasing samples heavily towards higher likelihoods under the reference. Then if RL posttrained models really are just sharpened versions of the base model, we should be able to explicitly specify a target sampling distribution that achieves the same effect.
We organize this section as follows. Section 4.1 presents this target sharpened distribution and provides some mathematical motivation for why its samples are amenable for reasoning tasks. Section 4.2 introduces a general class of Markov chain Monte Carlo (MCMC) algorithms aimed at actually sampling from this target distribution, and finally, Section 4.3 details our specific implementation for LLMs.
this section cite: []

Section: REASONING WITH POWER DISTRIBUTIONS
One natural way to sharpen a distribution p is to sample from the power distribution p α . Since
p(x) > p(x ′ ) =⇒ p(x) α p(x ′ ) α > p(x) p(x ′ ) (α ∈ [1, ∞]),(2)
it follows that exponentiating p increases the relative weight on higher likelihood sequences (x) while decreasing the relative weight on lower likelihood ones (x ′ ) (see Figure 2 for a visualization).
A related but well-known sharpening strategy is low-temperature sampling (Wang et al., 2020), which exponentiates the conditional next-token distributions at each step:
p temp (x t |x 0 . . . x t-1 ) = p(x t |x t-1 . . . x 0 ) α x ′ t ∈X p(x ′ t |x t-1 . . . x 0 ) α ,(3)
where the temperature is τ = 1/α. A common misconception is that sampling with (3) over T tokens is equivalent to sampling from p α ; however, this is false in a subtle yet crucial way, as we illuminate in the following.
Proposition 1. Low-temperature sampling does not sample from the power distribution p α .
Proof. We show that the associated conditional next-token distributions are distinct at each timestep t. The conditional distribution on x t for p α is given by
p pow (x t |x 0 . . . x t-1 ) = x>t p(x 0 , . . . , x t , . . . , x T ) α x ≥t p(x 0 , . . . , x t , . . . , x T ) α .(4)
Using Bayes rule
p(x t |x t-1 . . . x 0 ) = p(x 0 , . . . , x t ) p(x 0 , . . . , x t-1 ) = x>t p(x 0 , . . . , x t , . . . , x T ) x ≥t p(x 0 , . . . , x t , . . . , x T ) ,(5)
we can rewrite the low-temperature marginal (3) as
p temp (x t |x 0 . . . x t-1 ) = x>t p(x 0 , . . . , x t , . . . , x T ) α x ′ t x>t p(x 0 , . . . , x t , . . . , x T ) α .(6)
Ignoring normalizations for clarity, the relative weight on token x t for sampling from p α is given by a sum of exponents
p pow (x t |x <t ) ∝ x>t p(x 0 , . . . , x t , . . . , x T ) α .(7)
Meanwhile, the relative weight for low-temperature sampling is given by an exponent of sums
p temp (x t |x <t ) ∝ x>t p(x 0 , . . . , x t , . . . , x T ) α .(8)
Since the relative weights of next-token prediction are distinct for each sampling strategy, it follows that the joint distribution over seqeunces must also be distinct for each sampler. Hence, the distribution on sequences given by low-temperature sampling is not the same as the one given by p α .
One intuitive way to understand this difference is that low-temperature sampling does not account for how exponentiation sharpens the likelihoods of "future paths" at time step t, instead "greedily" averaging all these future likelihoods (exponent of sums (8)). On the other hand, sampling from p α inherently accounts for future completions as it exponentiates all future paths (sum of exponents (7)) before computing the weights for next-token prediction. This has the following consequence:
Observation 1. The power distribution upweights tokens with few but high likelihood future paths, while low-temperature sampling upweights tokens with several but low likelihood completions.
Example 1. We can observe this phenomenon with a simple example. Let us consider the token vocabulary X = {a, b} and restrict our attention to two-token sequences (x 0 , x 1 ): aa, ab, ba, bb. Let p(aa) = 0.00, p(ab) = 0.40, p(ba) = 0.25, p(bb) = 0.25, In other words, even though a has lower conditional likelihood under both p and p temp , p α upweights a and samples the highest likelihood two-token sequence. b has many future paths contributing to a higher likelihood under p and p temp , but leads to low likelihood sequences. We provide a stronger formalization of this phenomenon in Appendix A.2.
Thus, sampling from p α encourages sampling tokens which have fewer but higher likelihood "future paths", as opposed to tokens with several lower likelihood completions. This type of behavior is immensely valuable for reasoning tasks. For example, choosing "wrong" tokens that have high average likelihoods but trap outputs in low likelihood individual futures are examples of critical windows or pivotal tokens (Li et al., 2025;Abdin et al., 2024), a phenomenon where a few tokens are highly influential in the correctness of language model outputs. In fact, sharp critical windows have been shown to correlate strongly with reasoning failures (Li et al., 2025). Instead, embedded in sampling from the power distribution is an implicit bias towards planning for future high likelihood tokens.
this section cite: ['b33', 'b18', 'b0', 'b18']

Section: THE METROPOLIS-HASTINGS ALGORITHM
Now that we have seen how sampling from p α can in theory assist the underlying LLM's ability to reason, our aim now turns towards proposing an algorithm to accurately sample from it. Given an LLM p, we have access to the values p α over any sequence length; however, these values are unnormalized. Direct sampling from the true probabilities requires normalizing over all sequences (x 0 , . . . , x T ) ∈ X T , which is computationally intractable.
To get around this, we invoke a Markov Chain Monte Carlo (MCMC) algorithm known as Metropolis-Hastings (MH) (Metropolis et al., 1953), which targets exactly what we want: approximate sampling from an unnormalized probability distribution. The MH algorithm constructs a Markov chain of sample sequences (x 0 , x 1 , . . . , x n ) using an arbitrary proposal distribution q(x|x i ) to select the next candidate x i+1 . With probability
A(x, x i ) = min 1, p α (x) • q(x i |x) p α (x i ) • q(x|x i ) ,(9)
candidate x is accepted as x i+1 ; otherwise, MH sets x i+1 = x i . This algorithm is especially convenient as it only requires the relative weights given by p α (as the normalization weights in A cancel) and works with any generic but tractable sampler q with minimal restrictions. Remarkably, for large enough n, this process converges to sampling from the target distribution p α under the following (quite minimal) conditions on the proposal distribution (Neal, 1993):
Definition 1. The proposal distribution q is irreducible if for any set X with nonzero mass under the target distribution p α , q has nonzero probability of eventually sampling from X. The proposal is aperiodic if the induced chain of samples does not return to the same sample after a fixed interval number of steps.
Thus, we must simply ensure that our proposal distribution satisfies irreducibility and aperiodicity, and Metropolis-Hastings takes care of the rest. On a practical level, we would also like both q(x|x i ) and its reverse q(x i |x) to be easily computable.
Consider the following family of random resampling proposal distributions (see Figure 3). Let p prop be a proposal LLM. With uniform probability 1 T , select a random t ∈ [1, T ] and resample the sequence starting at index t using p prop . Then the transition likelihood q(x|x i ) is simply the likelihood of the resampling. Note that at each candidate selection step, we have a nonzero probability of transitioning between any two sequences x, x ′ ∈ X , since with some probability we can always resample as early as the beginning of x. This ensures our proposal distribution is both irreducible and aperiodic. Moreover, q(x i |x) is easy to calculate by symmetry, since we can treat x i as a resampled version of x.
With the flexibility endowed by Metropolis-Hastings, we can choose the proposal LLM p prop to be any LLM with any sampling strategy (e.g., low-temperature sampling).
this section cite: ['b21']

Section: POWER SAMPLING WITH AUTOREGRESSIVE MCMC
A direct implementation of Metropolis-Hastings for LLMs would involve initializing with a sampled token sequence of length T , subsequently generating new candidates of length T with (9) over many, many iterations. This process is computationally expensive, however, due to the repeated, full sequence inference calls to the LLM.
In fact, the main downside to MCMC algorithms in practice is the potential for an exponential mixing time (Gheissari et al., 2017), where a poor choice of initialization or proposal distribution can result in an exponentially large number of samples required before convergence to the target distribution. This problem is exacerbated if the sample space has high dimensionality (Bandeira et al., 2022;Schmidler & Woodard, 2013), which is precisely exhibited by the sequence space of tokens X T , especially for long sequences/large values of T .
To remedy this, we propose an algorithm that leverages the sequential structure of autoregressive sampling. We define a series of intermediate distributions which we progressively sample from, until converging to the target distribution p α . In particular, samples from one intermediate distribution initiate a Metropolis-Hastings process for the next, helping avoid pathological initializations.
Fix block size B and proposal LLM p prop , and consider the sequence of (unnormalized
) distributions ∅ -→ p(x 0 , . . . , x B ) α -→ p(x 0 , . . . , x 2B ) α -→ • • • -→ p(x 0 , . . . , x T ) α ,(10)
where p(x 0 , . . . , x kB ) denotes the joint distribution over token sequences of length kB, for any k. For convenience, let π k denote the distribution given by π k (x 0:kB ) ∝ p(x 0:kB ) α . (11) Suppose we have a sample from π k . To obtain a sample from π k+1 , we initialize a Metropolis-Hastings process by sampling the next B tokens x kB+1:(k+1)B with p prop . We subsequently run the MCMC sampling procedure for N MCMC steps, using the random resampling proposal distribution q from the previous section. The full details are presented in Algorithm 1.
Note that Algorithm 1 is single-shot: even though multiple inference calls are made, the decision to accept vs. reject new tokens is made purely by base model likelihoods to simulate sampling a single sequence from p α . We can interpret this as a new axis for inference-time scaling, as we expend additional compute during sampling to obtain a higher quality/likelihood sample.
To quantify the scaling, we can estimate the average number of tokens generated by Algorithm 1. Note that each candidate generation step when sampling from π k (x 0:kB resamples an average of kB 2 tokens, N MCMC times. Summing over all k, the expected number of tokens generated is
E tokens = N MCMC ⌈T /B⌉ k=1 kB 2 ≈ N MCMC T 2 4B . (12
) Algorithm 1: Power Sampling for Autoregressive Models Input : base p; proposal p prop ; power α; length T Hyperparams: block size B; MCMC steps N MCMC Output : (x 0 , . . . , x T ) ∼ p α 1 Notation: Define the unnormalized intermediate target π k (x 0:kB ) ∝ p(x 0:kB ) α . 2 for k ← 0 to ⌈ T B ⌉ -1 do 3
Given prefix x 0:kB , we wish to sample from π k+1 . Construct initialization x 0 by extending autoregressively with p prop :
x (0) t ∼ p prop x t | x <t , for kB + 1 ≤ t ≤ (k + 1)B. Set the current state x ← x 0 . 4 for n ← 1 to N MCMC do 5
Sample an index m ∈ {1, . . . , (k + 1)B} uniformly.
this section cite: ['b8', 'b1', 'b28']

Section: 6
Construct proposal sequence x ′ with prefix x 0:m-1 and resampled completion:
x ′ t ∼ p prop x t | x <t , for m ≤ t ≤ (k + 1)B. 7
Compute acceptance ratio ( 9)
A(x ′ , x) ← min 1, π k (x ′ ) π k (x) • p prop (x | x ′ ) p prop (x ′ | x) .
Draw u ∼ Uniform(0, 1); The key tradeoff here is between the block size B and number of MCMC steps N MCMC . A larger B requires larger "jumps" between intermediate distributions, requiring a larger N MCMC to adequately transition. In Section 5, we empirically find a value for B that makes Algorithm 1 performant for relatively small values of N MCMC .
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUP
Evaluation. We use a standard suite of reasoning benchmarks ranging across mathematics, coding, and STEM (MATH500, HumanEval, GPQA), along with a non-verifiable benchmark (AlpacaEval 2.0) evaluating general helpfulness. We evaluate all of our methods and baselines single-shot; i.e., on one final response string.
• MATH500: The MATH dataset (Lightman et al., 2024) consists of competition math problems spanning seven categories including geometry, number theory, and precalculus. There are 12500 problems total, with 7500 training problems and 5000 test problems. MATH500 is a specific randomly chosen subset of the test set standardized by OpenAI.
• HumanEval: HumanEval is a set of 164 handwritten programming problems covering algorithms, reasoning, mathematics, and language comprehension (Chen et al., 2021). Each problem has an average of 7.7 associated unit tests, where solving the problem corresponds to passing all unit tests.
• GPQA: GPQA (Rein et al., 2024) is a dataset of multiple-choice science questions (physics, chemistry, and biology) which require advanced reasoning skills to solve. We use subset GPQA Diamond for evaluation, which consists of 198 questions which represent the highest quality subset of the GPQA dataset.
• AlpacaEval 2.0: The AlpacaEval dataset is a collection of 805 prompts (Dubois et al., 2024) that gauge general helpfulness with questions asking e.g., for movie reviews, recommendations, We benchmark the performance of our sampling algorithm on MATH500, HumanEval, GPQA, and AlpacaEval 2.0. We bold the scores of both our method and GRPO, and underline whenever our method outperforms GRPO. Across models, we see that power sampling is comparable to GRPO on in-domain reasoning (MATH500), and can outperform GRPO on out-of-domain tasks.
and reading emails. The model responses are graded by an automated LLM judge (GPT-4-turbo), which determines a preference for the model responses over those from a baseline (also GPT-4turbo). The resulting score is a win rate of model responses normalized for the length of the model response.
this section cite: ['b5']

Section: Models.
To demonstrate the efficacy of our sampling algorithm, we use the base models Qwen2.5-Math-7B, Qwen2.5-7B, and Phi-3.5-mini-instruct. For our RL baselines, we use the implementation of GRPO in Shao et al. (2025), which posttrains these models on the MATH training split. For both the Qwen2.5 models, we use the default hyperparameters used to benchmark their performance in Shao et al. (2025). For the Phi-3.5 model, we use a set of hyperparameters selected from Abdin et al.
(2024) that avoids training instabilities and converges to improvement over the base model over a large number of epochs.
this section cite: ['b29', 'b29']

Section: Sampling Algorithm.
For our implementation of power sampling (Algorithm 1), we set the maximum T to be T max = 3072 (termination can happen earlier with an EOS token) and block size B = 3072/16 = 192. Empirically, we find α = 4.0 coupled with a proposal LLM p prop chosen as the base model with sampling temperature 1/α to be most performant for reasoning tasks. For Al-pacaEval 2.0, we find that having a proposal distribution of higher temperature (τ = 0.5) improves performance.
this section cite: []

Section: RESULTS
Main results. We display our main results in Table 1. Across base models of different families, our sampling algorithm achieves massive, near-universal boosts in single-shot accuracies and scores over different reasoning and evaluation tasks that reach, e.g., up to +51.9% on HumanEval with Phi-3.5-mini and +25.2% on MATH500 with Qwen2.5-Math. In particular, on MATH500, which is in-domain for RL-posttraining, power sampling achieves accuracies that are on par with those obtained by GRPO. Furthermore, on out-of-domain reasoning, our algorithm again matches GRPO on GPQA and actually outperforms on HumanEval by up to +59.8%. Similarly, power sampling consistently outperforms on the non-verifiable AlpacaEval 2.0, suggesting a generalizability of our boosts to domains beyond verifiability.
The surprising success of this fundamentally simple yet training-free sampling algorithm underscores the latent reasoning capabilities of existing base models.
Filter an input list of strings only for ones that start with a given prefix.
(Phi-3.5-mini-instruct: HumanEval) Method Response Passed Ours return [s for s in strings if s.startswith(prefix)] true GRPO return [string for string in strings if string.startswith(f'{prefix}' * 2)] false
Table 2: Sample responses on HumanEval: Phi-3.5-mini-instruct. We present an example where our method solves a simple coding question, but GRPO does not.
this section cite: []

Section: ANALYSIS
We analyze how the reasoning characteristics of power sampling relate to those of GRPO. We present an example in Table 2, with further examples in Appendix A.5.
Reasoning trace likelihoods and confidences. By design, power sampling targets sampling higher likelihood sequences from the base model. In Figure 4, the left graph plots a histogram of the output sequence log-likelihoods (averaged by length) of the base model, power sampling, and GRPO responses on MATH500, where likelihoods are taken relative to the Qwen2.5-Math-7B base model.
Our method samples from higher likelihood regions of the base model, as intended, but still maintains noticeable spread. Meanwhile, GRPO samples are heavily concentrated at the highest likelihood peak. We plot the log-likelihoods (relative to the base model) of original, power sampling, and GRPO responses over MATH500. Right: We do the same but for confidences relative to the base model. We observe that GRPO samples from the highest likelihood and confidence regions with power sampling close behind, which correlates with higher empirical accuracy.
We also plot the base model confidence of MATH500 responses, defined to be the average negative entropy (uncertainty) of the next-token distributions (Prabhudesai et al., 2025):
Conf(x 0:T ) = 1 T + 1 T t=0 x∈X p(x|x <t ) log p(x|x <t ). (13
)
The right plot of Figure 4 demonstrates that our method's and GRPO responses sample from similarly high confidence regions from the base model, which again correspond to regions of higher likelihood and correct reasoning.
this section cite: ['b25']

Section: Reasoning trace lengths.
Another defining characteristic of RL-posttraining is long-form reasoning (Guo et al., 2025), where samples tend to exhibit longer responses. On MATH500, Qwen2.5-Math-7B averages a response length of 600 tokens, while GRPO averages 671 tokens. Surprisingly, power sampling achieves a similar average length of 679 tokens, without explicitly being encouraged to favor longer generations. This emerges naturally from the sampling procedure.
Diversity and pass@k performance. Again, notice the peaked and highly concentrated likelihoods/confidences of GRPO relative to the distributional spread of power sampling in Figure 4. This suggests GRPO exhibits a collapse in diversity while our sampler does not, aligning with the obser-vation that RL-posttraining strongly sharpens the base model distribution at the expense of diversity (Song et al., 2025). To quantify the comparative diversity of power sampling relative to GRPO, we can plot the pass@k accuracy rate, where a question is solved if at least one of k samples is accurate. Figure 5 shows exactly this: unlike GRPO, whose pass@k performance tapers off for large k, power sampling strongly outperforms for k > 1. Moreover, our performance curve supersedes that of the base model until finally converging in performance. In particular, we are able to achieve GRPO-level single-shot performance without compromising multi-shot performance (see Appendix A.4 for other domains), addressing a long-standing downside to RL-posttraining. We plot the pass@k accuracy (correct if at least one of k samples is accurate) of power sampling (ours) and RL (GRPO) relative to the base model (Qwen2.5-Math-7B). Our performance curve is strictly better than both GRPO and the base model, and our pass rate at high k matches the base model, demonstrating sustained generation diversity.
this section cite: ['b9', 'b32']

Section: CONCLUSION
In this work, we present an algorithm that samples directly from a base model without any additional training or access to an external signal, achieving a single-shot reasoning performance that is on par with, and sometimes even better than, that of a state-of-the-art RL-posttraining algorithm. We use the discussion of RL distribution sharpening to motivate defining the power distribution as a valuable target distribution for reasoning. Although exact power distribution sampling is intractable, we employ classic MCMC techniques alongside the sequential structure of autoregressive generation to define our power sampling algorithm, which demonstrates strong empirical performance.
Our results suggest that base model capabilities are underutilized at sampling time and point towards a close relationship between high likelihood regions of the base model and strong reasoning capabilities. Employing additional compute at sampling-time with a stronger understanding of base model capabilities offers a promising direction for expanding the scope of reasoning beyond verifiability.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2024)
Ref_id:b1 Title: On free energy barriers in gaussian priors and failure of cold start mcmc for high-dimensional unimodal distributions Year: (2022)
Ref_id:b2 Title:  Year: ()
Ref_id:b3 Title: Central limit theorem for sequential monte carlo methods and its application to bayesian inference Year: (2004)
Ref_id:b4 Title: Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc Year: ()
Ref_id:b5 Title: Length-controlled alpacaeval: A simple way to debias automatic evaluators Year: (2024)
Ref_id:b6 Title: Quest: Quality-aware metropolis-hastings sampling for machine translation Year: (2024)
Ref_id:b7 Title: Proteina: Scaling flow-based protein structure generative models Year: (2025)
Ref_id:b8 Title: Exponentially slow mixing in the mean-field swendsen-wang dynamics Year: (2017)
Ref_id:b9 Title: Deepseek-r1: Incentivizing reasoning capability in LLMs via reinforcement learning Year: (2025)
Ref_id:b10 Title: Rewarding the unlikely: Lifting GRPO beyond distribution sharpening Year: (2025)
Ref_id:b11 Title: Measuring mathematical problem solving with the MATH dataset Year: ()
Ref_id:b12 Title: Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model Year: (2025)
Ref_id:b13 Title: Reguidance: A simple diffusion wrapper for boosting sample quality on hard inverse problems Year: (2025)
Ref_id:b14 Title: Test-time alignment of diffusion models without reward over-optimization Year: (2025)
Ref_id:b15 Title: Diffusion models as constrained samplers for optimization with unknown constraints Year: (2025)
Ref_id:b16 Title: Tülu 3: Pushing frontiers in open language model post-training Year: (2024)
Ref_id:b17 Title: Mcmc for multi-modal distributions Year: (2025)
Ref_id:b18 Title: Blink of an eye: A simple theory for feature localization in generative models Year: (2025)
Ref_id:b19 Title: Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with AlphaCode Year: (2022)
Ref_id:b20 Title: Let's verify step by step Year: (2024)
Ref_id:b21 Title: Equation of state calculations by fast computing machines Year: (1953)
Ref_id:b22 Title: Probabilistic inference using markov chain monte carlo methods Year: ()
Ref_id:b23 Title: Annealed importance sampling Year: (1998)
Ref_id:b24 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b25 Title: Maximizing confidence alone improves reasoning Year: (2025)
Ref_id:b26 Title: GPQA: A graduate-level google-proof q&a benchmark Year: ()
Ref_id:b27 Title: A parallel tempering algorithm for probabilistic sampling and optimization Year: (2014)
Ref_id:b28 Title: Lower bounds on the convergence rates of adaptive mcmc methods Year: (2013)
Ref_id:b29 Title: Spurious rewards: Rethinking training signals in RLVR Year: (2008)
Ref_id:b30 Title: Deepseek-math: Advancing mathematical reasoning through step-by-step exploration Year: (2024)
Ref_id:b31 Title: Feynman-kac correctors in diffusion: Annealing, guidance, and product of experts Year: (2025)
Ref_id:b32 Title: Outcome-based exploration for LLM reasoning Year: (2025)
Ref_id:b33 Title: Contextual temperature for language modeling Year: (2020)
Ref_id:b34 Title: Inference-time policy steering through human interactions Year: (2025)
Ref_id:b35 Title: Temporal score rescaling for temperature sampling in diffusion and flow models Year: (2025)
Ref_id:b36 Title: Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint Year: (2025)
Ref_id:b37 Title: Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild Year: (2025)
Ref_id:b38 Title: Inference-time scaling of diffusion models through classical search Year: (2025)
Ref_id:b39 Title: Probabilistic inference in language models via twisted sequential monte carlo Year: (2024)
Ref_id:b40 Title: Learning to reason without external rewards Year: (2025)
