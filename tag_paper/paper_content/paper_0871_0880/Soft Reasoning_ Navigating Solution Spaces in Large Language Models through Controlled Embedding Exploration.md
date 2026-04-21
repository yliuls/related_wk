Title: Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration
Abstract: Large Language Models (LLMs) struggle with complex reasoning due to limited diversity and inefficient search. We propose Soft Reasoning, an embedding-based search framework that optimises the embedding of the first token to guide generation. It combines (1) embedding perturbation for controlled exploration and (2) Bayesian optimisation to refine embeddings via a verifierguided objective, balancing exploration and exploitation. This approach improves reasoning accuracy and coherence while avoiding reliance on heuristic search. Experiments demonstrate superior correctness with minimal computation, making it a scalable, model-agnostic solution.

Section: Introduction
Large language models (LLMs) have demonstrated remarkable potential in various reasoning tasks, particularly on relatively simple and common benchmarks (Huang & Chang, 2023;Xu et al., 2025). Despite this, they still face significant limitations in complex tasks (Lightman et al., 2024;Wang et al., 2023a), which often require deeper levels of thought, and answers generated solely based on maximum likelihood are frequently incorrect. To increase the probability that the correct answer is included among generated candidates, many existing approaches aim to enhance generation diversity through multiple sampling (Lightman et al., 2024). A common mechanism for achieving such diversity is temperature scaling, which adjusts the randomness of token selection (Brown et al., 2024). Complementary to this, planning-based methods, such as chain-of-thought reasoning (Wei et al., 2022;Wang et al., 2023a) or treestructured search (Yao et al., 2023), attempt to locate the Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
correct answer by following language-based instructions.
Despite these efforts, two key challenges remain: (1) Enhancing generation diversity typically relies on increasing the temperature parameter, which flattens the token distribution. This, however, does not necessarily result in better coverage of the correct answer, as increasing low-probability token likelihood indiscriminately may introduce noise rather than meaningful exploration (Holtzman et al., 2020). (2) Existing planning and search methods such as sampling multiple reasoning paths rely heavily on heuristic strategies, guided by prompts (Hao et al., 2023;Qi et al., 2025b). However, these approaches do not directly adjust for the model's internal representations, thereby making the search process inefficient and highly dependent on surface-level prompt variations. This often leads to a "wild-goose chase", where search remains constrained by randomness and indirect heuristics rather than systematic optimisation.
To address these challenges, we propose Soft Reasoning, a novel approach using controlled embedding exploration: (1) By injecting a Gaussian embedding into the decoding of the first answer token, we can adjust the distribution of lowprobability tokens in a more controlled manner than uniform temperature tuning, leading to more flexible generation. (2) Treating the LLM as a black box verifier, we apply Bayesian optimisation (Frazier, 2018) on the injected embedding to maximise a verification-based reward. This allows us to use observerd rewards to directly guide the exploration in the embedding space. As a result, Soft Reasoning improves performance without a strong verifier-even when both generation and verification originate from the same model.
As illustrated in Figure 1, Soft Reasoning leverages injected vectors to change the distribution of the next generated token, rather than simply flattening the output probability curve. In this injection-based generation process, decoding is performed via greedy search, ensuring that each injected vector corresponds to a unique generated sequence. This guarantees both controllability and repeatability. The effect of each injected vector can then be evaluated using a reward function that accounts for both correctness and coherence of the generated sequence. Next, in a sequential way, we identify promising directions for further exploration based on all observed injection-reward pairs by utilising Bayesian optimisation. Again, thanks to the one-to-one mapping between initial vectors and generation outcomes, this process enables an effective search for optimal vectors that enhance generation quality. A notable advantage of this approach is that it operates without requiring access to the internal parameters of the LLM, allowing for efficient, low-resource control over generation behaviour.
Our contributions can be summarised as follows:
• We propose a reasoning method, Soft Reasoning, which combines embedding perturbation and Bayesian optimisation to better control low-probability token selection, enabling more flexible and diverse generation than temperature tuning.
• Instead of language-based instruction or heuristic search, our method directly optimises the embedding of the first generated token to control the direction of thinking and exploration in LLM, effectively reducing the searching and reasoning complexity, and improving efficiency and accuracy.
• Soft Reasoning is able to control and optimise the reasoning without accessing model parameters or requiring additional verifier, allowing seamless integration into different mainstream LLMs. Experiments across a variety of LLMs and reasoning tasks demonstrate improved correctness and efficiency of Soft Reasoning over traditional decoding.
this section cite: ['b12', 'b53', 'b23', 'b23', 'b2', 'b50', 'b56', 'b61', 'b11', 'b9', 'b61', 'b6']

Section: Related Work

this section cite: []

Section: Decoding Strategies and Diversity
Recent advances in LLM decoding aim to enhance diversity for tasks requiring creativity and exploration. Traditional methods such as greedy and beam search often produce repetitive outputs (Holtzman et al., 2020;Welleck et al., 2019), while sampling-based approaches (top-k, nu-cleus) introduce randomness but struggle to balance quality and diversity (Fan et al., 2018;Holtzman et al., 2020). High-temperature settings can lead to incoherent outputs (Minh et al., 2025), and adaptive methods like min-p sampling (Minh et al., 2025) require careful tuning. Debiasing-Diversifying Decoding (D3) mitigates amplification bias but increases computational cost (Bao et al., 2024). Crucially, most methods overlook the impact of initial token selection, which significantly influences reasoning outcomes (Wang & Zhou, 2024). Our approach addresses this by perturbing initial token embeddings with Gaussian noise, reshaping the probability distribution to improve exploration while maintaining quality and efficiency.
this section cite: ['b11', 'b51', 'b5', 'b11', 'b27', 'b27', 'b0']

Section: Efficient Exploration of Solution Spaces
Efficient solution space exploration is crucial for enhancing LLM reasoning while maintaining practical computational costs. Increasing generated samples improves coverage (Brown et al., 2024) but is computationally prohibitive. Optimising test-time compute allocation is more effective than scaling model size (Snell et al., 2025), though it requires task-specific strategies. Mutual reasoning frameworks leveraging self-play and MCTS (Qi et al., 2025b;Yan et al., 2024), as well as Tree of Thoughts (ToT) (Yao et al., 2023), explore multiple reasoning paths but incur high computational overhead. Thought Space Explorer (TSE) (Zhang & Liu, 2024) enhances reasoning breadth but at additional cost. Soft Reasoning refines these approaches by integrating controlled initial-token embedding perturbations with a strategic search algorithm inspired by MCTS and mutual reasoning. By introducing exploration early through embedding perturbation and guiding search via a verifier, we improve efficiency without excessive computational overhead, striking a balance between exploration and exploitation to optimise reasoning performance.
this section cite: ['b2', 'b40', 'b54', 'b56']

Section: Preliminary: Temperature Scaling
A common approach for generating diverse outputs is temperature scaling, which controls the randomness in the token generation process by modifying the softmax distribution over the model's output logits. For a given temperature τ > 0, the probability of selecting token w (t) at time step t is given by:
P (w (t) | w (1:t-1) ; θ, τ ) = exp(ℓ t,w (t) /τ ) w exp(ℓ t,w /τ ) ,
where w (1:t-1) represents the sequence of tokens {w (1) , . . . , w (t-1) } generated from the first token up to the (t-1) th token, ℓ t,w denotes the logit at time t corresponding to token w, and τ controls the sharpness of the distribution. This scaling flattens the distribution but preserves the relative ranking of token probabilities. When τ is low, the results concentrate on a few high-probability tokens, leading to overly deterministic generations with limited diversity. When τ is high, the model may sample low-probability tokens, leading to incoherent outputs.
While this approach increases diversity, it lacks control, blindly flattening token probabilities; adaptability, as it ignores verifier feedback; and efficiency, often requiring multiple samples or retraining (Joy et al., 2023;Xie et al., 2024). These limitations make it ineffective for structured reasoning tasks that demand precise and efficient exploration.
Generating accurate answers in complex tasks requires both exploring reasoning paths and verifying for their correctness. To achieve this, we propose a two-step framework as shown in Figure 2: (1) Embedding perturbation applies a Gaussian adjustment to the first-token embedding for controlled modifications beyond uniform tuning, (2) Bayesian optimisation refines the perturbed embedding to maximise a verifier-guided reward, improving reasoning path selection.
this section cite: ['b52']

Section: Embedding Perturbation
Given a generative model g θ and a natural language question prompt q, the first token w (1) is generated using greedy decoding, which selects the token with the highest probability from the model's predicted distribution:
w (1) = argmax w P (w | q; θ),
where P (• | q; θ) represents the probability distribution over the possible tokens predicted by the model g θ , with input q.
Let z ∈ R D represent the embedding of the token w (1) . This embedding serves as a prior, representing a "correct starting point" in the latent space. To explore the neighbourhood of this embedding, we define a set of perturbed embeddings x i for i = 1, . . . , k as follows:
x i = z + σε i , ε i ∼ N (0, I),
where ε i represents independent random perturbations drawn from a standard normal distribution, and σ is a scaling factor controlling the magnitude of the perturbation. This formulation allows us to sample from the local vicinity of the original embedding z, exploring variations around the initial token representation.
For each perturbed embedding x i , we introduce a corresponding special token mapped to x i and add it to the vocabulary. This special token is then used as the first token for generating an answer. Since we use greedy decoding, this x i fully determines the entire output sequence, i.e. the remaining tokens w
(2) i , w
i , . . . , w (L) i(3)
are then deterministically generated in a sequential manner:
w (t) i = argmax w P (w | x i , w (2:t-1) i , q; θ).
We denote y i := w (1:L) i to be the complete output based on the initial perturbed embedding x i . We then repeat this process k times to generate k different answers: y 1 , . . . , y k . Since each output y i is fully determined by x i , embedding perturbation effectively serves as a sampling mechanism over the entire answer space.
this section cite: []

Section: Exploring the Embedding Space
Randomly sampling points with infinite computational resources could theoretically approximate the optimal solution, but this approach is highly inefficient, especially given the computational expense of sampling with an LLM. Instead, we adopt Bayesian optimisation, which consists of two key components: an objective function and an acquisition function that determine where to sample next. We use Expected Improvement (EI) as our acquisition function, which offers a closed-form solution (Frazier, 2018) with negligible computational cost, making it significantly more efficient by comparison. EI effectively balances exploration (searching uncertain regions) and exploitation (refining promising areas), selecting the point with the highest EI at each iteration to guide the optimisation process toward convergence.
Optimisation Objective. To evaluate the objective function with k sampled perturbed embeddings, we consider the sequence x 1:k = {x 1 , . . . , x k }, where each x i ∈ R D . The corresponding answers are then generated as described above: y 1:k = {y 1 , . . . , y k }. Comparing and refining multiple generated answers has been shown to improve performance (Miao et al., 2024). Additionally, since LLMs are primarily trained for text generation rather than explicit judgment, prompting them to regenerate and compare outputs can yield better results (Zhang et al., 2024). Building on these insights, we propose a verifier-guided approach, where the model evaluates a batch of candidate answers and produces a refined output y v = V(y 1:k ). The correctness of
w 𝟏 (𝟏) w 𝟏 (𝟐) … w 𝟐 (𝟏) w 𝟐 (𝟐) … w k (𝟏) w k (𝟐) … 𝑦 1 𝑦 2 𝑦 𝑘 … … 𝑥 1 𝑥 2 𝑥 𝑘 … Diverse Outputs 𝑦 1:𝑘 Verifier r 𝑣𝑒𝑟𝑖𝑓𝑖𝑒𝑟 (𝑦 1 ) r 𝑣𝑒𝑟𝑖𝑓𝑖𝑒𝑟 (𝑦 2 ) r 𝑣𝑒𝑟𝑖𝑓𝑖𝑒𝑟 (𝑦 𝑘 ) 𝒓 𝒗𝒆𝒓𝒊𝒇𝒊𝒆𝒓 (𝒚 𝟏:𝒌 ) 𝒓 𝒄𝒐𝒉𝒆𝒓𝒆𝒏𝒄𝒆 (𝒚 𝟏:𝒌 ) + = 𝒇(𝒙 𝟏:𝒌 ) 𝒇 𝒙 | 𝒇(𝒙 𝟏:𝒌 ) LLM 𝜃 𝒘 (𝟏) 𝒘 (𝟐) … 𝑥 2 𝑥 1 𝑥 𝑘 𝑥 𝑖 = 𝑧 + 𝜎𝜀 𝑖 𝑬𝑰(𝒙) 𝑓 𝑥 𝐸𝐼 𝑥 Bayesian credible intervals for 𝑓(𝑥) Estimate of 𝑓(𝑥) • The point maximise 𝐸𝐼(𝑥) 𝑧 𝑥 2 𝑥 1 𝑥 𝑘 𝑧 Where to sample next 𝑥 𝑘+1 𝑥 𝑥
Embedding of 𝑤 (1)   Loop until convergence Starting with a natural language question prompt, the model generates initial token embeddings w (1) , which, due to greedy decoding, determine the entire output. These embeddings are perturbed to create candidate embeddings x 1:k , leading to outputs y 1:k through greedy search, which are then evaluated for coherence and verifier feedback. A Bayesian optimisation framework updates its estimation of the space based on this feedback and selects the next sampling point that maximises the expected improvement, balancing exploration and exploitation to refine the search for high-quality outputs.
an answer y is then assessed as a binary indicator (0 or 1), based on its alignment with the verifier's final output. The verifier is the same model as the generator employed.
The embedding space may not be uniform, implying that perturbations in different directions can lead to uneven semantic shifts (Li et al., 2023;Park et al., 2024). In some dimensions, even small perturbations can significantly alter meaning, potentially disrupting grammar or context consistency and leading to incoherent outputs. To address this issue, we introduce the coherence term to prune low-quality generations, ensuring that only outputs with desirable semantic and syntactic properties are retained. To evaluate the quality of a generated output y, we define an objective function f (x) that balances correctness and fluency:
f (x) = r verifier (y) + r coherence (y),(1)
where:
• Verifier Score (r verifier ): This is a binary indicator provided by the verifier, reflecting the correctness of y: r verifier (y) = 1 {yv=y} ;
• Coherence (r coherence ): This term evaluates the fluency of the generated sequence based on token probabilities: r coherence (y) =
T i=1 log P (w (i) ), where P (w (i) ) is the probability of generating token w (i) from the LLM's entire vocabulary. Bayesian Optimisation. Our goal is to maximise f (•), as defined in (1), over the embedding space R D . To optimise this black-box function, Bayesian optimisation uses a prior distribution on the domain to represent our beliefs about the behavior of the function and iteratively updates this prior using newly acquired data. Specifically, we model the prior joint distribution as a multivariate Gaussian distribution:
f (x 1:n ) ∼ N µ 0 (x 1:n ), Σ 0 (x 1:n , x 1:n ) ,
where µ 0 (x 1:n ) is the prior mean vector, and Σ 0 (x 1:n , x 1:n ) is the prior covariance matrix.
After observing f (x 1:k ), we aim to infer the value of f (x) at a new point x. Using Bayes' rule (Rasmussen & Williams, 2006), we update the posterior distribution of f (x) conditioned on these observed values:
f (x) | f (x 1:k ) ∼ N (µ k (x), σ 2 k (x)).(2)
Here, µ k (x) and σ 2 k (x) represent the posterior mean and variance, respectively. A detailed discussion on the choice of the prior distribution and the computation of the posterior distribution is provided in Appendix A.2.
A naive way to find the maximiser at this stage would be to select among the previously evaluated points x 1 , . . . , x k the one with the highest observed function value. Let f * k := max m≤k f (x m ) denote this value. If we were to sample another point x ∈ R D and observe f (x), then the value of the best observed point would either be f (x
) (if f (x) ≥ f * k ) or f * k (if f (x) < f * k ).
The improvement in the value of the best observed point could be expressed as [f (x) -f * k ] + := max(f (x) -f * k , 0). While we would ideally choose x to maximise this improvement, f (x) is unknown until after the evaluation. Instead, we select x that maximises the expected improvement under the posterior distribution, defined as
EI k (x) := E k [f (x) -f * k ] + ,(3)
where E k denotes the expectation taken with respect to the posterior distribution (2). Using integration by parts, we can write EI (3) in a closed-form expression:
EI k (x) = µ k (x) -f * k + + σ k (x)ϕ µ k (x) -f * k σ k (x) -µ k (x) -f * k Φ µ k (x) -f * k σ k (x) ,
where ϕ and Φ denote the probability density function and the cumulative distribution function of the standard normal distribution, respectively.
Our next sampling point, x k+1 ∈ R D , is the maximiser of EI. We then iteratively update the posterior distribution and the EI function. Details on how we select x to maximise EI k (x) can be found in Appendix A.3. Convergence is considered achieved when the change in the objective function between consecutive iterations satisfies
|f k -f k-1 | < ϵ,
where ϵ is a predefined threshold. Additionally, the algorithm terminates after a maximum of K iterations if convergence has not been reached.
In defining f (x), we assume an ideal verifier with perfect accuracy, meaning it provides an error-free assessment of correctness. However, in practice, the verifier's accuracy is less than 1, introducing uncertainty into its evaluations.
To address this noise in Bayesian optimisation, we use an adaptive version of the EI acquisition function that explicitly incorporates observation uncertainty. This adaptation dynamically adjusts the exploration rate based on uncertainty, ensuring a higher probability of convergence while balancing exploration and exploitation (Vakili et al., 2021;Tran-The et al., 2022). Theoretical foundations and implementation details are provided in Appendix A.4.
this section cite: ['b6', 'b21', 'b33', 'b38', 'b44', 'b42']

Section: Dimension Reduction.
One shortcoming of using traditional Bayesian optimisation methods for identifying the point with maximum EI (Mockus, 1975;Hvarfner et al., 2024) is that they perform poorly when the search space exceeds 20-30 dimensions due to the curse of dimensionality (Kandasamy et al., 2015;Letham et al., 2020;Wang et al., 2023b). In high-dimensional spaces, surrogate models require an exponentially larger number of points to accurately estimate the maximum of the EI function, making optimisation highly inefficient. With the dimension of embedding vectors for LLMs typically ranging from 768 to 8192 or more, traditional methods are impractical in our setting.
To address this, we leverage a dimension reduction approach based on random embeddings (Wang et al., 2016;Nayebi et al., 2019). Specifically, if a function f : R D → R has an effective dimension d e ≤ D, then with high probability, there exists a lower-dimensional representation g(u) := f (Au), where A is a random projection matrix. This allows optimisation to be performed in a lower-dimensional space R d instead of the original R D . Using this approach, we iteratively optimise the function in the reduced space and map solutions back to the original space. Theoretical foundations and implementation details are provided in Appendix A.5.
this section cite: ['b28', 'b14', 'b18', 'b20', 'b49']

Section: Experiments
We benchmark Soft Reasoning against strong baselines and conduct ablation studies.
this section cite: []

Section: Experimental Setup
Datasets and Models. We conduct experiments using three LLMs: Llama-3.1-8B-Instruct (Meta, 2024), Qwen2-7B-Instruct, Qwen2-70B-Instruct (Yang et al., 2024), and Mistral-8B-Instruct (Jiang et al., 2023). The models are evaluated on four benchmark datasets, including three complex mathematical tasks (GSM8K (Cobbe et al., 2021), GSM-Hard (Gao et al., 2023), SVAMP (Patel et al., 2021)), and one commonsense reasoning task StrategyQA (Geva et al., 2021). For the Qwen2-70B-Ins model, we additionally evaluate its performance on the AIME-2024 benchmark.
Baselines. Our baselines include: (1) CoT Prompting, which includes zero-shot CoT (Kojima et al., 2022) and few-shot CoT (Wei et al., 2022); (2) Self-Consistency (SC) Decoding (Wang et al., 2023c), which involves sampling answers at various temperatures τ ∈ {0.4, 0.6, 0.8} and selecting the final answer through majority voting;
(3) FIRE (Chen et al., 2025), which adjusts the decoding process by setting the temperature of the first token to 30 to enhance diversity, while subsequent tokens are generated using the standard temperature setting; (4) CoT-Decoding (Wang & Zhou, 2024), which generates k answers by sampling the top-k tokens from the probability distribution of the first token. Each of these top-k tokens is used as the starting point for decoding the remainder of the answer; and (5) RAP (Hao et al., 2023), which uses Monte Carlo Tree Search to explore reasoning paths strategically, balancing exploration and exploitation to find solutions efficiently. Note that RAP requires problem decomposition via examples; hence we only report its performance in the few-shot setting. Additionally, we compare Soft Reasoning with recent controlled generation approaches, including Trainable Prefix Scorers (Mudgal et al., 2024) and Constrained Fine-tuning (Qi et al., 2025a), with further details provided in Appendix B.2.
this section cite: ['b55', 'b16', 'b4', 'b7', 'b34', 'b19', 'b50', 'b3', 'b9', 'b29']

Section: Setup & Hyperparameters.
Experiments are conducted in zero-shot and few-shot settings, with prompts including 1, 2, 4, and 8 exemplars for few-shot settings. To reduce variance, each configuration is repeated five times with different random seeds. We report the mean and standard deviation of accuracy across all runs. The convergence threshold is set to 0.01.
this section cite: []

Section: Experimental Results
Overall Performance. Table 1 presents the accuracy of Soft Reasoning compared to baselines across four benchmarks and three LLMs under zero-shot and few-shot (8-shot) settings. The full table and the results for the Qwen2-70B-Ins model can be found in Tables 13 and 12, respectively, in Appendix B.8. Our approach consistently outperforms the best-performing baseline across different models, especially in the zero-shot setting (average improvement of 5% on GSM8K and 3% on GSM-Hard). Similar gains appear in the few-shot setting, where our method achieves the highest accuracy on most tasks and model variants. While effective, SC requires extensive hyperparameter tuning (e.g. varying temperature values) for each individual model and dataset to achieve optimal performance. In contrast, our more systematic search method improves solution quality consistently without the need for separate tuning in each scenario.
this section cite: []

Section: Coverage Analysis.
For each method, we calculate the probability of covering the correct answer in at least one of the generated answers. Our approach consistently achieves the highest coverage across all models and datasets. For instance, on GSM8K with LLaMA3-8B-Ins in the zero-shot setting, our method attains 91.8% coverage, outperforming FIRE (84.5%) and CoT-Decoding (85.3%). Detailed coverage probabilities for all models and datasets can be found in Table 14 in Appendix B.8. These results demonstrate that our controlled exploration strategy effectively enhances the likelihood of generating correct answers, highlighting its robustness over traditional methods.
this section cite: []

Section: Effect of Exploration with Embedding Perturbations and Bayesian Optimisation.
A natural question to consider is why adding noise to embeddings leads to more diverse answer generation than temperature tuning. We follow Naik et al. (2024) to investigate this from the perspective of neuron activations in the Transformer's MLP layers. As shown in Figure 3, applying our method increases the activation rate of neurons by roughly 3-4% in nearly all layers relative to the Self-Consistency (SC) baseline, suggesting that our perturbations stochastically trigger more diverse neural pathways.
To probe whether a specific subset of "critical neurons" may be responsible for correct reasoning, we identify neurons  in a single sample whose activations exhibit the strongest correlation with correctness. In Figure 4, we track the activation rate of these critical neurons across our Bayesian optimisation iterations. We observe a steady increase, particularly in layers 15-30, suggesting that our iterative sampling and verification increasingly activates these key pathways.
Together, these findings support the hypothesis that our embedding perturbation and controlled exploration approach not only diversifies generation but also systematically uncovers and reinforces the neuron activations crucial for deriving correct answers. For more detailed experimental procedures, please refer to Appendix B.3.
this section cite: []

Section: Convergence of Bayesian Optimisation.
Another question regarding our search algorithm is how quickly and reliably it converges. To investigate this, we track two key metrics across our Bayesian optimisation iterations: (1) The evolution of the correlation matrix among the sampled embedding points. As shown in Figure 5, the correlation matrix becomes more structured over iterations, showing higher correlations among top-performing candidates.
(2) The pro-portion of test examples that terminate after the n th iteration for each dataset in both zero-shot and few-shot settings, as reported in Table 2. With a maximum of 4 iterations, no search exceeds the fourth iteration, and only a small fraction require iteration 4. This rapid termination suggests that the EI-driven sampling strategy quickly identifies promising regions of the embedding space for most queries, minimising the need for further rounds of exploration.
Accuracy Correlation Matrix 1 st iteration 2 nd iteration Initialisation 3 rd iteration   Efficiency and Performance Analysis. Table 3 presents a comparison of our approach with RAP in both performance and efficiency. Our method achieves better accuracy on all tasks while drastically reducing computational overhead. Specifically, our input token consumption averages only 6.19% of RAP's, our output token usage is 63.28% of RAP's, and our inference time is 14.3% of RAP's. These results highlight that our method not only improves accuracy but also substantially reduces token usage, thereby delivering superior overall efficiency. Additional analysis of inference time and memory usage is provided in Appendix B.5.
this section cite: []

Section: Ablation studies
Objective Function. To evaluate the importance of each reward component, we removed either the verifier score (w/o r verifier ) or the coherence term (w/o r coherence ) from our method. Table 1 compare these ablated variants with our full approach. In all tasks and model configurations, omitting either term degrades both accuracy and coverage, indicating that both components are vital. The verifier score clearly helps filter out incorrect or spurious solutions, while the coherence penalty ensures each output remains semantically consistent, particularly for complex or multi-step reasoning. Indeed, both correctness-guided verification and semantic coherence play essential roles in navigating the solution space effectively.
Why Choose EI? There are various acquisition functions for Bayesian Optimization (BO), such as the UCB score, Probability of Improvement (PI), and GP-UCB. While PI and GP-UCB are viable alternatives to Expected Improvement (EI), the cumulative regret bound for GP-UCB matches that of EI (Shahriari et al., 2015). In contrast, PI considers only the probability of improvement and ignores its magnitude, making it less theoretically grounded and more prone to premature exploitation (Srinivas et al., 2010). As shown in Table 4, the experiments show that PI consistently underperforms compared to EI, as expected from the theoretical discussion above. For GP-UCB, its performance is sensitive to the choice of the exploration parameter and is, in most settings, worse than EI. We also note that the optimal parameter choice for GP-UCB varies across different tasks, making it difficult to guarantee good performance in unseen settings. In contrast, EI performs robustly without requiring task-specific tuning.
Optimisation Scope To investigate how the number of optimised tokens affects performance, we conducted additional experiments where we optimised embeddings for the first k tokens (instead of just the first token). As shown in Tab 5, the performance generally degrades as k increases, especially beyond 5 tokens. This suggests that naively extending to multiple tokens can introduce instability or overfitting. We also compared with RAP (one of our baselines), a tree-search-based method that operates at the sequence level rather than token-by-token-though it shares similar ideas with token-wise search. While RAP achieves strong performance, it incurs substantially higher cost and still underperforms our approach. Developing a multi-token optimisation strategy that can achieve both high accuracy and cost-effectiveness would require deeper investigation and extensive experimentation.
this section cite: ['b39', 'b41']

Section: Impact of Lower-Dimensional Space Dimensionality.
We evaluate our Bayesian optimisation approach under various reduced dimensions before mapping back to the full embedding space (Figure 6). Across all four tasks and both zero-and few-shot settings, performance tends to improve up to d = 50. Although increasing d to 60 sometimes yields a small additional gain, the differences are minor, and d = 50 consistently achieves near-best or best results. To further evaluate the robustness of random projection, we conducted additional stability experiments, which are presented in Appendix B.4.
this section cite: []

Section: Impact of Special Token Placement.
We compare three ways of inserting the perturbed special token into the prompt: at the beginning (First), somewhere in the middle (Middle), or as an appended token (Last). Table 6 shows for both zeroshot and few-shot settings, placing the special token at the end of the prompt (Last) generally yields higher accuracy and better coverage. One possible explanation is that placing the special token last ensures minimal disruption to the original semantics of the prompt, while still allowing Soft Reasoning to alter the initial token embedding and induce sufficiently diverse generation pathways. Based on this observation, we adopt the Last placement strategy in all subsequent experiments. Verifier Comparison: Judgement vs. Generation. Inspired by recent work suggesting that LLMs can be more adept at generating correct outputs than critiquing existing ones (Miao et al., 2024;Zhang et al., 2024), we explore four verifier strategies: Single-Judge, which evaluates each candidate independently; Single-Generate, which regenerates a purportedly correct answer for each candidate; Multi-Judge, which scores multiple candidates collectively; and Multi-Generate, which produces a new solution from multiple candidates, labeling any matching candidate as correct.
Given its consistently strong performance across settings, we select Multi-Generate as the default verifier in our experiments. The detailed definitions of these prompt templates are provided in Appendix B.6. Table 7 reports the binary classification accuracies for each verifier. Multi-Generate yields the highest verification accuracy on all datasets. This indicates that leveraging the model's generative capabilities leads to more reliable correctness assessment. We compare final solution accuracy in
Table 8. While single verifier judge or generate answers individually, multicandidate generation leads to the highest end-to-end performance. Multi-Generate outperforms alternatives in both zero-shot and few-shot settings, harnessing the model's generative capacity more effectively than judgment-based verifiers. Notably, even substituting simpler verifiers keeps our framework competitive with strong baselines, underscoring the robustness and efficacy of generation-based verification. Details on how the number of sampled embeddings k influences performance are provided in Appendix B.7.
this section cite: []

Section: Conclusions and Future Directions
We introduce an embedding-based optimisation framework that enhances LLM reasoning by refining the first-token embedding. By integrating controlled perturbations with Bayesian optimisation, Soft Reasoning improves accuracy, is model-agnostic, and remains computationally efficient.
Our approach relies on a verifier that may provide unreliable feedback, impacting optimisation. It also operates at the token level, which poses challenges for interpreting how perturbations influence reasoning. Future work will focus on improving verifier reliability, extending optimisation beyond the first token, and enhancing interpretability to better understand perturbation effects on reasoning.
this section cite: []

Section: References
Ref_id:b0 Title: Decoding matters: Addressing amplification bias and homogeneity issue in recommendations for large language models Year: (2024-11)
Ref_id:b1 Title: URL Year: ()
Ref_id:b2 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b3 Title: Flaming-hot initiation with regular execution sampling for large language models Year: (2025-04)
Ref_id:b4 Title: Training verifiers to solve math word problems Year: (2021-05)
Ref_id:b5 Title: Hierarchical neural story generation Year: (2018-07)
Ref_id:b6 Title: A tutorial on bayesian optimization Year: (2018)
Ref_id:b7 Title: Pal: program-aided language models Year: (2023)
Ref_id:b8 Title: Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies Year: ()
Ref_id:b9 Title: Reasoning with language model is planning with world model Year: (2023-12)
Ref_id:b10 Title: URL Year: ()
Ref_id:b11 Title: The curious case of neural text degeneration Year: (2020)
Ref_id:b12 Title: Towards reasoning in large language models: A survey Year: (2023-07)
Ref_id:b13 Title: URL Year: ()
Ref_id:b14 Title: Vanilla bayesian optimization performs great in high dimensions Year: (2024)
Ref_id:b15 Title:  Year: ()
Ref_id:b16 Title: Mistral 7b Year: (2023)
Ref_id:b17 Title: Sample-dependent adaptive temperature scaling for improved calibration Year: ()
Ref_id:b18 Title: High dimensional bayesian optimisation and bandits via additive models Year: (2015)
Ref_id:b19 Title: Large language models are zero-shot reasoners. Advances in neural information processing systems Year: (2022)
Ref_id:b20 Title: Re-examining linear embeddings for high-dimensional bayesian optimization Year: (2020)
Ref_id:b21 Title: Perturbscore: Connecting discrete and continuous perturbations in nlp Year: (2023-12)
Ref_id:b22 Title: URL Year: ()
Ref_id:b23 Title: Let's verify step by step Year: (2024)
Ref_id:b24 Title: Locating and editing factual associations in GPT Year: (2022)
Ref_id:b25 Title: Introducing meta llama3: The most capable openly available llm to date Year: (2024)
Ref_id:b26 Title: Using llms to zero-shot check their own step-by-step reasoning Year: ()
Ref_id:b27 Title: Turning up the heat: Min-p sampling for creative and coherent LLM outputs Year: (2025)
Ref_id:b28 Title: On bayesian methods for seeking the extremum Year: (1975)
Ref_id:b29 Title: Controlled decoding from language models Year: (2024)
Ref_id:b30 Title: Diversity of thought improves reasoning abilities of llms Year: ()
Ref_id:b31 Title: A framework for bayesian optimization in embedded subspaces Year: ()
Ref_id:b32 Title:  Year: (2019)
Ref_id:b33 Title: Not All Classes Stand on Same Embeddings: Calibrating a Semantic Distance with Metric Tensor Year: (2024-11-13)
Ref_id:b34 Title: Are NLP models really able to solve simple math word problems? Year: (2021-06)
Ref_id:b35 Title: URL Year: ()
Ref_id:b36 Title: Safety alignment should be made more than just a few tokens deep Year: ()
Ref_id:b37 Title: Mutual reasoning makes smaller LLMs stronger problem-solver Year: ()
Ref_id:b38 Title: Gaussian Processes for Machine Learning Year: (2006)
Ref_id:b39 Title: Taking the human out of the loop: A review of bayesian optimization Year: (2015)
Ref_id:b40 Title: Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning Year: (2025)
Ref_id:b41 Title: Gaussian process optimization in the bandit setting: no regret and experimental design Year: (2010)
Ref_id:b42 Title: Regret bounds for expected improvement algorithms in gaussian process bandit optimization Year: (2022)
Ref_id:b43 Title:  Year: ()
Ref_id:b44 Title: On information gain and regret bounds in gaussian process bandits Year: (2021-04)
Ref_id:b45 Title: Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models Year: (2023-07)
Ref_id:b46 Title: Chain-of-thought reasoning without prompting Year: (2024)
Ref_id:b47 Title: Recent advances in bayesian optimization Year: (2023)
Ref_id:b48 Title: Self-consistency improves chain of thought reasoning in language models Year: ()
Ref_id:b49 Title: Bayesian optimization in a billion dimensions via random embeddings Year: (2016)
Ref_id:b50 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b51 Title: Neural text generation with unlikelihood training Year: (2019)
Ref_id:b52 Title: Calibrating language models with adaptive temperature scaling Year: (2024)
Ref_id:b53 Title: Towards large reasoning models: A survey of reinforced reasoning with large language models Year: (2025)
Ref_id:b54 Title: Multiple-perspective self-reflection method for knowledge-rich reasoning Year: (2024-08)
Ref_id:b55 Title: Qwen2 technical report Year: (2024)
Ref_id:b56 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b57 Title: Thought Space Explorer: Navigating and Expanding Thought Space for Large Language Model Reasoning Year: (2024-12)
Ref_id:b58 Title: Self-contrast: Better reflection through inconsistent solving perspectives Year: (2024-08)
Ref_id:b59 Title: Prompt for Single-Judge Based on the given question and the previous answers, please provide your judgment on the correctness of the final answer. Question: Jessie currently weighs 9 kilograms Year: ()
Ref_id:b60 Title: Question: {User Question} Answer: {Previous answers} Correct: Prompt for Multi-Judge Based on the given question and the previous answers, please provide your judgment on the correctness of the final answer. Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Year: ()
Ref_id:b61 Title: Thought: The total amount of water is 2 liters = 2000 ml. The amount of salt is 20% of 2000 ml = 0.20 × 2000 ml = ⟨⟨0.20 × 2000 = 400⟩⟩400 ml Year: ()
Ref_id:b62 Title: liters. Since there are 1000 ml in 1 liter Year: ()
Ref_id:b63 Title: × 1000 = ⟨⟨0.4 × 1000 = 400⟩⟩400 ml Year: ()
Ref_id:b64 Title: Since Jack has 2 liters of seawater, he will get 0.2 × 2 = ⟨⟨0.2 × 2 = 0.4⟩⟩0.4 liters of salt. Since there are 1000 ml in 1 liter Year: ()
Ref_id:b65 Title: liters. There are 1000 ml in 1 liter Year: ()
Ref_id:b66 Title: × 1000 = ⟨⟨0.4 × 1000 = 400⟩⟩400 ml Year: ()
Ref_id:b67 Title: Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration Prompt for Single-Generate Based on the given question and the previous answers, please provide your analysis and final answer, starting the final answer with "Answer Year: ()
Ref_id:b68 Title: Let's think step by step. Jack has 2 liters of seawater, and 20% of it is salt. 2 liters = 2000 ml, so the amount of salt is 20% of 2000 ml = 0.20 × 2000 = 400 ml of salt Year: ()
