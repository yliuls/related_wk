Title: Predictable Scale (Part II) -Farseer: A Refined Scaling Law in LLMs
Abstract: Training Large Language Models (LLMs) is prohibitively expensive, creating a critical scaling gap where insights from small-scale experiments often fail to transfer to resource-intensive production systems, thereby hindering efficient innovation. To bridge this, we introduce Farseer, a novel and refined scaling law offering enhanced predictive accuracy across scales. By systematically constructing a model loss surface L(N, D), Farseer achieves a significantly better fit to empirical data than prior laws (e.g., Chinchilla's law). Our methodology yields accurate, robust, and highly generalizable predictions, demonstrating excellent extrapolation capabilities, outperforming Chinchilla's law, whose extrapolation error is 433% higher. This allows for the reliable evaluation of competing training strategies across all (N, D) settings, enabling conclusions from small-scale ablation studies to be confidently extrapolated to predict large-scale performance. Furthermore, Farseer provides new insights into optimal compute allocation, better reflecting the nuanced demands of modern LLM training. To validate our approach, we trained an extensive suite of approximately 1,000 LLMs across diverse scales and configurations, consuming roughly 3 million NVIDIA H100 GPU hours. To foster further research, we are comprehensively open-sourcing all code, data, results 3 , all training logs 4 , all models used in scaling law fitting 5 .

Section: 
As N deviates from the middle, the Chinchilla's error becomes divergent!  ), but it underestimates the requirements for larger scale regimes. In contrast, our analysis predicts a steadily increasing optimal D/N , which is consistent with the actual training configurations used in recent large language models (e.g., Llama 3.1 [16], Qwen3 [45], etc.).
this section cite: ['b16', 'b45']

Section: Introduction
Recent remarkable progress in Large Language Models (LLMs) such as GPT-3 [6], GPT-4 [2], and Llama [39] is largely attributed to scaling laws, notably proposed by Kaplan et al. [22]. These laws demonstrate that model performance, typically measured by loss L, exhibits a predictable improvement trend as model parameters (N ) and training data (D) increase. This relationship follows power-law dynamics expressed as:
L(N, D) = N c N α N α D + D c D α D ,(1)
'DWHVHW6L]H'WRNHQV Subsequently, DeepMind's Chinchilla [20] proposed optimal compute-scaling strategies and a revised scaling law:
L(N, D) ≈ A N α + B D β + E.(2)
where all parameters other than N and D are fitted. While valuable, Chinchilla's formulation has limitations in modeling the interplay between model size and data scaling. Specifically, we argue that its term B D β , which describes how loss improves with data D, uses constant parameters B and β. This implies that a rate of improvement with data is uniform across all model sizes N , thereby lacking adequate modeling of N 's influence on data scaling dynamics. Consequently, Chinchilla's law tends to capture an average data scaling behavior across the N values used for fitting. As a result, it performs best for models near the midpoint of this range, but less accurately at the extremes of N , as shown in Fig. 1 (a). This characteristic inherently limits its extrapolation capabilities, especially for model sizes significantly different from its calibration set, making cost-effective prediction across arbitrary (N, D) surfaces a persistent challenge. These limitations in accurately predicting performance with existing scaling laws underscore a broader difficulty: the very exploration of superior scaling laws is severely hampered by a significant scaling gap. The immense computational cost of state-of-the-art LLM training (often > 10 25 FLOPs) means insights from affordable, small-scale experiments frequently fail to transfer to production scales. This "scaling gap" refers to the phenomenon where conclusions drawn from small-scale experiments do not consistently hold when scaling up to larger models. A more accurate scaling law can mitigate this by providing more reliable predictions, making small-scale exploratory experiments more valuable (see Appendix E for an example).
To bridge this gap, we introduce Farseer, a refined scaling law (Eq. 3) and an experimental methodology developed from training over 1,000 LLMs. Farseer employs Differential Piecewise and Multi-round Iterative fitting to model the loss surface L(N, D):
L(N, D) = e a3•N γ +b3 + e a2•N β +b2 • D -e a 1 •N α +b 1(3)
where all parameters other than N and D are fitted. Our key contributions are:
• Refined scaling law. We propose Farseer (Eq. 3), providing a significantly more accurate fit to empirical LLM data (Fig. 1 (a) and 2) through a novel fitting approach where data scaling effects are explicitly N -dependent.
• Superior extrapolation. Farseer enables reliable large-scale performance prediction from tractable small-scale experiments, effectively bridging the scaling gap.
• Improved compute guidance. Our analysis yields new, data-driven insights for optimal D/N allocation in modern LLM training, diverging from simpler heuristics (Fig. 1 (b)).
• Comprehensive open-sourcing. We release all models, data from ∼1,000 trained LLMs, detailed logs, and the Farseer code to foster further research.
this section cite: ['b5', 'b1', 'b39', 'b22', 'b20']

Section: Preliminaries

this section cite: []

Section: General Loss Formulation
Our Farseer method systematically samples L(N, D) via small-scale experiments (smaller N, D) and applies a mathematical scaling formulation to predict performance at significantly larger scales. This enables robust assessment of a training strategy's scaling potential.
For clarity, we decompose the loss L(N, D) as:
L(N, D) = E + L N (N ) + L D (D) + L N D (N, D)(4)
where E is a constant term, L N (N ) is the model-size-dependent loss component, L D (D) is the data-size-dependent loss component, and L N D (N, D) represents the interaction effect between N and D. Farseer focuses on determining the functional forms and parameters of L N (N ), L D (D) and L N D (N, D) using only small-scale experimental data.
this section cite: []

Section: Basic Settings
We train approximately 1,000 models with a standard language modeling objective [41,11,31]. The training data comprises a mix of web text, mathematical content, and code. This data is processed using a Byte Pair Encoding (BPE) [14] tokenizer with a vocabulary size of 65,536. We evaluated two distinct data mixtures following [39,25], with specific component weightings detailed in Appendix B.
Our model architecture design follows the Llama [39,16] design, using the AdamW optimizer [29] with β values of [0.9, 0.95], an epsilon of 10 -8 , a weight decay of 0.1, and a gradient clipping norm of 1.0. We set the parameters N and D for these models using a geometric progression with a common ratio of √ 2, and specific details can be found in the Appendix F. A visualization of the experimental (N, D) grid is provided in Fig. 7 (blue circles). Our learning rate schedule includes a linear warmup for the first 2,000 steps, followed by cosine decay to 1 × 10 -5 for the remainder of training. The model uses a fixed sequence length of 2,048 tokens. Ultimately, we define model size N by excluding embedding layer parameters (see Appendix G for details). Further elaboration on the metric computations, and considerations for optimal hyperparameter settings [25], including model aspect ratios, can be found in Appendix A. The high R 2 in (a) (0.9807) demonstrates a consistent power-law relationship between ∆ D L and D, so we adopt this form in our main analysis. The R 2 value is the average across all fitted lines. As detailed in Appendix I, the associated terms A(N ) and B(N ) further offer improved numerical behavior.
For evaluation, we utilize a high-quality, specially constructed validation set containing 30 million tokens. This dataset is entirely separate from our training data, ensuring that all validation samples are unseen. It comprises a diverse mix of web pages, books, and academic papers, rigorously filtered to be more distilled and of higher quality than the training corpus. Instead of validation loss, we employ Bits Per Character (BPC) as our primary evaluation metric to measure the model's compression efficiency on this validation set. Further details are provided in Appendix D.
this section cite: ['b41', 'b10', 'b31', 'b13', 'b39', 'b25', 'b39', 'b16', 'b29', 'b25']

Section: Methodology
This section summarizes our Differential Piecewise Fitting methodology for the performance surface L(N, D) = L D (D) + L N D (N, D) + E + L N (N ). Section 3.1 presents a differential analysis establishing the necessity of the interaction term L N D (N, D). Section 3.2 details the modeling of the combined data-dependent terms L D (D) + L N D (N, D) (Stages 1 & 2), guided by empirical observations to a power-law form in D. Appendix C provides the full algorithm. Finally, Section 3.3 describes fitting the model-dependent term E + L N (N ) (Stage 3). The overall procedure yields a fitted scaling law:
L(N, D) ≈ f B (N ; θ * B )D -f A (N ;θ * A ) + f L N (N ; θ * L N ).
this section cite: []

Section: Differential Analysis for the Existence of Higher-Order Interaction Term L N D (N, D)
To understand the contributions of the terms in Eq. ( 4), we employ a differential technique. We compute finite differences of the loss L(N, D):
• ∆ D L(N, D) = L(N, D) -L(N, λD), this difference primarily reflects changes in the datadependent terms:
∆ D L(N, D) ≈ [L D (D) -L D (λD)] + [L N D (N, D) -L N D (N, λD)].
This operation cancels E and L N (N ).
• ∆ N L(N, D) = L(N, D) -L(λN, D). Similarly, this difference highlights changes in model-size dependent terms:
∆ N L(N, D) ≈ [L N (N ) -L N (λN )] + [L N D (N, D) - L N D (N, λD)].
This cancels E and L D (D).
Empirical analysis of these differential terms, as shown in Fig. 3, compellingly demonstrates the necessity of a non-degenerate interaction term L N D (N, D):
• ∆ D L(N, D) exhibits systematic dependence on both N and D (Fig. 3 (a) and (b)).
• ∆ N L(N, D) similarly depends systematically on both N and D (Fig. 3 (c) and (d)).
This dual dependence confirms that L N D (N, D) cannot be simplified to depend only on N or only on D, nor can it be additively separated. This motivates a fitting approach that can capture such coupled interactions.
this section cite: []

Section: Functional Form for Data-Dependent Term L
D (D) + L N D (N, D)
This section details the fitting of the data-dependent terms L D (D) + L N D (N, D), guided by the differential analysis in Sec. 3.1, with the goal of deriving parametric forms for the functions f A and f B in Eq. 5. The insights from the differential analysis in Section 3.1 guide the modeling of the data-dependent terms L D (D) + L N D (N, D). Specifically, the behavior of ∆ D L(N, D) when plotted against D on log-log axes (Fig. 3 (a)) reveals a consistent and striking linear trend across various model sizes N . This indicates a robust power-law relationship for ∆ D L(N, D) with respect to D.
This empirical power-law behavior of the difference term ∆ D L(N, D) strongly motivates modeling the integrated data-dependent loss component, L D (D) + L N D (N, D), using a power law in D. We therefore propose that this component can be approximated as:
L D (D) + L N D (N, D) = B(N )D -A(N ) + ε R (N, D) = f B (N ; θ B )D -f A (N ;θ A ) + ε R (N, D) (5
)
The functions A(N ) = f A (N ; θ A ) and B(N ) = f B (N ; θ B ) capture the model-size dependencies of the exponent and coefficient, respectively. Where, ε R (N, D) is a stochastic term jointly dependent on N and D that aggregates two sources of variability: the experimental measurement noise inherent to each (N, D) setting and the residual error arising from the fit of the parametric model.
this section cite: []

Section: Stage 1: Initial estimation.
For each unique model size N , we first compute the observed quantity of loss difference, it also can be approximated as:
R N (D) = L(N, D) -L(N, λD) ≈ B(N )(1 -λ -A(N ) )D -A(N ) = RN (D),(6)
where λ = √ 2 in experiments. And RN (D) is predict value of R N (D) which is projected to log-log space for linear regression:
log( RN (D)) = log( BN ) -A N * log(D),(7)
where B N = BN /(1 -λ -A N ). Using Normal Equation [15] for minimizing the error of the loss differences of each N , respectively, as ℓ R,N = D (R N (D) -RN (D)) 2 .
Consequently, for each model size N we obtain a discrete pair of parameters-the exponent A N and the coefficient BN -which constitute the the best linear fit for that particular model size. Collecting these pairs yields the arrays {A N } and {B N }.
this section cite: ['b14']

Section: Stage 2: Parameterization and Iterative Refinement.
Building on the discrete estimates {A N , B N } obtained in Stage 1, we now seek continuous functions f A (N ; θ A ) and f B (N ; θ B ) that simultaneously (i) admit a compact analytical form and (ii) minimize the global error of the loss differences ℓ R = N (ℓ R,N ). Let G = {g (1) , g (2) , . . .} denote a small dictionary of simple, monotone transformations (identity, logarithm, and power functions were found sufficient in practice). For every ordered quadruple 4 we perform the coordinate projection
(g i , g j , g k , g m ) ∈ G
g i A N = a 1 g j (N ) + b 1 + ε A,N , g k B N = a 2 g m (N ) + b 2 + ε B,N ,(8)
where the coefficients (a 1 , b 1 ) and (a 2 , b 2 ) are obtained via the least squares method, yielding best estimates.
For every candidate transform pair we compute the projection-space residual sums of squares ℓ A = N ε 2 A,N and ℓ B = N ε 2 B,N . The transform quadruple that minimizes ℓ A + ℓ B is selected. As results, g i , g k is log function and g j , g m is a power-law function, minimized the residuals, implying that both f A and f B follow a stretched-exponential form: Subsequently, we refine the exponents α and β using an iterative refinement strategy: (i) holding β fixed, we update α to minimize the global residual error ℓ R ; (ii) holding the new α fixed, we update β according to the same criterion. This process is repeated until convergence, which empirically takes only 1-2 iterations. Implementation details, as well as a systematic comparison of alternative functional families and their empirical errors, are provided in Appendix C. Fig 4 illustrates the quality of the resulting fits for f A (N ) and f B (N ) using these stretched exponential forms.
f A (N ; θ A ) = exp a 1 N α + b 1 ,(9)
f B (N ; θ B ) = exp a 2 N β + b 2 . (10
)
this section cite: []

Section: Functional Form for Model-Dependent Residual E + L N (N )
By construction, E + L N (N ) = L(N, D) -(L D (D) + L N D (N, D)), where L(N, D) is directly obtained from experimental data whereas the exact form of L D (D) + L N D (N, D) is not experimentally accessible. Hence E + L N (N ) cannot be directly observed. It can, however, be accessed indirectly through
O(N, D) ≜ L(N, D) -B(N ) D -A(N ) = E + L N (N ) -ε R (N, D),(11)
where ε R (N, D) is the residual defined in Section 3.2. Grouping by model size N and averaging over D yields
G(N ) ≜ Avg D O(N, D) = E + L N (N ) -Avg D ε R (N, D) .(12)
The difference Fitting E + L N (N ). We now model G(N ) ≈ f L N (N ; θ L N ) by repeating the transformation-fitselection procedure of Section 3.2. For each pair g i , g j ∈ G we regress g i G(N ) = a 3 g j (N ) + b 3 via the normal equation and choose the transform pair that minimizes the residual sum of squares The optimal form identified was a logarithmic transformation for g i and a power-law transformation 5HODWLYH(UURU 18SSHU%RXQGIRU)LWWLQJ H D 5REXVWQHVVRI%DVHOLQHUHFLSH E 5REXVWQHVVRI(1=+UHFLSH Figure 6: Robustness and data distribution generalizability of Farseer. (a) Relative error on excluded 6.4 B models as a function of the largest model size used in fitting, assessing robustness to fitting data volume. (b) Relative error on excluded 3.2 B models trained with an English-Chinese data recipe, demonstrating structural generalizability to different data mixes. Circle size and adjacent numbers indicate the number of model-size points used for fitting in each case.
G(N ) -O(N, D) = ε R (N, D) -Avg D ε R (N, D)(13)
g j (N ) = N γ for g j . The optimal exponent γ is identified via a grid search minimizing the fitting error, leading to a stretched-exponential form
E + L N (N ) ≃ G(N ) ≈ exp a 3 N γ + b 3 .(14)
Fig. 5 (a) shows that this fit attains a relative error of only 0.09%. Given the negligible estimation error of G(N ) in approximating E + L N (N ), we have thus completed a robust modeling and fitting of the model-dependent residual. Combining these stages yields the fully specified scaling law in Eq. 3, with all functional forms and parameters fixed. Further details, ablation studies, and noise analyses are provided in Appendix C.
this section cite: []

Section: Farseer's Properties: Robustness, Generalization, and Extrapolation
This section evaluates key properties of Farseer: its robustness to fitting data amount, its generalization across different training data recipes, and its extrapolation ability beyond the fitting range.
this section cite: []

Section: Robustness to Fitting Data
We measure how prediction accuracy changes with data size to assess Farseer's robustness, parameter stability, and prediction consistency as more model-size points become available. Specifically, we fit Farseer using subsets of models with progressively increasing upper bounds on model size N . For each fitting process based on a subset, we consistently measure the relative error on the 6.4 B models, which is deliberately excluded from all fitting subsets. As illustrated in Fig 6 (a), the predicted relative error on the excluded 6.4 B models decreases significantly as more data points are included in the fit. When fitting only models up to 1.9 B parameters, the relative error is 6.01 × 10 -3 ; expanding the fit to include models up to 5.4 B reduces it to only 5.87 × 10 -4 . Crucially, the relative error decreases nearly monotonically as additional model-size points are incorporated. This trend highlights Farseer's accuracy and robustness to fitting-data volume, yielding reliable predictions with limited data and predictable gains as more data are added.
this section cite: []

Section: Data Distribution Generalization
To investigate the generalizability of Farseer, we evaluate it on a different training data setup. All prior experiments used the Baseline data recipe, specified in Appendix B. Fig 6 (b) extends this analysis to evaluate Farseer using data from models trained with the English-Chinese (EN-ZH) recipe (also detailed in Appendix B), which represents a dramatically different bilingual data mix. Similar to the robustness analysis, as the fitting data increases, the relative error on deliberately excluded 3.2 B models trained with the EN-ZH recipe steadily converges and stays consistently low, reaching 7.60×10 -4 at the final point. This shows that Farseer captures key scaling trends and delivers reliable predictions even under a dramatically altered bilingual data mix. This result underscores Farseer's structural generalizability across diverse training-data compositions and suggests its applicability to models trained on varied datasets. As an illustration, a detailed surface comparison of different English-Chinese data mixture ratios is provided in Appendix E.
this section cite: []

Section: Extrapolation Capabilities
25.1 B To quantify the extrapolative capacity of Farseer, we fit its parameters on a √ 2-spaced sampling grid of small model sizes and dataset sizes, then predict BPC values for substantially larger and off-grid combinations. Fig. 7 presents six such extrapolation targets (red stars) that situate well outside the calibration region (blue circles). Most notably, the 25.1 B model extends the evaluation domain beyond the largest calibrated scale by more than an order of magnitude. Despite this, Farseer's prediction for this model exhibits a relative error of merely 0.47 %. The remaining validation points selected to assess extrapolation at both increased dataset sizes and off-grid (N, D) configurations demonstrate similarly low relative errors, ranging from 0.26 % to 0.72 %. Across these extrapolation targets, Farseer's average relative error is just 0.50 %, whereas the Chinchilla scaling law exhibits an average relative error of 2.68 %, a 433 % increase. These consistently low errors demonstrate that Farseer accurately captures the smooth functional dependence of BPC on both model and dataset scale. In particular, the highly accurate extrapolation at the 25.1 B model provides compelling evidence that Farseer generalizes robustly to previously unfitted scales, thereby serving as a reliable instrument to forecast the performance under yet larger computational budgets.
this section cite: []

Section: Formula and Monotonicity
As formulated in Eq. ( 3) and derived via our fitting method, Farseer possesses intrinsic mathematical properties that align with the theoretical expectations for loss functions in machine learning. This section details the fitted formula and its fundamental property of monotonicity.
Formula. The specific fitted law for Farseer is given by: L(N, D) = e -0.021•N 0.169 -0.091 + e 88.01•N -0.1 -6.287 • D -e -0.124•N 0.123 +0.424 (15) Monotonicity. A fundamental expectation is that model performance should improve (i.e., loss should decrease) with more resources. Farseer inherently satisfies this: L(N, D) is monotonically decreasing with respect to both increasing N and D. This behavior is a direct consequence of its functional form and the parameter constraints imposed by our fitting procedure. A detailed analysis of the partial derivatives confirming this monotonicity is provided in Appendix I.
this section cite: ['b14']

Section: Comparative Analysis of Farseer and Chinchilla
We compare our proposed scaling law Farseer against the widely recognized Chinchilla [20] in Eq. ( 2). This comparison highlights the advantages of our approach in terms of predictive accuracy, extrapolation, and guidance on optimal resource allocation.
this section cite: ['b20']

Section: Formula Comparison: Predictive Power
We compared the predictive power of Chinchilla and Farseer using our comprehensive dataset (Appendix B). Standard non-linear regression was used for fitting both, and Chinchilla was also evaluated with our multi-round iterative method (Appendix H). As shown in Fig. 1 (a) and Fig. 2, Farseer's predicted BPC aligns remarkably closely with empirical values across various N and D. Conversely, Chinchilla fit systematically deviates, underestimating at low Dand overestimating at high D. This superior fit underscores Farseer's higher expressive capacity, enabled by its A(N ) and B(N ) that capture decay rates specific to each model size, unlike Chinchilla's single average trend.
this section cite: []

Section: Property Comparison: Robustness and Extrapolation
To assess the robustness and extrapolation, we examine the generalization beyond the training domain in two scenarios (see Fig. 8), plotting relative error as the upper bound on model size N used for fitting increases. As shown in Fig. 8 (a), with model size fixed at 6.4 B parameters, we vary N from 1.9 × 10 9 to 5.4 × 10 9 and observe that Farseer formula yields both lower average relative error and smaller error variance than the Chinchilla fit, indicating enhanced robustness to changes in the fitting range. Fig. 8 (b) considers a fixed model size of 25.1 B, which is far beyond the fitting range. As N increases, Farseer's relative error shows a clear downward trend and reaches a low level, while Chinchilla's error remains persistently high. These results collectively demonstrate that Farseer not only offers greater robustness within the fitting range but also generalizes more reliably when extrapolating to larger model scales.
this section cite: []

Section: 5HODWLYH(UURU
)DUVHHU &KLQFKLOOD H 18SSHU%RXQGIRU)LWWLQJ D 5REXVWQHVVDWIL[HGN % E ([WUDSRODWLRQDWIL[HGN %
this section cite: []

Section: Application Comparison: Optimal Computing Resource Allocation
Efficient training requires balancing model size N and dataset size D, typically reflected in the optimal D/N ratio. we compare the practical guidance offered by Farseer and Chinchilla on optimal compute allocation during training, where the total budget is taken as C = 6N D [22,20]. Fig. 1 (b) presents the optimal D/N ratio predicted by both Farseer and Chinchilla as a instrument of the total compute budget C. Chinchilla, suggesting an optimal D/N ratio around 20 (typically 10-30), is primarily derived from and applicable to training at moderate compute budgets corresponding to smaller model sizes. This constant ratio provides inaccurate guidance for larger scale. In contrast, Farseer predicts a steadily increasing optimal D/N ratio as the compute budget (and consequently, the optimal model size) grows. This predicted trend aligns remarkably well with the actual training configurations adopted for recent state-of-the-art large language models (e.g., Qwen [46,45], Llama [39,40,16]), indicating that Farseer offers more accurate and relevant guidance for optimizing resource allocation in current and future large-scale model training.
this section cite: ['b22', 'b20', 'b46', 'b45', 'b39', 'b40', 'b16']

Section: Related Work
Training LLMs necessitates optimizing compute allocation via scaling laws. The discovery and application of these laws have provided the confidence to invest massive compute resources, knowing they can be predictably converted into model capability [22]. OpenAI's foundational work established a power-law relationship between cross-entropy loss and model/data scale (Eq. 1), advocating for training large models on moderate data with early stopping. However, this assumes oversimplified dynamics and fixed hyperparameters, limiting extrapolation reliability.
DeepMind's Chinchilla [20] refined this by proposing proportional scaling of model and data size based on an updated methodology (Eq. 2), empirically demonstrating improved efficiency. Under the same compute budget as Gopher [32], Chinchilla adopts a smaller model paired with more training data, yielding substantial gains on downstream tasks. Nevertheless, recent work [4] has raised concerns about the reproducibility of Chinchilla's findings and demonstrated that optimal token-to-parameter ratios are sensitive to data quality, suggesting current scaling laws are valuable heuristics but require calibration and lack universal applicability.
The landscape of scaling law research is rapidly evolving beyond classical formulations, exploring multifaceted avenues for enhanced efficiency and generalization. Recent empirical work includes deriving observational scaling laws from existing models to unify performance patterns and explain emergent phenomena [33,3,18,42], while other studies apply scaling principles to improve data curation, e.g., by strategically filtering datasets [27,26,35,30,48,13]. Theoretical inquiries are concurrently advancing our comprehension of core mechanisms: some demonstrate how sophisticated feature learning can potentially double scaling rates for intricate functions [5], while others establish direct connections between generalization error exponents and data manifold dimensionality [18]. Furthermore, the purview of scaling laws now extends significantly beyond pre-training, with investigations into their implications for inference dynamics [43,44,24,43,34], parameter and communication efficiency [1,8], and downstream task [21,9,44,19,36,10,12,37,38,28,23].
this section cite: ['b22', 'b20', 'b32', 'b3', 'b33', 'b2', 'b18', 'b42', 'b27', 'b26', 'b35', 'b30', 'b48', 'b12', 'b4', 'b18', 'b43', 'b44', 'b24', 'b43', 'b34', 'b0', 'b7', 'b21', 'b8', 'b44', 'b19', 'b36', 'b9', 'b11', 'b37', 'b38', 'b28', 'b23']

Section: Limitations
This study has several limitations. First, our empirical validation is primarily based on Llama-style decoder-only Transformers. While the core methodology may be generalizable, its applicability to other architectures, such as Mixture-of-Experts (MoE), requires further investigation. Second, our largest validated model has 25.1B parameters. Extrapolating to trillion-parameter models remains an open question, constrained by computational resources and the engineering complexities of largescale parallel training. Third, in line with other work in this area, our scaling law is empirically derived and lacks a first-principles theoretical justification. We have open-sourced our data to encourage community efforts on this front. Finally, this work focuses on pre-training loss (BPC); extending scaling laws to predict downstream task performance is a complex but important direction for future work. A more detailed discussion of limitations is available in Appendix J.
this section cite: []

Section: Conclusion
We propose Farseer, a refined scaling law and methodology for Large Language Models. By accurately modeling the loss surface L(N, D) through a novel fitting approach, Farseer provides significantly better empirical fit and superior extrapolation capabilities compared to prior scaling laws like Chinchilla's. This work enables reliable prediction of large-scale model performance from tractable small-scale experiments, bridging the critical scaling gap and facilitating more efficient evaluation of training strategies and compute allocation. Validated on a large corpus of trained models, Farseer offers valuable insights for LLM development.
this section cite: []

Section: References
Ref_id:b0 Title: Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs flops: Scaling laws for optimal sparsity for mixture-of-experts language models Year: (2025)
Ref_id:b1 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: Explaining neural scaling laws Year: (2024)
Ref_id:b3 Title: Chinchilla scaling: A replication attempt Year: (2024)
Ref_id:b4 Title: How feature learning can improve neural scaling laws Year: (2024)
Ref_id:b5 Title: Language models are few-shot learners Year: (2020)
Ref_id:b6 Title: A survey on mixture of experts Year: (2024)
Ref_id:b7 Title: Communication-efficient language model training scales reliably and robustly: Scaling laws for diloco Year: (2025)
Ref_id:b8 Title: Scaling laws for predicting downstream performance in llms Year: (2024)
Ref_id:b9 Title: Reproducible scaling laws for contrastive language-image learning Year: (2023)
Ref_id:b10 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b11 Title: Scaling laws do not scale Year: (2024)
Ref_id:b12 Title: Scaling laws for dense retrieval Year: (2024)
Ref_id:b13 Title: A new algorithm for data compression Year: (1994)
Ref_id:b14 Title: Theoria motus corporum coelestium in sectionibus conicis solem ambientium Year: ()
Ref_id:b15 Title:  Year: ()
Ref_id:b16 Title: The llama 3 herd of models Year: (2024)
Ref_id:b17 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b18 Title: Understanding scaling laws with statistical and approximation theory for transformer neural networks on intrinsically low-dimensional data Year: (2024)
Ref_id:b19 Title: Scaling laws for transfer Year: (2021)
Ref_id:b20 Title: Training Compute-Optimal Large Language Models Year: (2022-03)
Ref_id:b21 Title: Scaling laws for downstream task performance of large language models Year: (2024)
Ref_id:b22 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b23 Title: Scaling laws for precision Year: (2024)
Ref_id:b24 Title: A simple model of inference scaling laws Year: (2024)
Ref_id:b25 Title: Predictable scale: Part i-optimal hyperparameter scaling law in large language model pretraining Year: (2025)
Ref_id:b26 Title: (mis) fitting scaling laws: A survey of scaling law fitting techniques in deep learning Year: (2024)
Ref_id:b27 Title: Scalingfilter: Assessing data quality through inverse utilization of scaling laws Year: (2024)
Ref_id:b28 Title: Scaling laws for black box adversarial attacks Year: (2024)
Ref_id:b29 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b30 Title: The quantization model of neural scaling Year: (2023)
Ref_id:b31 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b32 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b33 Title: Observational scaling laws and the predictability of language model performance Year: (2024)
Ref_id:b34 Title: Beyond chinchilla-optimal: accounting for inference in language model scaling laws Year: (2024)
Ref_id:b35 Title: Scaling laws from the data manifold dimension Year: (2022)
Ref_id:b36 Title: Scaling law for recommendation models: Towards general-purpose user representations Year: (2023)
Ref_id:b37 Title: Beyond neural scaling laws: beating power law scaling via data pruning Year: (2022)
Ref_id:b38 Title: Unraveling the mystery of scaling laws: Part i Year: (2024)
Ref_id:b39 Title: Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models Year: (2023-02)
Ref_id:b40 Title: Llama 2: Open foundation and fine-tuned chat models Year: ()
Ref_id:b41 Title: Attention is all you need Year: (2017)
Ref_id:b42 Title: Data efficient neural scaling law via model reusing Year: (2023)
Ref_id:b43 Title: Inference scaling laws: An empirical analysis of compute-optimal inference for llm problem-solving Year: (2025)
Ref_id:b44 Title: Unveiling downstream performance scaling of llms: A clustering-based perspective Year: (2025)
Ref_id:b45 Title: Qwen3 technical report Year: (2025)
Ref_id:b46 Title: Guanting Dong, Qwen Team, and Alibaba Group et al. Qwen2 technical report Year: (2024)
Ref_id:b47 Title: Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer Year: (2022)
Ref_id:b48 Title: Scaling laws for dataefficient visual transfer learning Year: (2025)
