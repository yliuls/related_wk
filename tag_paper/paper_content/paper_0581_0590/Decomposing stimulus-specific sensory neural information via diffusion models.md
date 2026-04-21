Title: Decomposing stimulus-specific sensory neural information via diffusion models
Abstract: To understand sensory coding, we must ask not only how much information neurons encode, but also what that information is about. This requires decomposing mutual information into contributions from individual stimuli and stimulus features-a fundamentally ill-posed problem with infinitely many possible solutions. We address this by introducing three core axioms-additivity, positivity, and locality-that any meaningful stimulus-wise decomposition should satisfy. We then derive a decomposition that meets all three criteria and remains tractable for high-dimensional stimuli. Our decomposition can be efficiently estimated using diffusion models, allowing for scaling up to complex, structured and naturalistic stimuli. Applied to a model of visual neurons, our method quantifies how specific stimuli and features contribute to encoded information. Our approach provides a scalable, interpretable tool for probing representations in both biological and artificial neural systems.

Section: Introduction
A central question in sensory neuroscience is how much, but also what information neurons transmit about the world. While Shannon's information theory provides a principled framework to quantify the amount of information neurons encode about all stimuli, it does not reveal which stimuli contribute most, or what stimulus features are encoded [25,2,8]. As a concrete example, it is known that neurons in the early visual cortex are 'sensitive' to stimuli in a small region of space (their receptive field). However, it is not clear how such simple intuitions carry to more complex scenarios, e.g. with large, noisy & non-linear population of neurons and high-dimensional stimuli.
Several previous measures of neural sensitivity have been proposed. For example, the Fisher information quantifies the sensitivity of neural responses to infinitesimal stimulus perturbations [23,3,31,18,9]. However, as the Fisher is not a valid decomposition of the mutual information it cannot say how different stimuli contribute to the total encoded information. On the other hand, previous works have proposed stimulus dependent decompositions of mutual information, which define a function I(x) such that I(R; X) = E[I(x)] [8,4,5,17]. However, this decomposition is inherently ill-posed: infinitely many functions I(x) satisfy the constraint, with no principled way to select among them. Further, different decompositions behave in qualitatively different ways, making it hard to interpret what are they are telling us. Finally, most proposed decompositions are computationally intractable for the high-dimensional stimuli and non-linear encoding models relevant for neuroscience.
To resolve these limitations, we propose a set of axioms that any stimulus specific and feature-specific information decomposition should satisfy in order to serve as a meaningful and interpretable measure of neural sensitivity. These axioms formalize intuitive desiderata: that the information assigned to each stimulus, and stimulus feature, should be non-negative, and additive with respect to repeated measurements. We also require the decomposition to respect a form of locality: changes in how a neuron responds to a stimulus x should not affect the information attributed to a distant stimulus x ′ . Finally, the attribution must be insensitive to irrelevant features, which do not contribute to the total information. Together, these constraints ensure that the decomposition is both interpretable and theoretically grounded.
We show that existing decompositions violate one or more of these axioms, limiting their interpretability and use as information theoretic measures of neural sensitivity. We then introduce a novel decomposition that satisfies all of our axioms. It generalizes Fisher information by capturing neural sensitivity to both infinitesimal and finite stimulus perturbations. Moreover, it supports further decomposition across individual stimulus features (e.g., image pixels), enabling fine-grained analysis of neural representations.
Beyond satisfying our theoretical axioms, our decomposition is computationally tractable for large neural populations and high-dimensional naturalistic stimuli, through the use of diffusion models. We demonstrate the power of our method by quantifying the information encoded by a model of visual neurons about individual images and pixels. Our approach uncovers aspects of the neural code that are not picked up by standard methods, such as the Fisher information, and opens the door to similar analyses in higher-order sensory areas, and artificial neural networks.
this section cite: ['b24', 'b1', 'b7', 'b22', 'b2', 'b30', 'b17', 'b8', 'b7', 'b3', 'b4', 'b16']

Section: Desired properties of stimulus-specific decomposition of information
We aim to decompose the mutual information I(R; X) between a stimulus X and a neural response R into local attributions I(x), assigning to each stimulus x a measure of its contribution to the total information. By construction, such a decomposition must satisfy:
• Axiom 1: Completeness. The average of local attributions must recover the total mutual information:
I(R; X) = E X [I(x)].(1)
This constraint alone does not uniquely determine the function I(x), as many decompositions satisfy completeness. To further constrain the attribution, we impose additional desiderata that reflect desirable properties of local information measures.
First, for I(x) to serve as an interpretable, stimulus-specific measure of neural sensitivity, it should satisfy a locality principle: perturbations to the likelihood &/or prior in a neighbourhood of x 0 should have a vanishing influence on I(x) for distant stimuli x ̸ = x 0 . Without this property, changes in I(x) may reflect changes in neural sensitivity to any stimuli, undermining interpretability.
• Axiom 2: Locality. Let p(R, X) and p(R, X) be two joint distributions over the response, R, and stimulus, X. Suppose there exists finite ϵ > 0 and x 0 such that: p(R, x) = p(R, x) for all x / ∈ B ϵ (x 0 ),
where B ϵ (x 0 ) is the open ball of radius ϵ centered at x 0 . Assuming X has infinite support on an unbounded or semi-infinite domain, we require that, for every δ > 0 there should exist some finite value d, such that:
I(x) -Ĩ(x) ≤ δ for all x ̸ ∈ B d (x 0 ),(3)
where I(x) and Ĩ(x) denote the corresponding information decompositions derived from p(R, x) and p(R, X), respectively. That is, local perturbations to the likelihood or prior near x 0 should not affect the information assigned to distant stimuli, x.
this section cite: []

Section: Mutual information is globally non-negative: I(R; X) ≥ 0.
To preserve this property in our decomposition, we require the pointwise contributions I(x) to be non-negative as well. Intuitively, observing a neural response can only refine our beliefs about the stimulus-it cannot undo information.
In addition, negative attributions can harm interpretability, since they can cancel out, obscuring how different stimuli contribute to the total information.
• Axiom 3: Positivity. Local information attributions must be non-negative:
I(x) ≥ 0 for all x ∈ X.(4)
Finally, Shannon 1948 posited additivity as a fundamental property of mutual information: the total information from multiple sources should equal the sum of their individual contributions. We extend this principle pointwise, requiring that information combine additively across measurements.
• Axiom 4: Additivity. For two responses R and R ′ , the local attribution should decompose as:
I R,R ′ (x) = I R (x) + I R ′ |R (x), for all x ∈ X(5)
where we have renamed I(x) as I R (x) here, to make explicit its dependence on the response, R. I R ′ |R (x) is the conditional pointwise information from R ′ given R, and I R,R ′ (x) denotes the pointwise information from observing both responses. By construction, we require:
I(R ′ ; X|R) = E x I R ′ |R (x) and I(R, R ′ ; X) = E x [I R,R ′ (x)].
Remark 1: Local data processing inequality. Any decomposition that fulfils both additivity and positivity, as stated above, also obeys a local form of the data processing inequality. That is, postprocessing should not increase information, even at the level of the individual attributions. Formally,
If X → R → R ′ , then I R (x) ≥ I R ′ (x) for all x ∈ X (6
)
To prove this we use additivity to write I R ′ ,R (x) in two ways:
I R (x) + I R ′ |R (x) = I R ′ (x) + I R|R ′ (x)(7)
Positivity gives I R|R ′ (x) ≥ 0 for all x, so:
I R (x) + I R ′ |R (x) ≥ I R ′ (x) (8
) Now, if R ′ is independent of X given R, then I(R ′ ; X|R) = E X [I R ′ |R (x)] = 0. Since if I R ′ |R (x) ≥ 0 pointwise, this implies I R ′ |R (x) = 0 for all x.
Substituting into the inequality above yields the desired result: I R (x) ≥ I R ′ (x).
this section cite: []

Section: Remark 2: Invariance to invertible transformations.
The data processing inequality implies that local information attributions are invariant under invertible transformations of the response variable. That is, for any invertible function ϕ(R), we have:
I ϕ(R) (x) = I R (x). (9
)
This follows from the fact that I(R; X) = I(ϕ(R); X), and hence
E X [I ϕ(R) (x)] = E X [I R (x)].
Supposing, for contradiction, that I ϕ(R) (x) < I R (x) for some x, equality of expectations would require I ϕ(R) (x) > I R (x) for some other x, violating the pointwise data processing inequality. This highlights why our axioms are important for interpretability: they imply that I (x) quantifies how information is transmitted through the system X → R, independently of how the responses are parameterized. Table 2: Satisfaction of axioms by different information-theoretic measures of neural sensitivity. Only our proposed decomposition, I local , fulfils all the axioms. Axiom J F isher (x) I sp (x) I SSI (x) I surp (x) I CiSSI (x)
Completeness ✗ ✓ ✓ ✓ ✓ Locality ✓ ✗ ✗ ✗ ✗ Positivity ✓ ✗ ✗ ✓ ✓ Additivity ✓ ✓ ✓ ✓ ✓
this section cite: []

Section: Previous stimulus-dependent decompositions
Several previous works have proposed decompositions of mutual information into stimulus-specific contributions, such as the stimulus-specific information I SSI [4], the specific information I sp , the stimulus-specific surprise I surp , [8], and the coordinate-invariant stimulus-specific information I CiSSI [17]:
I sp (x) = H(R) -H(R|x) (10
) I SSI (x) = H(X) -E R|x [H(X|R = r)] (11
) I surp (x) = D KL (p(R|x)∥p(R)) (12
) I CiSSI (x) = E R|x [D KL (p(X|R = r)∥p(X))] .(13)
While these decompositions satisfy some of our proposed axioms (Table 2), none satisfy locality. This is because they all depend on global terms such as the marginal response distribution p(r) = p(r|x ′ )p(x ′ ) dx ′ , or the posterior p(x|r) = p(r|x)p(x)/ p(r|x ′ )p(x ′ ) dx ′ , both of which can be influenced by changes to the likelihood &/or prior for any stimulus. Further, the requirement that I CiSSI (x) is invariant to invertible transformations of x is incompatible with locality; both axioms can't be fulfilled simultaneously. As discussed above, this limits the interpretability of previous decompositions as measures of neural sensitivity, since a non-zero attribution at x could be due to changes in neural sensitivity anywhere in the stimulus space.
Unlike the above decompositions, the Fisher information is inherently local. However, while previous authors found a relation between the Fisher and mutual information [3,29], this only holds approximately, and in certain limits (e.g. low-noise). Therefore, the Fisher information cannot be used to quantify how different stimuli contribute to the total encoded information (i.e. it fails the completeness axiom).
Recently Kong et al. 2024 used diffusion models to decompose the mutual information into contributions from both the stimulus, x, and the response, r. Two different decompositions, I(r, x), were proposed. However, averaging these over p(R|x) does not give stimulus-wise decompositions that fulfil our axioms (Appendix B). In one case, we obtain I surp (x) (Eqn 12), which is non-local; in the other case, the decomposition is not additive, which is a fundamental information theoretic constraint.
In the following we propose a new stimulus-wise decomposition of the mutual information which fulfils all our axioms, combining the advantages of the Fisher information (locality, positivity & additivity) while being a valid decomposition of the mutual information.
this section cite: ['b3', 'b7', 'b16', 'b2', 'b28']

Section: Diffusion-based information decomposition
We first derive an expression for the mutual information, I (R; X), that can be used to construct a stimulus-dependent decomposition that fulfils all of the above axioms.
this section cite: []

Section: Exact relation between Shannon information and Fisher information
We consider a population of neurons which show a response, R, to a stimulus, X, with probability p(R|X). We then consider a noise-corrupted version of the stimulus, X γ = X + √ γZ, where Z ∼ N (0, I). From the fundamental theorem of calculus we can write:
- ∞ γ0 dI(X γ ; R) dγ dγ = I(R; X γ0 ) -lim γ→∞ I(R; X γ ) = I(R; X γ0 ),(14)
since in the limit γ → ∞, X γ is just noise, and thus I(R; X γ ) → 0. It follows that,
I(R; X γ0 ) = - ∞ γ0 dI(X γ ; R) dγ dγ = ∞ γ0 E p(R) dh(X γ |r) dγ - dh(X γ ) dγ dγ (15
) where h (X γ ) ≡ -E p(Xγ ) [log p(X γ )] and h (X γ |r) ≡ -E p(Xγ |r) [log p (X γ |r)].
Assuming that p(X) and p(X|R) have finite second order moments, we can apply de Bruijn's identity for all γ > 0, to obtain,
I(R; X γ0 ) = - 1 2 Trace ∞ γ0 E p(Xγ ,R) ∇ 2 xγ log p (X γ |R) -∇ 2 xγ log p (X γ ) dγ = - 1 2 Trace ∞ γ0 E p(Xγ ,R) ∇ 2 xγ log p (R|X γ ) dγ = 1 2 Trace ∞ γ0 E p(Xγ ) [J (X γ )] dγ (16
)
where J (x γ ) is the Fisher information with respect to a noise-corrupted stimulus, X γ . Finally, since lim γ0→0 I(R; X γ0 ) = I(R; X), and Trace(E p(Xγ ) [J(X γ )]) is always non-negative, monotone convergence yields:
I(R; X) = 1 2 Trace ∞ 0 E p(Xγ ) [J (X γ )] dγ.(17)
This is a general result that holds for both discrete and continuous X and R, so long as p(X) and p(X|R) have finite first and second moments.
The above identity provides a direct relation between the mutual information I(R; X) and the Fisher information, J (x γ ). Further, it also admits a natural interpretation in terms of de-noising diffusion models trained to predict X from a noisy observation X γ . To see this, first we use an alternative formulation of the Fisher information, in terms of the mean-squared score:
I(R; X) = 1 2 ∞ 0 E p(Xγ ,R) ∇ xγ log p (R|X γ ) 2 dγ = 1 2 ∞ 0 E p(Xγ ,R) ∇ xγ log p (X γ |R) -∇ xγ log p (X γ ) 2 dγ (18
)
Next, from Tweedie's formula [20,14] we have:
I(R; X) = ∞ 0 1 2γ 2 E p(Xγ ,R) ∥x(X γ , R) -x(X γ )∥ 2 dγ,(19)
where
x(x γ ) = E[X|x γ ] and x(x γ , r) = E[X|x γ , r].
These conditional means can be approximated using denoising diffusion models trained to sample from the prior, p(X), and posterior p(X | R), respectively [26]. I SSI (x) F x x E I surp (x) I CiSSI (x) -5 0 5 0 2 C x I local (x) x G -5 0 5 0 2 D x I sp (x) Figure 1: Demonstration of locality. (A) A neuron responds to a 1D stimulus x with tuning curve f (x). We compare two cases: (i) tuning changes only near x = -3 (black), and (ii) additional changes near x = 6 (red dashed). (B-C) The Fisher information, J (x) and local information I local (x) converge as we go far from the region where the two tuning curves differ. (C-F) For other decompositions (Eqs. 10-13), local changes in the tuning curve result in non-local changes to the information attribution, for all x.
this section cite: ['b19', 'b13', 'b25']

Section: Local stimulus-specific information
Building on the integral representation of mutual information in Eqn 17, we define a stimulus-specific decomposition:
I local (x) = 1 2 i ∞ 0 E p(Xγ |x) [J ii (X γ )] dγ,(20)
where X γ = x + √ γZ, with Z ∼ N (0, I), and
J ii (x γ ) = E p(R|xγ ) -∇ 2 (xγ )i log p(R | x γ )
is the i th diagonal element of the the Fisher information matrix, evaluated at x γ . This expression parallels Eq. 17, with the key distinction that the expectation is now conditioned on a fixed stimulus x, rather than averaging over the full stimulus distribution 3 . Consequently, the decomposition satisfies the completeness axiom, since by construction, I(R; X) = E [I local (x)] .
Next we outline why our decomposition fulfils the locality axiom (for the formal proof, see Appendix A). Recall that the Fisher information matrix J (x) characterizes the local curvature of the loglikelihood, log p(R | x), and thus quantifies the local sensitivity of the response to changes in the stimulus [23]. The term E Xγ |x [J (X γ )] generalizes this notion, measuring neural sensitivity when we only observe noise-perturbed versions of the stimulus, X γ ∼ N (x, γI). For finite γ, this term is dominated by values of X γ close to x, and thus, it depends only on the local shape of the likelihood and prior around x. It receives a vanishingly small contribution from changes to the likelihood and/or prior for distant stimuli x ′ , if ∥x ′ -x∥ ≫ √ γ. Meanwhile, as γ → ∞, X γ becomes pure noise and
thus E p(Xγ |x) [J (X γ )] → 0 for all x.
Taken together, this implies that our decomposition, obtained by integrating E p(Xγ |x) [J (X γ )] over all γ > 0, satisfies the locality axiom: local perturbations to p(R, x) affect I local (x ′ ) only for nearby x ′ , while their influence vanishes as ∥x ′ -x∥ → ∞.
The remaining axioms follow directly from standard properties of the Fisher information matrix.
Positivity follows from the fact that the Fisher information is positive semi-definite. Additivity follows from the identity [32]. Both properties are preserved when we average the Fisher information over p(X γ |x) and integrate over γ > 0, to obtain I local (x). C E x x B I local (x) D F I CiSSI (x) x x I SSI (x) gaussian bimodal gaussian bimodal gaussian bimodal gaussian bimodal
J R ′ ,R (x γ ) = J R (x γ ) + J R ′ |R (x γ )
-2 -1 0 1 2 0 1 2 p(X | r) x 2 x 1 δ 0 0.2 0.4 x p(X) p(X | r) r ∼ p(R | x 1 ) x 2 x 1 r ∼ p(R | x 2 ) δ A p(X | r) x 2 x 1 δ x p(X) p(X | r) r ∼ p(R | x 1 ) x 2 x 1 r ∼ p(R | x 2 ) δ
this section cite: ['b22', 'b31']

Section: Feature-Wise Decomposition
We assume the stimulus x is a vector of image features x i (e.g. image pixels). Given that the local stimulus information (Eqn. 20) is expressed as a sum over diagonal elements of the Fisher information matrix, it is natural to decompose I local (x) into feature-wise contributions I i (x).
As with the stimulus-wise decomposition, this problem is ill-posed: infinitely many decompositions exist in theory. However, the same axioms constrain the feature-wise decomposition. To satisfy additivity, I i (x) must be a linear combination of Fisher diagonal terms:
I i (x) = 1 2 j a ij ∞ 0 E Xγ |x [J jj (X γ )] dγ.
Completeness requires i a ij = 1, while positivity enforces a ij ≥ 0. Finally, to fully specify the weights, a ij , we need to introduce one further axiom, which ensures that the attribution I i (x) is zero for irrelevant stimulus features, X i , which are independent of the response, R.
• Axiom 5: Insensitivity to irrelevant features.
If X i is independent of R, then I i (x) = 0 ∀x ∈ X.
For this axiom to hold, we need to set a ij = δ ij , so that
I i (x) = 1 2 ∞ 0 E Xγ |x [J ii (X γ )] dγ.
If, on the contrary, a ij ̸ = δ ij , then a neuron's sensitivity to other features (i.e. J jj (x) > 0) could 'leak over' to make I i (x) > 0 even when X i is independent of R, violating the axiom.
this section cite: []

Section: Results

this section cite: []

Section: Locality
To illustrate the implication of the locality axiom, we analyzed the responses of a model neuron to a one-dimensional stimulus drawn from a Gaussian prior. The neuron's response was modeled as a Gaussian random variable with mean f (x) and fixed standard deviation. We compared two tuning curves: one with a single peak at x = -3 (Fig. 1A, black), and another with an additional peak at x = 6 (Fig. 1A, red).
With gaussian noise, Fisher information scales with f ′ (x) 2 , and thus peaked where the tuning curves were steepest (Fig. 1B). Similar qualitative behaviour was observed for I local (x) (Fig 1C). Crucially, -2 0 2 0 0.5 1 SSI(x) x R R′ = | R | x p(x) x mean response I local (x) x I sp (x) x D E A B C F I surp (x) the difference in I local (x) between the two tuning curves vanished outside the region where they differ, consistent with the locality axiom. In contrast, existing attribution methods (Fig. 1D-G) showed global sensitivity: adding a second peak at x = 6 altered the attributed information across the entire stimulus space, including far from the added feature.
x R R′ = | R | R R′ = | R | R R′ = | R | R R′ = | R |
this section cite: []

Section: Effect of Prior
We next examined how I local (x) responds to changes in the shape of the stimulus prior. For this, we considered two different stimulus priors: a zero-mean gaussian, and a bimodal mixture of two gaussians with peaks at x = -1 and x = 1 (Fig. 2A-B, upper panels). To isolate the effect of the prior, we used a simple linear-Gaussian likelihood model: r ∼ N (x, σ 2 ). Under this model, the neuron's sensitivity is uniform across all stimuli, so any variation in attributed information must arise solely from the prior.
Intuitively, one can assess neural sensitivity by measuring how much the posterior distribution p(X | r) changes, on average, in response to small perturbations in the stimulus x. With the bimodal prior (Fig. 2B), the posterior is highly sensitive near x = 0, where the two modes compete (Fig. 2B, middle panel), and relatively stable near the modes themselves, e.g., around x = 1 (Fig. 2B, lower panel). Our attribution measure I local (x) reflects this structure, peaking at x = 0, and decaying elsewhere (Fig. 2C). For the Gaussian prior, where posterior sensitivity is constant, I local (x) is flat. In contrast, previously proposed attribution methods behave inconsistently: some remain constant across both priors (Fig. 2D), others respond in the opposite direction (Fig. 2E), and some varied strongly even under a Gaussian prior, where the posterior sensitivity is uniform (Fig. 2F).
this section cite: []

Section: Data Processing Inequality
Next we illustrate how I local (x) respects the data processing inequality while certain other attribution methods do not. For this, we used a bimodal prior (Fig. 3A) and compared a linear-Gaussian neuron (r ∼ N (x, σ 2 r )) to a downstream neuron with response r ′ = |r|. Since this transformation is noninvertible, information must be lost. Consistent with the inequality, I local (x) decreased at every x (Fig 3C). In contrast, both I SSI and I sp increased at some x and decreased at others, violating the pointwise data processing inequality (Fi 3D-E). I surp , by comparison, respected the inequality (Fig 3F).
this section cite: []

Section: Scaling to high-dimensions
We can use Eqn 20 to write the feature-wise decomposition in terms of the outputs of an unconditional and conditional diffusion model, trained to output x(x γ ) = E [X|x γ ] and x(x γ , r) = E [X|x γ , r], respectively:
I i (x) = ∞ 0 1 2γ 2 E Xγ |x,R|Xγ (x i (X γ , R) -xi (X γ )) 2 dγ (21
)
To obtain a Monte-carlo approximation of this expression, we need to sample from x γ ∼ p (X γ |x), followed by, r ∼ p (R|x γ ). Sampling from p (R|x γ ) is prohibitively expensive (since it requires first sampling from x ∼ p(x 0 |x γ ), which requires a full backward pass of the diffusion model).
To get around this, we adopt an approximation used by Chung et al. 2023, instead approximating I i (x) using samples r ∼ p(R|x (x)), which can be computed efficiently using one pass through the de-noising network. The integral over γ was approximated numerically with evenly spaced γ (see Appendix C). Since the Fisher decays to zero for large γ, truncating the integral has little effect on our approximation of the integral.
this section cite: []

Section: Pixel-wise decomposition of encoded information
We applied our method to identify which regions of an image contribute most to the total information encoded by a population of visual neurons. For illustrative purposes, we used stimuli from the MNIST dataset and modelled a simple population of neurons with mean responses given by Af (w • x + b), where A and b are constants, f (•) is a sigmoid nonlinearity, and w is a linear filter representing the neuron's receptive field (RF). Neural responses were corrupted by Poisson noise. We simulated 49 neurons with RFs arranged in a uniform grid (Fig. 4A).
As a baseline, we first evaluated neural sensitivity using the diagonal of the Fisher information matrix (Fig. 4B-C, E-F). In this model, the Fisher information reduces to a weighted sum of squared RFs, where each neuron's contribution is scaled by its activation level. This yields characteristic "blob-like" patterns, with each blob centered on the neuron's RF.
We then used a diffusion model trained on MNIST to estimate the pixel-wise information decomposition, I local (x) (Fig 4D, G; additional images are shown in Supp Fig 2). This decomposition revealed that information was concentrated along object edges-regions where the decoded images (i.e. samples form the posterior) are most sensitive to small changes in the presented stimulus (cf. Fig. 2A). Unlike the Fisher information, our measure integrates how both the local sensitivity of neurons (via their RFs) and the statistical structure of the input (captured by the diffusion model), contribute towards the total encoded information.
Later, we investigated the behavior of our information decomposition on a diffusion model trained on natural images, with a model of recorded ganglion cell responses from the retina [15] (Appendix section C.6, and supplementary Figure 3). We observed qualitatively similar behavior to before, with I local peaking in regions of high local spatial contrast, around the edges of objects.
this section cite: ['b14']

Section: Discussion
We introduced a principled, information-theoretic measure of neural sensitivity to stimuli. This measure satisfies a core set of axioms that ensure interpretability and theoretical soundness. Crucially, the measure can be estimated using diffusion models, making it scalable to high-dimensional inputs and complex, non-linear neural populations. We empirically demonstrated how each axiom shapes interpretability through simple, illustrative examples. Finally, we show how the method can be applied in a high-dimensional setting to quantify the information encoded by a neural population about visual stimuli.
this section cite: []

Section: References
Ref_id:b0 Title: Optimal short-term population coding: When fisher information fails Year: (2002)
Ref_id:b1 Title: Synergy in a neural code Year: (2000)
Ref_id:b2 Title: Mutual information, fisher information, and population coding Year: (1998)
Ref_id:b3 Title: How much information is associated with a particular stimulus? Year: (2003)
Ref_id:b4 Title: Tuning curves, neuronal variability, and sensory coding Year: (2006)
Ref_id:b5 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2023)
Ref_id:b6 Title: Diffusion PID: Interpreting diffusion via partial information decomposition Year: (2024)
Ref_id:b7 Title: How to measure the information gained from one symbol Year: (1999)
Ref_id:b8 Title: Information geometry of the retinal representation manifold Year: (2023)
Ref_id:b9 Title: Context-dependent selectivity to natural images in the retina Year: (2022-09)
Ref_id:b10 Title: Accelerate: Training and inference at scale made simple, efficient and adaptable Year: (2022)
Ref_id:b11 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b12 Title: Information-theoretic bounds and approximations in neural population coding Year: (2018)
Ref_id:b13 Title: Solving linear inverse problems using the prior implicit in a denoiser Year: (2020)
Ref_id:b14 Title: Neural system identification for large populations separating "what"and "where Year: (2017)
Ref_id:b15 Title: Interpretable diffusion via information decomposition Year: (2024)
Ref_id:b16 Title: Coordinate invariance as a fundamental constraint on the form of stimulus-specific information measures Year: (2018)
Ref_id:b17 Title: Neural tuning and representational geometry Year: (2021)
Ref_id:b18 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b19 Title: Estimating high order gradients of the data distribution by denoising Year: (2021)
Ref_id:b20 Title: Power-law efficient neural codes provide general link between perceptual bias and discriminability Year: (2018)
Ref_id:b21 Title: Improving diffusion models for inverse problems using optimal posterior covariance Year: (2024)
Ref_id:b22 Title: Information and the accuracy attainable in the estimation of statistical parameters Year: (1992)
Ref_id:b23 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b24 Title: A mathematical theory of communication Year: (1948)
Ref_id:b25 Title: Axiomatic attribution for deep networks Year: (2017)
Ref_id:b26 Title: Diffusers: State-of-the-art diffusion models Year: (2022)
Ref_id:b27 Title: A bayesian observer model constrained by efficient coding can explain'anti-bayesian'percepts Year: (2015)
Ref_id:b28 Title: Mutual information, fisher information, and efficient coding Year: (2016)
Ref_id:b29 Title: Maximum a posteriori natural scene reconstruction from retinal ganglion cells with deep denoiser priors Year: (2022)
Ref_id:b30 Title: Fisher and shannon information in finite neural populations Year: (2012)
Ref_id:b31 Title: A proof of the fisher information inequality via a data processing argument Year: (1998)
