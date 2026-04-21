Title: New Bounds for Sparse Variational Gaussian Processes
Abstract: Sparse variational Gaussian processes (GPs) construct tractable posterior approximations to GP models. At the core of these methods is the assumption that the true posterior distribution over training function values f and inducing variables u is approximated by a variational distribution that incorporates the conditional GP prior p(f|u) in its factorization. While this assumption is considered as fundamental, we show that for model training we can relax it through the use of a more general variational distribution q(f|u) that depends on N extra parameters, where N is the number of training examples. In GP regression, we can analytically optimize the evidence lower bound over the extra parameters and express a tractable collapsed bound that is tighter than the previous bound. The new bound is also amenable to stochastic optimization and its implementation requires minor modifications to existing sparse GP code. Further, we also describe extensions to non-Gaussian likelihoods. On several datasets we demonstrate that our method can reduce bias when learning the hyperparameters and can lead to better predictive performance.

Section: Introduction
Gaussian processes (GPs) are nonparametric models for learning functions using Bayesian learning. Thanks to their flexibility and ability to quantify uncertainty, GPs have found many applications in machine learning (Rasmussen & Williams, 2006), spatial modeling (Cressie, 1993), computer experiments (O 'Hagan, 1978;Gramacy, 2020), Bayesian optimization (Jones et al., 1998;Garnett, 2023), robotics and control (Deisenroth & Rasmussen, 2011), unsupervised learning (Lawrence, 2005) and others.
Despite the numerous applications, GPs suffer from O(N 3 ) 1 Google DeepMind. Correspondence to: Michalis K. Titsias <mtitsias@google.com>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). time cost and O(N 2 ) storage where N is the number of training examples. This has originated a large body of research on scalable or sparse GP methods expanded in several decades; see e.g., Chapter 8 in Rasmussen & Williams (2006) for an early review and Heaton et al. (2018); Liu et al. (2020); Leibfried et al. (2022) for recent treatments. An important class of methods bases an approximation on a small set of M ≪ N inducing points (Csato & Opper, 2002;Lawrence et al., 2002;Seeger et al., 2003;Snelson & Ghahramani, 2006;Quiñonero-Candela & Rasmussen, 2005;Banerjee et al., 2008;Finley et al., 2009;Titsias, 2009;Hensman et al., 2013;Bui et al., 2017;Burt et al., 2020) that reduce the time complexity to O(N M 2 ) and the storage to O(N M ).
Among inducing point methods, the sparse variational Gaussian process (SVGP), introduced for standard regression (Titsias, 2009), applies variational inference to obtain a posterior approximation and selects hyperparameters and inducing points by maximizing an evidence lower bound. Unlike the prior approximation framework (Quiñonero-Candela & Rasmussen, 2005), SVGP leaves the GP prior unchanged and instead it reduces the cost to O(N M 2 ) by imposing a special structure on the variational distribution. This framework has been extended to stochastic gradient optimization (Hensman et al., 2013) and non-Gaussian likelihoods (Chai, 2012;Hensman et al., 2015;Lloyd et al., 2015;Dezfouli & Bonilla, 2015;Sheth et al., 2015). Also, it has been explained as KL minimization between stochastic processes (de G. Matthews et al., 2016).
An important aspect of the SVGP method is that it uses a special form for the variational distribution. It approximates the exact posterior distribution p(f, u|y) over the training function values f and the inducing variables u (see Section 2 for precise definitions) by a variational distribution of the form q(f, u) = p(f|u)q(u), where q(u) is some optimizable distribution over the inducing variables, while p(f|u) is the conditional GP prior. This special form of the variational approximation seems to be fundamental, and it has been applied also to more complex GP models, such as those with multiple outputs ( Álvarez et al., 2010;Nguyen & Bonilla, 2014;Yousefi et al., 2019), uncertain inputs (Titsias & Lawrence, 2010;Damianou et al., 2016) and multiple layers (Damianou & Lawrence, 2013;Salimbeni & Deisenroth, 2017). However, an open question regarding the SVGP framework is whether this particular form of variational distribution is really necessary to obtain scalable computations. The answer we give in this paper is that "it is not", since at least for training a GP model it can be relaxed.
To this end, we derive new variational bounds for training sparse GP regression models by replacing p(f|u) in the variational distribution with a more general conditional distribution q(f|u). This q(f|u) depends on N additional parameters (on top of the parameters of p(f|u)), i.e., as many as the training examples, and it is constructed to enable better covariance approximation of the underlying true factor p(f|u, y). We show how to analytically optimize over the N parameters and obtain a better posterior approximation together with a tighter collapsed evidence lower bound. The new bound is also amenable to stochastic gradient optimization, and its simple form suggests that it can be implemented with minor modifications to existing sparse GP code. We also describe extensions of the method to non-Gaussian likelihoods. Furthermore, we point out the concurrent work of Bui et al. (2025) who derived similar sparse GP approximations and variational training objectives by using the same form for the q(f|u) distribution.
The remainder of the paper is as follows. Section 2 provides an overview of GPs and the variational approach to sparse GPs using inducing points. Section 3 derives the new evidence lower bounds for training. Section 4 discusses connections with previous works. Section 5 presents experiments using several datasets showing that the new bounds can reduce underfitting bias and can lead to better predictive performance. Section 6 concludes with a discussion and suggestions for future work.
this section cite: ['b37', 'b9', 'b21', 'b27', 'b20', 'b15', 'b28', 'b37', 'b23', 'b31', 'b10', 'b29', 'b42', 'b45', 'b36', 'b1', 'b18', 'b48', 'b24', 'b3', 'b5', 'b48', 'b36', 'b24', 'b7', 'b25', 'b33', 'b16', 'b43', 'b13', 'b57', 'b34', 'b55', 'b12', 'b11', 'b39', 'b4']

Section: Background
A GP is a distribution over functions specified by a mean function m(x) and a covariance or kernel function k(x, x ′ ), where the kernel function is parametrized by θ. By assuming that m(x) = 0 we denote a GP draw as
f (x) ∼ GP(0, k(x, x ′ )). For a finite set of inputs X = {x n } N n=1 the distribution over the function values f = {f n } N n=1 (stored as N × 1 vector with f n := f (x n )) is the multivariate Gaussian p(f) = N (f|0, K ff ) where the N × N covariance matrix K ff has entries [K ff ] ij = k(x i , x j ).
We consider standard GP regression where we are given a set of training inputs X and corresponding noisy outputs y = {y n } N n=1 where y n ∈ R. Conditionally on the latent values f, these outputs follow a factorized Gaussian likelihood, p(y|f) = N n=1 N (y n |f n , σ 2 ) = N (y|f, σ 2 I). The joint distribution over outputs y and latent values f is p(y|f)p(f) = N (y|f, σ 2 I)N (f|0, K ff ).
(
To learn the hyperparameters (θ, σ 2 ) we can maximize the log marginal likelihood which is analytically available, log p(y) = p(y|f)p(f)df = log N (y|0, K ff + σ 2 I).
(2)
After training we can perform predictions at test inputs X * by first computing the posterior over the corresponding test function values f * :
p(f * |y) = p(f * |f)p(f|y)df = (3) N (f * |K f * f (K ff +σ 2 I) -1 f, K f * f * -K f * f (K ff +σ 2 I) -1 K ff * )
and then writing the predictive density as p(y * |y) = N (y * |f * , σ 2 I)p(f * |y)df * , which is the same as the above Gaussian but with σ 2 I added to the covariance.
While the log marginal likelihood and predictive density have closed-form expressions, they require the inversion of K ff + σ 2 I which costs O(N 3 ) and it is prohibitive for large datasets. Next we review methods using inducing points and particularly the variational approach (Titsias, 2009) that our method in Section 3 improves upon.
this section cite: ['b48']

Section: Sparse Variational Gaussian Process (SVGP)
The idea of inducing points is to base a GP approximation on a smaller set of M ≪ N function values; see e.g., Csato & Opper (2002); Seeger et al. (2003); Snelson & Ghahramani (2006); Quiñonero-Candela & Rasmussen (2005). Snelson & Ghahramani (2006) introduced pseudo inputs by instantiating extra GP function values u = {f (z m )} M m=1 evaluated at locations Z = {z m } M m=1 that can be optimized freely with gradient-based methods. However, the GP prior modification procedure (Quiñonero-Candela & Rasmussen, 2005;Snelson & Ghahramani, 2006) does not result in a rigorous approximation to the GP model. An alternative variational inference method (Titsias, 2009), next referred to as SVGPfoot_0 , does not modify the GP prior but instead it augments the model with extra function values u:
p(y, f, u) = p(y|f)p(f|u)p(u)
augmented joint ( 4)
p(f|u) = N (f|K ff K -1 uu u, K ff -K fu K -1 uu K uf ) cond. GP (5
) p(u) = N (u|0, K uu ) inducing GP prior (6)
where K uu is the M × M covariance matrix on the inducing inputs Z, K fu is the N × M cross covariance between points in X and Z, while K uf = K ⊤ fu . SVGP approximates the exact posterior p(f, u|y) by a variational distribution q(f, u) through the minimization of KL[q(f, u)||p(f, u|y)].
A critical assumption is the following choice for q:
q(f, u) = p(f|u)q(u),(7)
where q(u) is an optimizable M -dimensional variational distribution, while p(f|u) is the same conditional GP prior from Equation (5) that appears in the joint in (4). The KL minimization is expressed as the maximization of an evidence lower bound (ELBO) on the log marginal likelihood,
log p(y) ≥ p(f|u)q(u) log p(y|f) ¨p (f|u)p(u) ¨p (f|u)q(u) dfdu = q(u) log exp{ p(f|u) log p(y|f)df}p(u) q(u) du.
If we optimize over q(u) and obtain the optimal choice q * (u) ∝ exp p(f|u) log p(y|f)df p(u), then we can substitute this q * (u) in the last line above and express the so called collapsed bound, having the general form log p(y) ≥ F = log exp p(f|u) log p(y|f)df p(u)du which for the standard GP regression model takes the form
F = log N (y|0, Q ff + σ 2 I) DTC log lik - 1 2σ 2 tr (K ff -Q ff ) trace term ,(8)
where Q ff = K fu K -1 uu K uf is the M-rank Nystróm matrix. The first term in the bound is the deterministic training conditional (DTC) log likelihood (Seeger et al., 2003;Quiñonero-Candela & Rasmussen, 2005) while the second is a regularization term which, since tr(K ff -Q ff ) ≥ 0, promotes Q ff to stay close to K ff . The inducing points Z can be learned as variational parameters by maximizing the bound jointly with the hyperparameters (θ, σ 2 ), which requires O(N M 2 ) operations per optimization step. Hensman et al. (2013) further reduced the operations to O(M 3 ) per optimization step by applying stochastic minibatch training for maximizing the uncollapsed version of the bound; see Section 3.2.
To obtain the form of the GP posterior over any test function values f * we can first write the exact form
p(f * |y) = p(f * |f, u)p(f, u|y)dfdu,(9)
The SVGP method approximates p(f, u|y) by q(f, u) and therefore by plugging in this q into (9) we obtain
q(f * |y) = p(f * |f, u)p(f|u)q(u)dfdu = p(f * |u)q(u)du,(11)
where p(f * |u) = p(f * |f, u)p(f|u)df comes from the GP consistency. For completeness, in Appendix A we include further details about SVGP such as a derivation of the collapsed bound and the Gaussian form of the optimal q * (u).
We conclude this review of SVGP for regression with a couple of remarks that will be useful next. Remark 2.1. The approximation becomes exact when K ff = Q ff and the collapsed bound matches the log marginal likelihood in (2). However, to obtain good approximations we may need sufficiently large number of inducing points (Burt et al., 2020). Otherwise the bound will cause underfitting. For instance, as studied by Bauer et al. (2016) and Titsias (2009) the SVGP bound tends to overestimate the noise variance σ 2 . Remark 2.2. SVGP approximates p(f|u, y) in the exact posterior in (10) by the conditional GP p(f|u), in the variational posterior in (7), while q(u) is treated optimally by KL minimization. If p(f|u, y) = p(f|u) then KL[q(f, u)||p(f, u|y)] = 0 and the approximation becomes exact, meaning q(f * |y) = p(f * |y) for any f * = f * (X * ).
this section cite: ['b10', 'b42', 'b45', 'b36', 'b45', 'b36', 'b45', 'b48', 'b42', 'b36', 'b24', 'b5', 'b2', 'b48']

Section: Proposed Method: Tighter Bounds
Remark 2.1 suggests that it would be useful to tighten the collapsed bound in order to reduce underfitting bias and match better exact GP training. Remark 2.2 suggests that one way to tighten the bound is to replace p(f|u), in the variational approximation in (7), with another distribution that can better approximate p(f|u, y). Next we develop a method that does this while keeping the cost unchanged.
Let us write the exact form of p(f|u, y). By noting that this quantity is the exact posterior over f in a GP regression model with joint p(y|f)p(f|u) we conclude that this is
p(f|u, y) = N f|m(y, u), ( K -1 ff + 1 σ 2 I) -1 , where m(y, u) = E[f|u]+ K ff ( K ff +σ 2 I) -1 (y-E[f|u]) with E[f|u] = K fu K -1 uu u and K ff = K ff -Q ff .
Note that under this notation, p(f|u) = N (f|E[f|u], K ff ). We will construct a new q(f|u) that keeps the same mean E[f|u] as p(f|u) but it replaces K ff with a closer approximation to the covariance ( K
-1 ff + 1 σ 2 I) -1 of p(f|u, y).
We first write this matrix as
( K -1 ff + 1 σ 2 I) -1 = K 1 2 ff (I + 1 σ 2 K ff ) -1 K 1 2 ff . (12
)
Then we approximate the inverse (I
+ 1 σ 2 K ff ) -1 by a di- agonal matrix V = diag(v 1 , . . . , v N ) of N variational pa- rameters v i > 0. In other words, in the initial q(f, u) = p(f|u)q(u) we will replace p(f|u) by q(f|u) = N (f|K fu K -1 uu u, (K ff -Q ff ) 1 2 V(K ff -Q ff ) 1 2 ).(13)
The ELBO now is written as
q(f|u)q(u) log p(y|f)p(f|u)p(u) q(f|u)q(u) dfdu = q(u) log e E q(
f|u) [log p(y|f)] p(u) q(u) -KL[q(f|u)||p(f|u)] du and the challenge is to see whether KL[q(f|u)||p(f|u)] and E q(f|u) [log p(y|f)] are computable in O(N M 2 ) time. We have the following results (proofs are in Appendix B). Lemma 3.1. KL[q(f|u)||p(f|u
)] = 1 2 N i=1 (v i -log v i -1). Lemma 3.2. Let us denote the diagonal elements of K ff - Q ff as k ii -q ii for i = 1, . . . , N . Then E q(f|u) [log p(y|f)] = log N (y|K fu K -1 uu u, σ 2 I) - 1 2σ 2 N i=1 v i (k ii -q ii ). (14
)
By combining the two lemmas the full bound is written as
q(u) log N (y|K fu K -1 uu u, σ 2 I)p(u) q(u) du - 1 2 N i=1 v i 1 + k ii -q ii σ 2 -log v i -1 .(15)
Proposition 3.3. Maximizing the bound in (15) with respect to q(u) and each v i results in the optimal settings q * (u) ∝ N (y|K fu K -1 uu u, σ 2 I)p(u) and
v * i = 1 + kii-qii σ 2 -1
. By substituting these values back to ( 15) we obtain
F new = log N (y|0, Q ff +σ 2 I)- 1 2 N i=1 log 1 + k ii -q ii σ 2 .(16)
The first term is the DTC log likelihood as in the original bound in ( 8), but the regularization term makes the bound tighter, i.e., log p(y) ≥ F new ≥ F, due to the inequality log(a + 1) ≤ a. Also since log(a + 1) < a for all a > 0, if K ff ̸ = Q ff (so there is at least one k ii -q ii > 0), then F new > F. This means that F new is strictly better than F unless both bounds match exactly the log marginal likelihood.
Clearly, F new has O(N M 2 ) cost and its implementation requires a minor modification to the initial bound. The optimal q * (u) is the same as in the initial SVGP method, while an interpretation of the optimal v * i values is the following. Remark 3.4. The diagonal matrix V * (with the optimal v * i values in its diagonal) is the inverse obtained after zeroing out the off-diagonal elements of I 12). Also note that in the ordering of positive definite matrices it holds V * ≤ I, from which it follows that q(f|u) has smaller covariance than p(f|u) and more accurately approximates the covariance of p(f|u, y). The latter as implied by Equation ( 12), has also smaller covariance than p(f|u).
+ 1 σ 2 (K ff -Q ff ), i.e., V * = diag[I + 1 σ 2 (K ff -Q ff )] -1 which approximates (I + 1 σ 2 (K ff -Q ff )) -1 in Equation (
this section cite: []

Section: Predictions
To perform predictions we will be using the same predictive posterior from Equation (11), i.e., q(f * |y) = p(f * |u)q(u)du, where the optimal q * (u) (see Appendix A) is exactly the same as in the standard SVGP method. The alternative expression (and strictly speaking more appropriate since our variational approximation is q(f|u)q(u)) is given by
q high cost (f * |y) = p(f * |f, u)q(f|u)q(u)dfdu. (17
)
But this is expensive since it has cost O(N 3 ). The reason is that p(f * |f, u)q(f|u)df does not simplify anymore since q(f|u) is not the conditional GP, which means that p(f * |f, u) and q(f|u) are not consistent under the GP prior. Nevertheless, q(f * |y) and q high cost (f * |y) have exactly the same mean, since q(f|u) and p(f|u) have the same mean, but the tractable q will give higher variances than q high cost .
this section cite: []

Section: Stochastic Minibatch Training
The initial SVGP method (Titsias, 2009) does the training in a batch mode where all data are used in each optimization step. Stochastic optimization using minibatches was proposed by Hensman et al. (2013). Here, we apply our new approximation to this stochastic method.
We start from Equation ( 15), and substitute only the optimal values for each v i without using the optimal setting for q(u). This results in the uncollapsed bound
N i=1 E q(u) [log N (y i |k fiu K -1 uu u, σ 2 )] - 1 2 log 1 + k ii -q ii σ 2 -KL[q(u)||p(u)],(18)
where k fiu is the 1 × M vector of all kernel values between the training input x i and the inducing inputs Z, while the expectation under q(u) in the first line is analytic; see Hensman et al. (2013). The above bound is strictly better than the previous uncollapsed bound in Hensman et al. (2013), since
-1 2σ 2 (k ii -q ii ) ≤ -1 2 log 1 + kii-qii σ 2
. Based on the above we can apply stochastic gradient methods to optimize q(u) and the hyperparameters by subsampling data minibatches to deal with the sum over the N training points, i.e., at each iteration we use the stochastic ELBO:
N |B| i∈B E q(u) [log N (y i |k fiu K -1 uu u, σ 2 )] - 1 2 log 1 + k ii -q ii σ 2 -KL[q(u)||p(u)],(19)
where B denotes a minibatch.
The most common parametrization of q(u) is q(u) = N (u|m, S) where the mean vector m and covariance matrix S are variational parameters. Another popular parametrization, for instance used as the default in GPflow (de G. Matthews et al., 2017), is the whitened parametrization that we consider in our experiments. For any choice of q(u), the new bound is always tighter than its corresponding previous uncollapsed bound and requires minor modifications to existing implementations, i.e., to replace the previous term
-1 2σ 2 (k ii -q ii ) with -1 2 log 1 + kii-qii σ 2 .
this section cite: ['b48', 'b24', 'b24', 'b24']

Section: Non-Gaussian Likelihoods
Consider a factorized likelihood p(y|f) = N i=1 p(y i |f i ) where p(y i |f i ) is non-Gaussian, e.g., Bernoulli for binary outputs or Poisson for counts. In this non-conjugate setting the sparse variational GP approximation imposes the same form for the variational distribution, i.e., q(f, u) = p(f|u)q(u) where p(f|u) is the conditional GP prior. As shown in several works (Chai, 2012;Hensman et al., 2015;Lloyd et al., 2015;Dezfouli & Bonilla, 2015;Sheth et al., 2015), this leads to the bound
N i=1 E q(fi) [log p(y i |f i )] -KL[q(u)||p(u)], (20
)
where q(f i ) = p(f|u)q(u)df -i du is the marginal over f i := f (x i ) with respect to the approximate posterior q(f, u). Given that q(u) is Gaussian with mean m and covariance S, q(f i ) can be computed fast in O(M 2 ) time (after precomputing the Cholesky factorization of K uu ) as follows
q(f i ) = N (f i |k fiu K -1 uu m, k ii -q ii + k fiu K -1 uu SK -1 uu k ufi ).
(21) For the discussion next it is useful to observe that the efficiency when computing q(f i ) comes from p(f|u) being a conditional GP prior, so expressing p(f i |u) is trivial.
Suppose now that we wish to impose the more structured variational approximation q(f, u) = q(f|u)q(u) where q(u) = N (u|m, S) and q(f|u) is given by Equation ( 13). The bound can be written as
N i=1 E q(fi) [log p(y i |f i )] - 1 2 N i=1 (v i -log v i -1) -KL[q(u)||p(u)],(22)
where we used the fact that KL[q(f|u)||p(f|u)] is obtained from Lemma 3.1. The above bound is not computationally efficient since the marginal q(f i ) = q(f|u)q(u)df -i du has O(N 3 ) cost. This is because the marginalization q(f i |u) = q(f|u)df -i cannot be trivially expressed, due to the complex structure of the covariance (K ff -Q ff )
1 2 V(K ff - Q ff ) 1 2 in q(f|u).
To overcome this, we will use a simplified version of q(f|u), in which we choose a spherical V = vI with v > 0. Then, things become tractable.
Proposition 3.5. Let q(f|u) = N (f|K fu K -1 uu u, v(K ff -Q ff )) for v > 0. Then (22) is computed in O(N M 2 ) time as N i=1 E q(fi) [log p(y i |f i )] - N 2 (v -log v -1) -KL[q(u)||p(u)],(23)
where the marginal is q(
f i ) = N (f i |k fiu K -1 uu m, v(k ii - q ii ) + k fiu K -1 uu SK -1 uu k ufi ).
The parameter v multiplies the term k ii -q ii inside the variance of q(f i ), and it also appears in the regularization term 23) reduces to ( 20), while by optimizing over v it can become a tighter bound. The optimization of v is done jointly with the remaining parameters m, S, Z, θ using gradient-based methods. Stochastic gradients can also be used by subsampling minibatches to deal with the sum
-N 2 (v -log v -1). If v = 1 the bound in (
N i=1 E q(fi) [log p(y i |f i )]
and reduce the complexity to O(M 3 ).
The above framework can be extended to non-conjugate models having multiple functions, such as multi-class GP classification, by introducing a separate v parameter per GP function. In our experiments, we consider only singlefunction non-conjugate GP models and we leave the experimentation with more complex models for future work.
this section cite: ['b7', 'b25', 'b33', 'b16', 'b43']

Section: Related Work
Several recent works on sparse GPs focus on constructing efficient inducing points, such as works that place inducing points on a grid (Wilson & Nickisch, 2015;Evans & Nair, 2018;Gardner et al., 2018), construct inter-domain Fourier features (Lázaro-Gredilla & Figueiras-Vidal, 2009;Hensman et al., 2018), provide Bayesian treatments to inducing inputs (Rossi et al., 2021) or use nearest neighbor sparsity structures (Tran et al., 2021;Wu et al., 2022). There exist also algorithms that allow to increase the number of inducing points using the decoupled method (Cheng & Boots, 2017;Havasi et al., 2018) and the related orthogonally decoupled approaches (Salimbeni et al., 2018;Shi et al., 2020;Sun et al., 2021;Tiao et al., 2023). Our contribution is orthogonal to these previous methods since we relax the conditional GP prior assumption in the posterior variational approximation. This means that our method could be used to improve previous variational sparse GP approaches, as the ones mentioned above as well as earlier schemes that select inducing points from the training inputs (Cao et al., 2013;Chai, 2012;Schreiter et al., 2016). Zhu et al. (2023) proposed inducing points GP approximations that change the conditional GP p(f|u) in the variational approximation to a modified conditional GP that uses different kernel hyperparameters in its mean vector. Note that our method differs since our q(f|u) directly tries to construct a better approximation to the exact posterior p(f|u, y), using the extra V variational parameters, without changing the kernel hyperparameters; see Section 3. More importantly, our method has O(N M 2 ) cost, while the ELBO in Zhu et al. (2023) (see Section 3.1 and Appendix A.1 in their paper) has cubic cost O(N 3 ) since it depends on the inverse of K ff -Q ff (denoted as Knn in their paper). Artemev et al. (2021) derived an upper bound on the log determinant log |K ff + σ 2 I| in the exact GP log marginal likelihood and obtained the following tighter upper bound to the initial trace regularization term -1 2σ 2 tr (K ff -Q ff ):
- N 2 log 1 + tr(K ff -Q ff ) N σ 2 . (24
)
Our bound is tighter since from Jensen's inequality it holds
-N 2 log 1 + tr(K ff -Q ff ) N σ 2 ≤ -1 2 N i=1 log 1 + kii-qii σ 2
. Further, the above regularization term can be interpreted as a restricted special case of our method, obtained through a q(f|u) from Equation (13) where the diagonal matrix V is constrained to be spherical V = vI; see Appendix B.4. Finally note, that unlike (24) (where the sum is inside the logarithm) our bound allows to apply stochastic optimization as described in Section 3.2. Finally, Bui et al. (2017) used power expectation propagation that minimizes α-divergence and derived an approximation to the log marginal likelihood that interpolates between the FITC (α = 1) log marginal likelihood (Snelson & Ghahramani, 2006;Quiñonero-Candela & Rasmussen, 2005) and the standard collapsed variational bound in (8) (α → 0). This approximation uses the regularization term
- 1 -α 2α N i=1 log 1 + α k ii -q ii σ 2 . (25
)
This is different from ours since there is no value of α such that the two regularization terms will become equal. For example, note that for α → 0, Equation (25) reduces to -1 2σ 2 tr (K ff -Q ff ) as discussed in Bui et al. (2017).
this section cite: ['b53', 'b17', 'b19', 'b30', 'b26', 'b38', 'b50', 'b54', 'b8', 'b22', 'b40', 'b44', 'b46', 'b47', 'b6', 'b7', 'b41', 'b56', 'b0', 'b3', 'b45', 'b36', 'b3']

Section: Experiments

this section cite: []

Section: Illustration in 1-D Regression
In the first regression experiment we consider the 1-D Snelson dataset (Snelson & Ghahramani, 2006). We took a subset of 40 examples of this dataset and we fitted the exact GP with the squared exponential kernel k(x, x ′ ) = σ 2 f exp(-(x-x ′ ) 2 2ℓ 2 ). We also fitted sparse variational GPs with either the standard collapsed bound (Titsias, 2009) from Equation (8 Figure 1 shows the results. Note that both SGPR and SGPRnew find similar inducing point locations. But SGPR-new, as a tighter bound (see panel (e)), is able to reduce some bias when estimating the hyperparameters since it finds a noise variance σ 2 closer to the one by exact GP (see panel (f)). This results in better predictions that match better the exact GP, as shown by the comparative visualization in panel (d). From panel (d), observe that both the mean and variances of SGPR-new are closer to the exact GP than SGPR.
this section cite: ['b45', 'b48']

Section: Medium Size Regression Datasets
To further investigate the findings from the previous section, we consider three medium size real-world UCI regression datasets (Pol, Bike, and Elevators) with roughly 10k training data points each, and for which we can still run the exact GP. We choose the ARD squared exponential kernel k(x, x ′ ) = σ 2 f exp(-
d i=1 (xi-x ′ i ) 2 2ℓ 2 i
). We run all three previous methods (Exact GP, SGPR, SGPR-new) five times with different random train-test splits; see Appendix D for experimental details. We also include in the comparison a fourth method (discussed in Related Work) which is the Artemev et al. (2021)'s bound (SGPR-artemev) that does training using the collapsed bound from Equation (36) in Appendix B.4. All sparse GP methods use M = 1024 or M = 2048 inducing points initialized by k-means. Figure 2 (in the first two lines) shows the objective function and the noise variance σ 2 across 10k optimization steps using Adam with base learning rate 0.01 and for M = 1024. For SGPR-new, the third line in Figure 2 shows histograms of the estimated final values of the optimal v i variational parameters. Figure 4 in Appendix D.1 shows the corresponding plots for M = 2048. We observe that for Pol and Bike, SGPR-new matches closer the exact GP training than SGPR and SGPR-artemev. Specifically, SGPR-new gives higher ELBO and estimates the noise variance with reduced underfitting bias. For the Elevators dataset, M = 1024 inducing points were enough for sparse GP methods to closely match exact GP training. This happens because in this case Q ff  accurately approximates K ff , i.e., the elements k ii -q ii get close to zero. For this latter dataset, observe that since the k ii -q ii values are close to zero the corresponding v i values are concentrated around one as shown by the corresponding (right-most) histogram in Figure 2.
Table 1 reports test log-likelihood predictions which show that SGPR-new outperforms SGPR and SGPR-artemev.
this section cite: ['b0']

Section: Large Scale Regression Datasets
We consider 8 regression datasets, with training data sizes ranging from tens of thousands to millions. We implemented the stochastic optimization versions of the two scalable sparse GP methods: (i) the one that trains using the previous uncollapsed bound from Hensman et al. (2013) (SVGP) and (ii) our new bound from Equation (18) (SVGP-new). We denote these stochastic optimization versions by SVGP to distinguish them from the corresponding SGPR methods that use the more expensive collapsed bounds. We run the SVGP methods with M = 1024 and 2048 inducing points, Matern3/2 kernel with common lengthscale, minibatch size 1024, Adam with base learning rate 0.01 and 100 epochs. These experimental settings match the ones in Wang et al. (2019) and Shi et al. (2020) as further described in Appendix D.2. Table 2 reports the test log likelihood scores for all datasets. In the comparison we also included two strong baselines from Table 2 in Shi et al. (2020), i.e., SOLVE-GP and ODVGP (Salimbeni et al., 2018).
From the predictive log likelihood scores in Table 2 and also the corresponding Root Mean Squared Error (RMSE) scores reported in Table 4 in Appendix D.2, we can conclude that training with the new SVGP-new variational bound provides a clear improvement compared to training with the previous SVGP bound. Note that this improvement requires no change in the computational cost, and in fact there is
parameters vi = 1 + k ii -q ii σ 2 -1 .
only a minor modification needed to be done in an existing SVGP implementation in order to run SVGP-new.
this section cite: ['b24', 'b40']

Section: Poisson Regression
We consider a non-Gaussian likelihood example where the output data are counts modeled by a Poisson likelihood
p(y|f) = N i=1 e f i yi! e -e f i
where the log intensities values follow a GP prior. For such case the new variational approximation includes a single additional variational parameter denoted by v, which is optimized together with the remaining parameters; see Section 3.3. We will compare training with the new ELBO from Equation (23) (we denote this method by SVGP-new) with the standard ELBO that is obtained by restricting v = 1 (SVGP).
Firstly, we consider an artificial example of 50 observations with 1-D inputs placed in the grid [-10, 10] where counts are generated using Poisson intensities given by λ(x) = 3.5 + 3 sin(x). We train the GP model with the SVGP bound and the proposed SVGP-new bound using 6 inducing points initialized to the same values for both methods; see Appendix D.3. Figure 3(left) shows the observed counts together with the predictions obtained by SVGP, SVGP-new and non-sparse variational GP (Full GP). From this figure and from the ELBO values, we observe that SVGP-new remains closer to Full GP.
Secondly, we consider a real dataset (NYBikes) about bicycles crossings going over bridges in New York Cityfoot_1 . This dataset is a daily record of the number of bicycles crossing into or out of Manhattan via one of the East River bridges over a period 9 months. The data contains 210 points and  we randomly choose 90% for training and 10% for test. We apply GP Poisson regression for the Brooklyn bridge counts where the input vector x is taken to be two-dimensional consisted of maximum and minimum daily temperatures. We train the sparse GPs with either SVGP or SVGP-new and with M = 8, 16, 32 inducing points initialized by k-means. Since the dataset is small we also run the non-sparse Full GP. The ELBO across iterations in Figure 3 (right) and the test log likelihood scores (
Table 5 in Appendix D.3) indicate that SVGP-new provides a better approximation than SVGP.
this section cite: []

Section: Conclusions
We have presented a method that relaxes the conditional GP assumption in the approximate distribution in sparse variational GPs. This leads to tighter collapsed and uncollapsed bounds, that maintain the computational cost with the previous bounds and can reduce training underfitting. For future work an interesting topic is to apply our method to more complex GP models, such as those with multiple outputs, with uncertain inputs and deep GPs. For the Bayesian GP-LVM, where the collapsed closed form bound has strong similarities with the previous GP regression collapsed bound in Equation ( 8), deriving a new collapsed bound is tractable as described in Appendix B.5. Finally, it might be useful to investigate whether theoretical convergence results on sparse GPs (Burt et al., 2020;Wild et al., 2023), can be improved given the new collapsed lower bound.
By maximizing wrt v we obtain v * = 1 + tr
(K ff -Q ff ) N σ 2 -1 , and by substituting this back into the bound we obtain Artemev et al. (2021)'s tighter bound on the initial trace regularization term. Overall this collapsed bound has the form log p(y) ≥ log N (y|0, Q ff + σ 2 I) -N 2 log 1 + tr(K ff -Q ff ) N σ 2 . (36)
This collapsed bound is what the method SGPR-artemev is using in Section 5.2. Note that Artemev et al. (2021) propose also additional but more expensive bounds for the first DTC log likelihood term that require running conjugate gradients. We do not consider those in our comparisons (such bounds could be used in all SGPR bounds since all share the same DTC log likelihood term) as they have higher cost.
this section cite: ['b5', 'b52', 'b0']

Section: References
Ref_id:b0 Title: Tighter bounds on the log marginal likelihood of gaussian process regression using conjugate gradients Year: (2021-07)
Ref_id:b1 Title: Gaussian predictive process models for large spatial data sets Year: (2008)
Ref_id:b2 Title: Under-standing probabilistic sparse gaussian process approximations Year: (2016)
Ref_id:b3 Title: A unifying framework for Gaussian process pseudo-point approximations using power expectation propagation Year: (2017)
Ref_id:b4 Title: Tighter sparse variational gaussian processes Year: (2025)
Ref_id:b5 Title: Convergence of sparse variational inference in gaussian processes regression Year: (2020)
Ref_id:b6 Title: Efficient optimization for sparse gaussian process regression Year: (2013)
Ref_id:b7 Title: Variational multinomial logit gaussian process Year: (2012)
Ref_id:b8 Title: Variational inference for Gaussian process models with linear complexity Year: (2017)
Ref_id:b9 Title: Statistics for spatial data Year: (1993)
Ref_id:b10 Title: Sparse online Gaussian processes Year: (2002)
Ref_id:b11 Title: Deep Gaussian processes Year: (2013-05-01)
Ref_id:b12 Title: Variational inference for latent variables and uncertain inputs in gaussian processes Year: (2016)
Ref_id:b13 Title: On sparse variational methods and the kullback-leibler divergence between stochastic processes Year: (2016)
Ref_id:b14 Title: Gpflow: A gaussian process library using tensorflow Year: (2017)
Ref_id:b15 Title: PILCO: A modelbased and data-efficient approach to policy search Year: (2011)
Ref_id:b16 Title: Scalable inference for gaussian process models with black-box likelihoods Year: (2015)
Ref_id:b17 Title: Scalable Gaussian processes with gridstructured eigenfunctions (GP-GRIEF) Year: (2018)
Ref_id:b18 Title: Improving the performance of predictive process modeling for large datasets Year: (2009)
Ref_id:b19 Title: Product kernel interpolation for scalable Gaussian processes Year: (2018)
Ref_id:b20 Title:  Year: (2023)
Ref_id:b21 Title: Gaussian Process Modeling, Design and Optimization for the Applied Sciences Year: (2020)
Ref_id:b22 Title: Deep Gaussian processes with decoupled inducing inputs Year: (2018)
Ref_id:b23 Title: A case study competition among methods for analyzing large spatial data Year: (2018)
Ref_id:b24 Title: Gaussian processes for big data Year: (2013)
Ref_id:b25 Title: Scalable variational Gaussian process classification Year: (2015)
Ref_id:b26 Title: Variational fourier features for gaussian processes Year: (2018)
Ref_id:b27 Title: Efficient global optimization of expensive black-box functions Year: (1998)
Ref_id:b28 Title: Probabilistic non-linear principal component analysis with Gaussian process latent variable models Year: (2005-11)
Ref_id:b29 Title: Fast sparse Gaussian process methods: the informative vector machine Year: (2002)
Ref_id:b30 Title: Inter-domain gaussian processes for sparse inference using inducing features Year: (2009)
Ref_id:b31 Title: A tutorial on sparse gaussian processes and variational inference Year: (2022)
Ref_id:b32 Title: When gaussian process meets big data: A review of scalable gps Year: ()
Ref_id:b33 Title: Variational inference for gaussian process modulated poisson processes Year: (2015-07)
Ref_id:b34 Title: Collaborative multi-output gaussian processes Year: (2014)
Ref_id:b35 Title: Curve fitting and optimal design for prediction Year: (1978)
Ref_id:b36 Title: A unifying view of sparse approximate Gaussian process regression Year: (2005)
Ref_id:b37 Title: Gaussian Processes for Machine Learning Year: (2006)
Ref_id:b38 Title: Sparse gaussian processes revisited: Bayesian approaches to inducing-variable approximations Year: (2021-04)
Ref_id:b39 Title: Doubly stochastic variational inference for deep Gaussian processes Year: (2017)
Ref_id:b40 Title: Orthogonally decoupled variational Gaussian processes Year: (2018)
Ref_id:b41 Title: Efficient sparsification for gaussian process regression Year: (2016)
Ref_id:b42 Title: Fast forward selection to speed up sparse gaussian process regression Year: (2003-01)
Ref_id:b43 Title: Sparse variational inference for generalized gp models Year: (2015-07)
Ref_id:b44 Title: Sparse orthogonal variational inference for gaussian processes Year: (2020-08)
Ref_id:b45 Title: Sparse Gaussian processes using pseudo-inputs Year: (2006)
Ref_id:b46 Title: Scalable variational gaussian processes via harmonic kernel decomposition Year: (2021-07)
Ref_id:b47 Title: Spherical inducing features for orthogonally-decoupled Gaussian processes Year: (2023-07)
Ref_id:b48 Title: Variational learning of inducing variables in sparse Gaussian processes Year: (2009)
Ref_id:b49 Title: Bayesian gaussian process latent variable model Year: (2010-05)
Ref_id:b50 Title: Sparse within sparse gaussian processes using neighbor information Year: (2021-07)
Ref_id:b51 Title: Exact Gaussian processes on a million data points Year: (2019)
Ref_id:b52 Title: Connections and equivalences between the nyström method and sparse variational gaussian processes Year: (2023)
Ref_id:b53 Title: Kernel interpolation for scalable structured Gaussian processes (KISS-GP) Year: (2015)
Ref_id:b54 Title: Variational nearest neighbor Gaussian process Year: (2022-07)
Ref_id:b55 Title: Multi-task learning for aggregated data using gaussian processes Year: (2019)
Ref_id:b56 Title: Variational gaussian processes with decoupled conditionals Year: (2023)
Ref_id:b57 Title: Efficient multioutput gaussian processes through variational inducing kernels Year: (2010-05)
