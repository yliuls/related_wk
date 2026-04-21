Title: Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data
Abstract: Different gradient-based methods for optimizing overparameterized models can all achieve zero training error yet converge to distinctly different solutions inducing different generalization properties. We provide the first complete characterization of implicit optimization bias for p-norm normalized steepest descent (NSD) and momentum steepest descent (NMD) algorithms in multi-class linear classification with cross-entropy loss. Our key theoretical contribution is proving that these algorithms converge to solutions maximizing the margin with respect to the classifier matrix's p-norm, with established convergence rates. These results encompass important special cases including Spectral Descent and Muon, which we show converge to max-margin solutions with respect to the spectral norm. A key insight of our contribution is that the analysis of general entry-wise and Schatten p-norms can be reduced to the analysis of NSD/NMD with max-norm by exploiting a natural ordering property between all p-norms relative to the max-norm and its dual sumnorm. For the specific case of descent with respect to the max-norm, we further extend our analysis to include preconditioning, showing that Adam converges to the matrix's max-norm solution. Our results demonstrate that the multi-class linear setting, which is inherently richer than the binary counterpart, provides the most transparent framework for studying implicit biases of matrix-parameter optimization algorithms.

Section: Introduction
The ever-increasing training cost of large language models (LLMs) has demanded better optimizer designs with improved performance and efficiency [11,1,23]. The de facto standard optimizers for deep learning training are Adam and AdamW [33,39]. However, these algorithms that employ diagonal preconditioners to independently adjust the learning rate of each coordinate, may fail to capture their inter-dependencies and fully leverage the geometry of the loss landscape [79]. This has spurred a series of research efforts on improving Adam or AdamW's computational efficiency [20,24,51,80], with LLM-training as the target application domain [31,67,46,37].
A noticeable work by Jordan et al. [31] proposed the Muon optimizer, which was shown to have remarkable performances on NanoGPT benchmarks. More recently, it has been shown that Muon can be used for large-scale LLM training with the potential to replace AdamW as the standard choice [37].
The key step in Muon is to orthogonalize the updates via the Newton-Schulz iteration [31,7]. More precisely, the update (denoted as ∆) is (approximately) replaced by the product of its singular-vector Table 1: Summary of margin convergence rates for NSD and NMD algorithms of different norm constraints for linear multiclass separable data with the CE loss. The (truncated) SVDs of the gradient and momentum are denoted as ∇ = U ΣV T and M = Ũ Σ Ṽ T respectively.
Method Norm Constraint Update ∆ Reference Rate 2 NGD Unit ∥•∥ 2 -ball ∇ ∥∇∥2
Hazan et al. [25] -NMD-GD M ∥M ∥2
Cutkosky and Mehta [14] -SignGD Unit ∥•∥ max -ball sign(∇) Bernstein et al. [8] -Signum sign(M ) Bernstein et al. [8] -Spectral-GD Unit |||•||| ∞ -ball U V T Bernstein and Newhouse [7] O( log t+n t 1/2 ) Muon 1 Ũ Ṽ T Jordan et al. [31] O( d log t+dn t 1/2
) 1 We consider EMA-style momentum of the form (5). 2 NGD and SignGD rates are the same as Spectral-GD; Signum and NMD-GD rates are the same as Muon.
matrices U V T (where the (truncated) singular value decomposition (SVD) of ∆ is ∆ = U ΣV T ).
Even though the benefits of orthogonalization are not fully understood, Jordan et al. [31] pointed out that it could promote updates in directions of small magnitudes given the weight matrices are typically low-rank. Moreover, if the above SVD approximation is exact and gradient accumulations are turned off, then Muon becomes spectral descent [13,7], which is the (normalized) steepest descent w.r.t the spectral norm [7]. As noted by Bernstein and Newhouse [7], spectral descent is also Shampoo (which won the AlgoPerf competition [52,15]) without accumulations in preconditioners. Thus, Muon can be viewed as (approximate) Shampoo when both optimizers are without accumulations. In essence, we observe that one important ingredient of Muon and Shampoo (without accumulations) is the spectral-descent step,
W † = W -ηU V T where ∇L(W ) = U ΣV T .
Theoretical investigations of spectral descent or Muon mainly focus on characterizing the convergence rates of the algorithm (e.g., the rate of decrease of the gradient norm in the non-convex setting [3,36,46]). However, modern machine learning models are overparameterized, leading to multiple weight configurations that achieve identical training loss but exhibit markedly different generalization properties [78,6]. The key insight is that gradient-based methods inherently prefer "simple" solutions according to optimizer-specific notions of simplicity. Understanding this implicit bias/regularization requires analyzing not just loss convergence, but the geometric trajectory of parameter updates throughout training. To this end, our work aims to address the fundamental question:
What is the implicit bias of spectral descent (and its momentum variants) in linear multiclass classification with separable data and cross-entropy loss?
The multiclass setting where the parameter is a matrix, is a natural place to study the class of spectraldescent algorithms, and provides an inherently richer setting. Our work captures this richness by establishing convergence with respect to not only entry-wise matrix norms, but also matrix Schatten norms. Hence, while the focus is on spectral descent and Muon, the analysis establishes implicit bias rates for a wide family of algorithms (Table 1), and we state the results in the most general form from the perspective of steepest descent with (unit) norm-ball constraints. Our contributions are as follows:
1. For multiclass separable data trained with the cross-entropy (CE) loss, we show that the iterates of normalized steepest descent (NSD) defined with respect to (w.r.t.) any matrix entry-wise or Schatten norms converge to a solution that maximizes the margin defined w.r.t. the same norm, with a rate O( 1 t 1/2 ). This includes sign descent (entry-wise max-norm) [8], normalized gradient descent (entry-wise 2-norm) [25], and spectral descent (Schatten ∞-norm) [7] as special cases.
To achieve this, we introduce a unified analysis framework that relates entry-wise and Schatten p-norms to the entry-wise max-norm, and construct a proxy function for the loss that closely traces both its value and gradient. We also show the same machinery applies to other multiclass losses such as the exponential loss [41] and the PairLogLoss [72]. 2. Under the same setting, we utilize the same framework and proxy function to show that a O( 1 t 1/2 ) margin convergence rate also holds for normalized momentum steepest descent (NMD). This includes the following algorithms in analogy to the ones above: sign momentum descent [8], normalized momentum gradient descent [14], and Muon [31]. The key step of the analysis is to use the proxy function to bound the sum-norm difference between the gradient and the momentum (i.e., the exponential moving averages (EMA) of the gradient), which translates to a bound on the dual norm through the fundamental norm-relationships used in the study of NSD. The margin convergence rates of various algorithms are summarized in Table 1. Furthermore, we extend the analysis to Adam (without the stability constant) and show its iterates maximize the margin w.r.t. the matrix max-norm (proof details and numerical validations in App. G).
3. We experimentally verify our theoretical predictions across all considered algorithms. First, for sign descent (SignGD) and Signum, we demonstrate that solutions favor the max-norm margin over the 2-norm margin-the opposite behavior to normalized gradient descent (NGD) and normalized momentum gradient descent (NMD-GD). Moreover, we show that both spectral descent (Spectral-GD) and Muon favor the spectral-norm margin over the other norms. We further extend the experiments to the non-linear setting with a two-layer neural network. We observe the (unnormalized) spectral-norm margin of Spectral-GD and Muon grow faster than other algorithms. Hence, the norm-preference trend in the linear setting can also exhibit in the non-linear setting.
this section cite: ['b10', 'b0', 'b22', 'b32', 'b38', 'b78', 'b19', 'b23', 'b50', 'b79', 'b30', 'b66', 'b45', 'b36', 'b30', 'b36', 'b30', 'b6', 'b24', 'b13', 'b7', 'b7', 'b6', 'b30', 'b30', 'b12', 'b6', 'b6', 'b6', 'b51', 'b14', 'b2', 'b35', 'b45', 'b77', 'b5', 'b7', 'b24', 'b6', 'b40', 'b71', 'b7', 'b13', 'b30']

Section: Preliminaries
Notations Matrices, vectors, and scalars are denoted by A, a, and a respectively. For matrix A, we denote its (i, j)-th entry as A[i, j], and for vector a, its i-th entry as a[i] or a i . We consider entry-wise matrix p-norms defined as ∥A∥ p = ( i,j |A[i, j]| p ) 1/p . Central to our results are: the infinity norm, denoted as ∥A∥ max := ∥A∥ ∞ = max i,j |A[i, j]| and called the max-norm, and the entry-wise 1-norm, denoted as ∥A∥ sum := ∥A∥ 1 = i,j |A[i, j]|. The entry-wise 1-norm is dual to the maxnorm. For vectors, the max-norm is equivalent to the infinity norm, denoted as ∥a∥ ∞ , while we denote the ℓ 1 norm as ∥a∥ 1 . We further denote the Schatten p-norm of A as |||A||| p := ( r i=1 σ p i ) 1/p , where σ 1 , σ 2 , . . . , σ r are the non-zero singular values of A. Let r = rank(A), then special cases of Schatten p-norm include: nuclear norm |||A||| 1 = r i=1 σ i , Frobenius norm |||A||| 2 = r i=1 σ 2 i , and spectral norm |||A||| ∞ = σ 1 . To simplify the discussions, we sometimes write ∥A∥ (dropping subscripts) to refer to any entry-wise or Schatten p-norm with p ≥ 1. We denote by ∥A∥ * the dual-norm with respect to the standard matrix inner product ⟨A, B⟩ = tr(A ⊤ B). We denote the gradient and its value at iteration t as ∇ := ∇L(W ) and ∇ t := ∇L(W t ) respectively.
Let S : R k → △ k-1 the softmax map of k-dimensional vectors to the probability simplex △ k-1 such that for any a ∈ R k , it holds that S(a) = exp(a[c]) c∈[k] exp(a[c]) k c=1 ∈ △ k-1 . Let S c (v) denote the c-th entry of S(v). Let S ′ (a) = diag(S(a)) -S(a)S(a) ⊤ denote the softmax gradient, with diag(•) a diagonal matrix. Finally, let {e c } k c=1 be the standard basis vectors of R k , and indicator δ ij be such that δ ij = 1 if and only if i = j. For any integer k, [k] denotes {1, . . . , k}.
Setup Consider a multiclass classification problem with training data h 1 , . . . , h n and labels y 1 , . . . , y n . Each datapoint h i ∈ R d is a vector in a d-dimensional embedding space (denote data matrix H = [h 1 , . . . , h n ] ⊤ ∈ R n×d ), and each label y i ∈ [k] represents one of k classes. We assume each class contains at least one datapoint. The classifier f W : R d → R is a linear model with weight matrix W ∈ R k×d . The model outputs logits ℓ i = f W (h i ) = W h i for i ∈ [n], which are passed through the softmax map to produce class probabilities p(c|h i ) = S c (ℓ i ).
We train using empirical risk minimization (ERM): L ERM (W ) := -1 n i∈[n] ℓ (W h i ; y i ) , where the loss function ℓ takes as input the logits of a datapoint and its label. The predominant choice in classification is the CE loss
L(W ) : = - 1 n i∈[n] log S yi (W h i ) .(1)
We focus our discussions on the CE loss due to its ubiquity in practice. However, our results hold for other multiclass losses such as the exponential [41] and the PairLogLoss [72] (see App. F). Define the maximum margin of the dataset w.r.t. any entry-wise or Schatten p-norm ∥ • ∥ as
γ := max ∥W ∥≤1 min i∈[n], c̸ =yi (e yi -e c ) ⊤ W h i .(2)
Optimization Methods We study iterative algorithms that update the weight matrix by W t+1 = W t -η t ∆ t .
For the NSD family [10], the update directionfoot_0 w.r.t. the norm ∥•∥ is ∆ t := arg max ∥∆∥≤1 ⟨∇ t , ∆⟩ .
(3) Note that this reduces to SignGD, Coordinate Descent (e.g., Nutini et al. [44]), or NGD when the max-norm (i.e. ∥•∥ ∞ ), the entry-wise 1-norm (i.e. ∥•∥ sum ), or the Frobenius Euclidean-norm (i.e. ∥•∥ 2 ) is used, respectively. Concretely, the update directions for SignGD and NGD are:
SignGD: ∆ t = sign(∇ t ), and NGD: ∆ t = ∇ t /∥∇ t ∥ 2 , where the sign(•) and division •
• operations are applied entry-wise. In the special case of spectral norm (i.e. |||•||| ∞ ), this becomes the Spectral-GD, for which ∆ t = U t V T t , where U t and V t are the left/right singular matrices of ∇ t respectively (i.e., ∇ t = U t Σ t V T t with singular values in Σ t > 0 arranged in non-increasing order). Finally, note that the Schatten 2-norm case reduces to NGD (as
|||•||| 2 = ∥•∥ 2 ).
We also consider the NMD family with the following update direction w.r.t. the norm ∥•∥ ∆ t := arg max ∥∆∥≤1 ⟨M t , ∆⟩ , (4) where the momentum M t is computed as the EMA of the gradient given by M t = β 1 M t-1 + (1 -β 1 )∇ t .
(5) This form of momentum is also known as the heavy-ball or the SGDM-style momentum [47,19,38]. Thus, an NMD algorithm chooses the update direction (among all feasible directions in the unit ∥•∥-ball) that best aligns with the momentum instead of the gradient direction (as chosen by an NSD algorithm). Similar to above, when the max-norm and the Frobenius-norm are used, the resulting Signum and NMD-GD update directions are:
Signum: ∆ t = sign(M t ), and NMD-MD:
∆ t = M t ∥M t ∥ 2 .
When spectral norm is used in (4), this becomes Muonfoot_1 for which the SVD is on M t (i.e. M t = Ũt Σt Ṽ T t ) and the update direction is ∆ t = Ũt Ṽ T t . Note that Muon reduces to Spectral-GD when the momentum parameter β 1 is set to 0. Similar reductions also hold for Signum (to SignGD) and NMD-GD (to NGD) as well.
Assumptions Establishing the implicit bias of the above mentioned gradient-based optimization algorithms requires the following assumptions. First, we assume data are linearly separable, ensuring the margin γ is strictly positive, an assumption routinely used in previous works [53,48,22,43,75].
Assumption 1. There exists W ∈ R k×d such that min c̸ =yi (e yi -e c ) T W h i > 0 for all i ∈ [n].
In this work, we consider learning rate schedule η t = Θ( 1 t a ), where a ∈ (0, 1]. Such schedules have been studied in the convergence and implicit bias of various optimization algorithms (e.g., Bottou et al. [9], Nacson et al. [43], and Sun et al. [55]) including Adam [77]. Assumption 2. The learning rate schedule {η t } is decreasing with respect to t and satisfies the following conditions: lim t→∞ η t = 0 and ∞ t=0 η t = ∞. Assumption 3 can be satisfied by the above learning rate for a sufficiently large t as shown in Zhang et al. [77,Lemma C.1]. It is used in our analysis of NMD and Adam. Assumption 3. The learning rate schedule satisfies the following: let β ∈ (0, 1) and c 1 > 0 be two constants, there exist time t 0 ∈ N + and constant c 2 = c 2 (c 1 , β) > 0 such that t s=0 β s (e c1 s τ =1 ηs-τ -1) ≤ c 2 η t for all t ≥ t 0 .
Finally, we assume that the 1-norm of the data is bounded. Similar assumptions were used in Ji and Telgarsky
this section cite: ['b40', 'b71', 'b9', 'b43', 'b46', 'b18', 'b37', 'b52', 'b47', 'b21', 'b42', 'b74', 'b8', 'b42', 'b54', 'b76', 'b76']

Section: A Unified Framework with a Proxy Function
Analyzing margin convergence begins with studying loss convergence through second-order Taylor expansion of the CE loss (recall that S ′ (v) = diag(v) -vv ⊤ ):
L(W + ∆) = L(W ) + ⟨∇L(W ), ∆⟩ + 1 2n i∈[n] h ⊤ i ∆ ⊤ S ′ (W h i )∆h i + o(∥∆∥ 3 F ), (6
)
To bound the loss at W t+1 = W t -η t ∆ t , we must bound both the first-order and second-order terms in (6). For NSD updates in Eq. ( 3), the first term evaluates to -η t ∥∇L(W )∥ * (recall that ∥ • ∥ * is the dual norm). This leads to two key tasks: (1) Lower-bounding the dual gradient norm; (2) Upper-bounding the second-order term.
For the proof to proceed, these bounds should satisfy two desiderata: (1) They are expressible as the same function of W , call it G(W ), up to constants. (2) The function G(W ) is a good proxy for the loss for small values of the latter. The former helps with combining the terms, while the latter helps with demonstrating descent. Next, we obtain these key bounds for the CE loss by determining the appropriate proxy G(W ).
Besides the need for a proxy G(W ), we use the following facts about the sum-norm dominating any entry-wise/Schatten p-norm. Concretely, for any matrix A and any p ≥ 1:
∥A∥ max ≤ |||A||| p ≤ ∥A∥ sum , and ∥A∥ max ≤ ∥A∥ p ≤ ∥A∥ sum .(7)
These relationships (proved in Lemma 11 in App. C) are crucial for unifying the analysis of NSD and NMD algorithms w.r.t. either the entry-wise or the Schatten norms (details below).
this section cite: ['b5']

Section: Construction of G(W )
Before showing our construction for the CE loss, it is insightful to discuss how previous works do this in the binary case with labels y b,i ∈ {±1}, classifier vector w ∈ R d and binary margin γ b := max ∥w∥≤1 min i∈[n] y b,i w ⊤ h i . For exponential loss, Gunasekar et al. [22] showed that ∥∇L(w)∥ ≥ γ b L(w). For logistic loss ℓ(t) = log(1 + exp(-t)), Zhang et al. [77] proved
∥∇L(w)∥ 1 ≥ γ b G(w), where G(w) = 1 n n i=1 |ℓ ′ (y b,i w ⊤ h i )|
and ℓ ′ is the first-order derivative. In both cases, one can take the common form
G b (w) = 1 n n i=1 |ℓ ′ (y b,i w ⊤ h i )|.
The proof relies on showing γ ≤ min r∈△ n-1 ∥H T r∥ via Fenchel Duality [59,22] and appropriately choosing r.
In the multiclass setting, where the loss function is vector-valued, it is unclear how to extend the binary proof or definition of G(W ). To this end, we realize that the key is in the proper manipulation of the gradient inner product ⟨A, -∇L(W )⟩ (for arbitrary matrix A ∈ R k×d ). The CE gradient evaluates to
∇L(W ) = 1 n n i=1 (e yi -S(W h i ))h ⊤
i and using the fact that S(W h i ) ∈ △ k-1 , it turns out that we can express (details in Lemma 9
): ⟨A, -∇L(W )⟩ = 1 n i∈[n] c̸ =yi S c (W h i )(e yi -e c ) ⊤ Ah i . This motivates defining G(W ) as: G(W ) := 1 n i∈[n] (1 -S yi (W h i )) .(8)
The lemma below, following from the inner-product calculation above and our definition of G(W ), confirms this is the right choice. For convenience, denote
s ic := S c (W h i ), for i ∈ [n], c ∈ [k].
Lemma 1 (Lower bounding the gradient dual-norm). For any W ∈ R k×d and any entry-wise or Schatten p-norm ∥•∥ with p ≥ 1, it holds that ∥∇L(W )∥ * ≥ γ • G(W ), where ∥•∥ * is the dual-norm.
The lemma completes the first task: lower bounding the gradient's dual norm. Importantly, the factor in front of G(W ) is the margin γ w.r.t. the norm ∥•∥, which is crucial in the forthcoming analysis.
G(W ) and second-order term We now show how to bound the second-order term in (6). For this, we establish the following essential lemma whose proof relies on the relationships in (7).
this section cite: ['b21', 'b76', 'b58', 'b21', 'b5']

Section: Lemma 2.
For any entry-wise or Schatten p-norm ∥•∥ with p ≥ 1, any s ∈ ∆ k-1 in the k-dimensional simplex, any index c ∈ [k], and v ∈ R k , it holds that
v ⊤ diag(s) -ss ⊤ v ≤ 4(1 -s c )∥vv T ∥.
Proof. Let S := diag(s) -ss ⊤ and q ≥ 1 such that 1/p + 1/q = 1. By norm duality, it holds that
v ⊤ Sv = tr Svv ⊤ ≤ ∥S∥ q ∥vv ⊤ ∥ ≤ ∥S∥ sum ∥vv ⊤ ∥,
where ∥•∥ q is the dual of ∥•∥ and the second inequality is by (7). Direct calculation yields ∥S∥ sum = 2 c∈[k] s c (1 -s c ). The advertised bound then follows by noting the following c∈[k] s c (1 -s c ) ≤ 2(1 -s c ′ ) for any c ′ ∈ [k] (verified in Lemma 13 in App. C). Next, we apply the above lemma with v ← ∆h i and c ← y i , and further use the inequalities: ∥vv T ∥ p = ∥v∥ 2 p ≤ ∥∆∥ 2 p ∥h∥ 2 q for entry-wise norms and vv ⊤ p = ∥v∥ 2 2 ≤ |||∆||| 2 ∞ ∥h∥ 2 2 ≤ |||∆||| 2 p ∥h∥ 2 2 for Schatten norms. Together with Ass. 4, this upper bounds the second-order term in the CE loss expansion in terms of the proxy function:
2B 2 ∥∆∥ 2 • 1 n i∈[n] (1 -S yi (W h i )) .
Properties of G(W ) We now show that G(W ) meets the second desiderata: being a good proxy for the loss L(W ). This is rooted in the elementary relationships between G(W ) and L(W ), which are used in the various parts of the proof. Below, we summarize these key relationships.
Lemma 3 (Properties of G(W ) and L(W )). Let W ∈ R k×d . The followings hold: (i) Under Ass.
4, 2B • G(W ) ≥ ∥∇L(W )∥ * ; (ii) 1 ≥ G(W ) L(W ) ≥ 1 -nL(W ) 2 ; (iii) If W satisfies L(W ) ≤ log 2 n or G(W ) ≤ 1 2n , then L(W ) ≤ 2G(W ).
Lemma 3 (i) extends Lemma 1 by establishing a sandwich relationship between G(W ) and the gradient's dual norm. The lemma's statements (ii) and (iii) show that G(W ) can substitute for the loss -it lower bounds L(W ) and serves as an upper bound when either L(W ) or G(W ) is sufficiently small. Specifically, the ratio G(W ) L(W ) converges to 1 as the loss decreases, with the convergence rate depending on the rate of loss decrease. The key property (ii) may seem algebraically complex, but it turns out (details in Lemma 18 in App. C) that both sides of the sandwich relationship follow from the elementary fact that ∀x > 0 : 1 -x ≤ e -x ≤ 1 -x + x 2 /2.
this section cite: ['b6']

Section: Implicit Bias of Normalized Steepest Descent
We now leverage our construction of G(W ) to show that the margin of NSD's iterates converges to the data margin defined w.r.t. the same entry-wise or Schatten p-norm that is used to define the algorithm (refer to eqns. (2) and (3) for the definitions of margin and NSD). We only highlight the key steps in the proof and defer details to App. D.
this section cite: []

Section: NSD Descent
We start by showing a descent property. By applying Lemmas 1 and 2 to lower and upper bound the first and second order terms in Eq. ( 6) yields
L(W t+1 ) ≤ L(W t ) -γη t G(W t ) + 2η 2 t B 2 G(W t ) sup ζ∈[0,1] G(W t -ζη t ∆ t ) G(W t ) .
Algebraic manipulations of the definition of G(W ) and the relationships in (7) allow us to bound the ratio in the right hand side.
Lemma 4 (Ratio of G(W )). For any ψ ∈ [0, 1], we have the following:
G(W -ψη∆) G(W )
≤ e 2Bηψ∥∆∥ max ≤ e 2Bηψ∥∆∥ (note that the second inequality is by (7)).
From this and ∥∆ t ∥ ≤ 1 for NSD, we obtain
L(W t+1 ) ≤ L(W t ) -γη t (1 -α s1 η t )G(W t ),(9)
where α s1 = 2B 2 e 2Bη0 /γ. Given a decay learning rate of the form η t = Θ( 1 t a ), we can conclude that the loss starts to monotonically decrease after some time.
this section cite: []

Section: NSD Unnormalized Margin
We now use the descent property in (9) to lower bound the unnormalized margin. An intermediate result towards this is recognizing that sufficiently small loss L(W ) ≤ log 2 n guarantees W separates the data (Lemma 19 in App. C). The descent property ensures that NSD iterates will eventually achieve this loss threshold, thereby guaranteeing separability. The main result of this section, shows that eventually the iterates achieve separability with a substantial (unnormalized) margin.
this section cite: ['b8']

Section: Lemma 5 (NSD Unnormalized Margin).
Assume there exists t such that L(W t ) ≤ log 2 n , ∀t > t. Then, it holds that for all t ≥ t (α
s2 = 2B 2 e 2Bη0 ) min i∈[n],c̸ =yi (e yi -e c ) T W t h i ≥ γ t-1 s= t η s G(W s ) L(W s ) -α s2 t-1 s= t η 2 s .(10)
NSD Margin Convergence Proceeding from Eq. ( 10) requires showing the convergence of the ratio G(W ) L(W ) . The two key ingredients are given in Lemma 3 (ii) and (iii). Lemma 3 (ii) suggests that it is sufficient to study the convergence of L(W ), which is captured in (9). However, to obtain an explicit rate via (9), we need to rewrite G(W t ) in terms of L(W t ). This is where Lemma 3 (iii) helps. Putting them together, we arrive at the following theorem (see Thm. 3 and Cor. 1 for details). Theorem 1. Suppose that Ass. 1, 2, and 4 hold. Set learning rate η t = Θ( 1 t 1/2 ). The following holds for the margin gap of NSD's iterates
γ - min i∈[n],c̸ =yi (e yi -e c ) T W t h i ∥W t ∥ ≤ O( log t + n t 1/2 ).
Remark 1. For margin convergence rates of NSD, Nacson et al. [43] showed a rate of O( log t t 1/2 ) in the binary setting, limited to the entry-wise p-norms and the exponential loss. Compared to this, our results hold for the more practical setting of multiclass data and CE loss. To the best of our knowledge, this is the first non-asymptotic result on the implicit bias of spectral-GD for linear multiclass separable data, and it holds for other p-norms as well. Upon completion of this work, we became aware of an update on the arXiv version of Tsilivis et al. [63], which includes an extension of their previous results to steepest descent w.r.t. the spectral norm. In comparison to ours, their gradient-flow analysis applies to homogeneous neural networks with the restriction of infinitesimal step-sizes. Moreover, it does not include normalization nor momentum (like Muon, which we analyze), and the convergence is (asymptotic) to a KKT point of a spectral-norm margin maximization problem.
this section cite: ['b8', 'b42', 'b62']

Section: Implicit Bias of Normalized Momentum Steepest Descent
In this section, we study the implicit bias of NMD algorithms (proof details in App E). Similar to Sec. , its updates are defined w.r.t. either the entry-wise or the Schatten norm ∥•∥. The analysis relies on the relationships in (7) and we show the same proxy function G(W ) naturally appears. Given that the NMD updates satisfy ∥∆∥ ≤ 1, the second-order term in (6) is bounded in the same way as NSD. The main difference is in bounding the first-order term as shown by the following lemma.
this section cite: []

Section: Lemma 6.
Let Ω t := M t -∇ t . It holds for all t ≥ 0 that
⟨∇ t , W t+1 -W t ⟩ ≤ 2η∥Ω t ∥ * -ηγG(W t ) .
Given the relationships in (7) hold for any p ≥ 1, we can bound the dual norm of ∥Ω t ∥ * via its sum norm (i.e. ∥Ω t ∥ * ≤ ∥Ω∥ sum ). Given the goal is to bound all the terms in the Taylor expansion (6) via the proxy function G(W t ), an natural next step is to bound ∥Ω t ∥ sum using the same proxy function. To do this, we decompose the proxy function per-class-wise, and apply the per-class proxy functions to bound the entries of Ω t associated with their corresponding classes. Concretely, we write the function G(W ) in two equivalent ways: G(W ) = c∈[k]  Next, we bound the entries in each row of Ω t (thus belonging to the same class) via the corresponding proxy functions G c (W t ) and Q c (W t ) to arrive at the following lemma. Its proof utilizes the nice properties of softmax map given in Lemma 15 in App. B. Lemma 7. Suppose that Ass. 1, 2, 3, and 4 hold. Let c ∈ [k] and j ∈ [d]. There exists time t 0 such that for all t ≥ t 0 and for α M := B(1 -β 1 )c 2 :
|M t [c, j] -(1 -β t+1 1 )∇L(W t )[c, j]| ≤ α M η t G c (W t ) + Q c (W t ) .
Given the result in Lemma 7, we can show that |Ω t [c, j]| ≤ β t+1 1 |∇L(W t )[c, j]| + α M η t T c (W t ), where T c (W t ) is defined to be T c (W t ) := G(W t ) + Q(W t ). Then, we sum over indices c ∈ [k] and j ∈ [d] and apply ∥∇∥ sum ≤ 2B • G(W ) (from Lemma 3 (i)) to obtain:
Lemma 8. It holds for all t ≥ 0 that ∥Ω t ∥ sum ≤ 2Bβ t/2 1 G(W t ) + 2α M dη t G(W t ).
This completes the bound on the first-order term for NMD algorithms via the proxy G(W ). The rest proof follows similar steps as NSD. We note that without the above per-class decomposition, an extra k-factor would appear in the second term of the bound on ∥Ω t ∥ sum (and thus also show up in the final rate). We state the main theorem for NMD algorithms. Theorem 2. Under the setting of Lem. 7, the margin gap of NMD with η t = Θ( 1
t 1/2 ) is O( d log t+dn t 1/2
). Remark 2. Wang et al. [69] studied implicit bias of un-normalized GD with momentum, and showed its iterates converge asymptotically to the max 2-norm margin solution. In contrast, our rates are non-asymptotic and cover a much wider family of algorithms converging to non-Euclidean geometric margins (w.r.t. entry-wise/Schatten norms). Note the convergence rate of NMD matches that of NSD (Thm. 1) up to a factor of d. It could be interesting to remove this dependence in a future work.
Implicit Bias of Adam Finally, observing that Adam [33] (without the stability constant, i.e., eqns (33a), (33b), and (33c) in App. G) shares the same form of momentum as NMD and the (entry-wise) updates are bounded by some constant as shown in Zhang et al. [80] and Xie and Li [76]. Thus, our analysis extends to Adam. Concretely, a similar proof strategy can be adapted once a bound on the second gradient moment via the proxy function is established (Lemma 35). In App. G, we prove a O( d log(t)+nd
t 1/3
) max-norm margin convergence rate for Adam (details in Thm. 5 and Cor. 2).
this section cite: ['b68', 'b32', 'b79', 'b75']

Section: Experiments

this section cite: []

Section: Synthetic Experiments
We generate snythetic multiclass separable data as follows: k = 10 class centers are sampled from a standard normal distribution; within each class, data is sampled from normal distribution N (0, σ 2 I), σ = 0.1. We set d = 25, sample 50 data points for each class, and ensure that margin is positive (thus data is separable). We run different algorithms to minimize CE loss using η t = η0 t a (η 0 = 0.1 for SignGD and NGD; η 0 = 0.05 for Spectral-GD and Muon), where (based on our theorems) a is set to 1 /2. We apply truncated SVD on the gradient and momentum for Spectral-GD and Muon respectively. Data margins w.r.t. different norms are found via CVXPY [17]. We denote max-margin classifiers defined w.r.t. the 2-norm, the max-norm, and the spectral-norm as V 2 , V ∞ , and V spec , respectively. Based on the margin-gap results in Figure 1, we observe that SignGD, NGD, and Spectral-GD favor max-norm, 2-norm, and spectral-norm margin respectively. Besides this, the behavior of Muon is very similar to that of Spectral-GD (in agreement with our theories). Figure 2 further confirms that the iterates of these algorithms correlate well with the corresponding max margin separators. Experiments on Signum, NMD-GD, and Adam are provided in App. A and App. G. Two-layer Neural Network We extend the experiments to the non-linear classification setting using the cross-entropy loss. We sample 100 data points from each of the 10 classes of the MNIST dataset [35]. The model is a two-layer neural network with the hidden dimension being 100 (the first and second layer weights are denoted as V and W respectively). The training is done in two ways: (a) Train the first-layer weight with the second-layer weight fixed and (b) Train both the first and second-layer weights. For options (a) and (b), the respective spectral-norm margins of the overall network are defined as 10 0 10 1 10 2 10 3 10 4 Iterations 0.70 0.75 0.80 0.85 0.90 0.95 Correlation Correlation with V Correlation with V2 Correlation with Vspec (a) SignGD 10 0 10 1 10 2 10 3 10 4 0.75 0.80 0.85 0.90 0.95 (b) NGD 10 0 10 1 10 2 10 3 10 4 0.75 0.80 0.85 0.90 0.95 (c) Spectral-GD 10 0 10 1 10 2 10 3 10 4 0.75 0.80 0.85 0.90 0.95 (d) Muon Figure 2: (a) Correlations between the iterates of SignGD (W t ) and max margin separators V ∞ , V 2 , and V spec against iterations (correlation defined as ⟨W ,V ⟩ ∥W ∥2∥V ∥2 ). (b, c, and d) Same as (a) with SignGD replaced by NGD, Spectral-GD, and Muon, respectively. SignGD and NGD correlate well with V ∞ and V 2 , respectively, while Spectral-GD and Muon correlate well with V spec . 10 0 10 1 10 2 10 3 10 4 Iterations 1.0 0.8 0.6 0.4 0.2 0.0 0.2 Spectral-Norm Margin SignGD NGD Spectral GD (a) Single Layer(γ V a ) 10 0 10 1 10 2 10 3 10 4 Iterations 1.0 0.8 0.6 0.4 0.2 0.0 0.2 Spectral-Norm Margin Signum NMD GD Muon (b) Single Layer(γ V a ) 10 0 10 1 10 2 10 3 10 4 Iterations 4 2 0 2 4 6 Spectral-Norm Margin SignGD NGD Spectral GD (c) Joint Training(γ V ,W b ) 10 0 10 1 10 2 10 3 10 4 Iterations 2 0 2 4 6 Spectral-Norm Margin Signum NMD GD Muon (d) Joint Training(γ V ,W b ) Figure 3: (a) Spectral-norm margin γ Vt a as a function of t for SignGD, NGD, and Spectral-GD. (b) Same as (a) with algorithms replaced by Signum, NMD-GD, and Muon, respectively. (c) Spectralnorm margin γ Vt,Wt b as a function of t for SignGD, NGD, and Spectral-GD. (d) Same as (c) with algorithms replaced by Signum, NMD-GD, and Muon, respectively.
(a) γ V a := min i∈[n],c̸ =yi (e yi -e c ) T W σ(V h i ) |||V ||| ∞ , (b) γ V ,W b := min i∈[n],c̸ =yi (e yi -e c ) T W σ(V h i ) max{|||V ||| ∞ , |||W ||| ∞ } ,
where σ(•) is the sigmoid function. In Figure 3, we track the quantities γ Vt a (Figure 3a and 3b) and γ Vt,Wt a (Figure 3c and 3d) as a function of the iteration counter t. For both definitions, we observe that the spectral-norm margin of the iterates of Spectral-GD and Muon grow faster than other algorithms. Hence, the observations in the linear settings can also hold in the non-linear settings.
this section cite: ['b16', 'b34']

Section: Related Works
Starting with GD, the foundational result by Soudry et al. [53] showed that gradient descent optimization of logistic loss on linearly separable data converges in direction to the L 2 max-margin classifier at a rate O(1/ log(t)). Contemporaneous work by Ji and Telgarsky [26] generalized this by relaxing the data separability requirement. Ji et al. [29] later connected these findings to earlier work on regularization paths of logistic loss minimization [49], which enabled extensions to other loss functions (e.g., those with polynomial tail decay). More recently, Wu et al. [75] extends these results to the large step size regime with the same O(1/ log(t)) rate. The relatively slow convergence rate to the max-margin classifier motivated investigation into adaptive step-sizes. Nacson et al. [43] showed that NGD with decaying step-size η t = 1/ √ t achieves L 2 -margin convergence at rate O(1/ √ t). This rate was improved to O(1/t) by Ji and Telgarsky [28] using constant step-sizes, and further to O(1/t 2 ) through a specific momentum formulation [30]. Besides linear classifications, implicit bias of GD has been studied for least squares [21,22,5], homogeneous [40,27,74] and non-homogeneous neural networks [12], as well as matrix factorization [21]; see Vardi [64] for a survey.
All the above mentioned works focus almost exclusively on binary classification. The noticeable gap in analysis of multiclass classification in most existing literature is highlighted by Thrampoulidis et al. [61], and more recently emphasized by Ravi et al. [48], who extended the implicit bias result of Soudry et al. [53] to multiclass classification for losses with exponential tails, including CE, multiclass exponential, and PairLogLoss. Their approach leverages a framework of Wang and Scott [73] that allows multiclass losses and separability conditions to be written in margin-based forms similar to binary cases. However, these works only focus on GD with the L 2 -geometry. In this work, we consider a wide range of algorithms with different geometries for multiclass classification.
Beyond GD, Gunasekar et al. [22] and Nacson et al. [43] showed that steepest descent w.r.t. entrywise p-norms yields updates that in the limit maximize the margin w.r.t the same norm. Sun et al. [54,55] showed that the iterates of mirror descent with the potential function chosen as the p-th power of the p-norm converge to the classifier that maximizes the margin w.r.t. the p-norm. In both cases, the convergence rate is slow at O(1/ log(t)). Wang et al. [70] further improved the rates for both steepest descent and mirror descent when p ∈ (1,2]. Note that all these results apply only to the exponential loss. More recently, Tsilivis et al. [63] showed that the iterates of steepest descent algorithms converge to a KKT point of a generalized margin maximization problem in homogeneous neural networks. Moreover, the implicit bias of Adam (with or without the stability constant) has been studied in both linear and non-linear settings. Wang et al. [68] demonstrated the normalized iterates of Adam (with non-negligible stability constant) converge to a KKT point of a L 2 -margin maximization problem for homogeneous neural networks. Zhang et al. [77] studied the implicit bias of Adam without the stability constant on (linearly) binary separable data. They showed that unlike GD, the Adam's iterates converge to a solution that maximizes the margin w.r.t the L ∞ -norm. The study of excluding the stability constant is also the focus of another recent work on the implicit bias of AdamW [76], where the authors again establish that convergence aligns with the L ∞ geometry.
this section cite: ['b52', 'b25', 'b28', 'b48', 'b74', 'b42', 'b27', 'b29', 'b20', 'b21', 'b4', 'b39', 'b26', 'b73', 'b11', 'b20', 'b63', 'b60', 'b47', 'b52', 'b72', 'b21', 'b42', 'b53', 'b54', 'b69', 'b0', 'b1', 'b62', 'b67', 'b76', 'b75']

Section: Conclusion
We have characterized the margin convergence rates of Spectral-GD and Muon for multiclass linear separable data. Given they are special cases of NSD and NMD w.r.t the spectral norm, the analysis is done on a wider scale by studying NSD/NMD w.r.t any entry-wise or Schatten p-norms. Thus, the rates also hold for optimizers of other geometries, such as the sign-descent (max-norm) or gradientdescent (2-norm) family. We further extend the analysis to Adam using the same framework. Future directions include removing the factor-d from the bound of NMD, obtaining a tighter convergence rate for Adam, and studying other related algorithms such as Shampoo that involves non-diagonal preconditioners. It is also important to extend our results to (multiclass) non-separable settings [60] and nonlinear models such as diagonal neural nets [45], self-attention mechanisms [58,4,57,66,32] and homogeneous neural nets [40,63,12], helping further bridge the gap to deep learning practices. Finally, from a complementary statistical perspective, future work could seek identifying specific scenarios where margin maximization with respect to norms other than Frobenius leads to better generalization (extending a long line of prior works, e.g., [50,16,34,42,71,18,56,65,2,62]).
this section cite: ['b59', 'b44', 'b57', 'b3', 'b56', 'b65', 'b31', 'b39', 'b62', 'b11', 'b49', 'b15', 'b33', 'b41', 'b70', 'b17', 'b55', 'b64', 'b1', 'b61']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Regularized linear regression for binary classification Year: (2024)
Ref_id:b2 Title: Asgo: Adaptive structured gradient optimization Year: (2025)
Ref_id:b3 Title: Max-margin token selection in attention mechanism Year: (2023)
Ref_id:b4 Title: Stochastic mirror descent on overparameterized nonlinear models Year: (2021)
Ref_id:b5 Title: Reconciling modern machinelearning practice and the classical bias-variance trade-off Year: (2019)
Ref_id:b6 Title: Old optimizer, new norm: An anthology Year: (2024)
Ref_id:b7 Title: Compressed optimisation for non-convex problems Year: (2018)
Ref_id:b8 Title: Optimization methods for large-scale machine learning Year: (2018)
Ref_id:b9 Title: Convex optimization Year: (2004)
Ref_id:b10 Title: Language models are few-shot learners Year: (2020)
Ref_id:b11 Title: Implicit bias of gradient descent for non-homogeneous deep networks Year: (2025)
Ref_id:b12 Title: Preconditioned spectral descent for deep learning Year: (2015)
Ref_id:b13 Title: Momentum improves normalized sgd Year: (2020)
Ref_id:b14 Title: Benchmarking neural network training algorithms Year: (2023)
Ref_id:b15 Title: A model of double descent for high-dimensional binary linear classification. Information and Inference: A Year: (2021-04)
Ref_id:b16 Title: Cvxpy: A python-embedded modeling language for convex optimization Year: (2016)
Ref_id:b17 Title: Fast rates for noisy interpolation require rethinking the effect of inductive bias Year: (2022)
Ref_id:b18 Title: Global convergence of the heavy-ball method for convex optimization Year: (2015)
Ref_id:b19 Title: A kronecker-factored approximate fisher matrix for convolution layers Year: (2016)
Ref_id:b20 Title: Implicit regularization in matrix factorization Year: (2017)
Ref_id:b21 Title: Characterizing implicit bias in terms of optimization geometry Year: (2018)
Ref_id:b22 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b23 Title: Shampoo: Preconditioned stochastic tensor optimization Year: (2018)
Ref_id:b24 Title: Beyond convexity: Stochastic quasi-convex optimization Year: (2015)
Ref_id:b25 Title: The implicit bias of gradient descent on nonseparable data Year: (2019)
Ref_id:b26 Title: Directional convergence and alignment in deep learning Year: (2020)
Ref_id:b27 Title: Characterizing the implicit bias via a primal-dual analysis Year: (2021)
Ref_id:b28 Title: Gradient descent follows the regularization path for general losses Year: (2020)
Ref_id:b29 Title: Fast margin maximization via dual acceleration Year: (2021)
Ref_id:b30 Title: Muon: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b31 Title: Optimizing attention with mirror descent: Generalized max-margin token selection Year: (2024)
Ref_id:b32 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b33 Title: Uniform convergence of interpolators: Gaussian width, norm bounds and benign overfitting Year: (2021)
Ref_id:b34 Title: Gradient-based learning applied to document recognition Year: (2002)
Ref_id:b35 Title: A note on the convergence of muon and further Year: (2025)
Ref_id:b36 Title: Muon is scalable for llm training Year: (2025)
Ref_id:b37 Title: An improved analysis of stochastic gradient descent with momentum Year: (2020)
Ref_id:b38 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b39 Title: Gradient descent maximizes the margin of homogeneous neural networks Year: (2019)
Ref_id:b40 Title: A theory of multiclass boosting Year: (2010)
Ref_id:b41 Title: Classification vs regression in overparameterized regimes: Does the loss function matter? arXiv preprint Year: (2020)
Ref_id:b42 Title: Convergence of gradient descent on separable data Year: (2019)
Ref_id:b43 Title: Coordinate descent converges faster with the gauss-southwell rule than random selection Year: (2015)
Ref_id:b44 Title: Implicit bias of sgd for diagonal linear networks: a provable benefit of stochasticity Year: (2021)
Ref_id:b45 Title: Training deep learning models with norm-constrained lmos Year: (2025)
Ref_id:b46 Title: Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics Year: (1964)
Ref_id:b47 Title: The implicit bias of gradient descent on separable multiclass data Year: (2024)
Ref_id:b48 Title: Margin maximizing loss functions Year: (2003)
Ref_id:b49 Title: The impact of regularization on highdimensional logistic regression Year: (2019)
Ref_id:b50 Title: Adafactor: Adaptive learning rates with sublinear memory cost Year: (2018)
Ref_id:b51 Title: A distributed data-parallel pytorch implementation of the distributed shampoo optimizer for training neural networks at-scale Year: (2023)
Ref_id:b52 Title: The implicit bias of gradient descent on separable data Year: (2018)
Ref_id:b53 Title: Mirror descent maximizes generalized margin and can be implemented efficiently Year: (2022)
Ref_id:b54 Title: A unified approach to controlling implicit regularization via mirror descent Year: (2023)
Ref_id:b55 Title: Generalization and stability of interpolating neural networks with minimal width Year: (2023)
Ref_id:b56 Title: Transformers as support vector machines Year: (2023)
Ref_id:b57 Title: Max-margin token selection in attention mechanism Year: (2023)
Ref_id:b58 Title: Margins, shrinkage, and boosting Year: (2013)
Ref_id:b59 Title: Implicit optimization bias of next-token prediction in linear models Year: (2024)
Ref_id:b60 Title: Imbalance trouble: Revisiting neural-collapse geometry Year: (2022)
Ref_id:b61 Title: Benign overfitting and the geometry of the ridge regression solution in binary classification Year: (2025)
Ref_id:b62 Title: Flavors of margin: Implicit bias of steepest descent in homogeneous neural networks Year: (2024)
Ref_id:b63 Title: On the implicit bias in deep-learning algorithms Year: (2023)
Ref_id:b64 Title: Benefits of stochastic mirror descent in high-dimensional binary classification Year: (2024)
Ref_id:b65 Title: Implicit bias and fast convergence rates for self-attention Year: (2024)
Ref_id:b66 Title: Soap: Improving and stabilizing shampoo using adam Year: (2024)
Ref_id:b67 Title: The implicit bias for adaptive optimization algorithms on homogeneous neural networks Year: (2021)
Ref_id:b68 Title: Does momentum change the implicit regularization on separable data? Year: (2022)
Ref_id:b69 Title: Faster margin maximization rates for generic optimization methods Year: (2023)
Ref_id:b70 Title: Benign overfitting in multiclass classification: All roads lead to interpolation Year: (2021)
Ref_id:b71 Title: Rank4class: a ranking formulation for multiclass classification Year: (2021)
Ref_id:b72 Title: Unified binary and multiclass margin-based classification Year: (2024)
Ref_id:b73 Title: Large stepsize gradient descent for logistic loss Year: (2024)
Ref_id:b74 Title: Implicit bias of gradient descent for logistic regression at the edge of stability Year: (2024)
Ref_id:b75 Title: Implicit bias of adamw: L-infinity norm constrained optimization Year: (2024)
Ref_id:b76 Title: The implicit bias of adam on separable data Year: (2024)
Ref_id:b77 Title: Understanding deep learning requires rethinking generalization Year: (2017)
Ref_id:b78 Title: Why transformers need adam: A hessian perspective Year: (2024)
Ref_id:b79 Title: Use fewer learning rates to gain more Year: (2024)
Ref_id:b80 Title: On the characterization of the extremal points of the unit sphere of matrices Year: (1988)
