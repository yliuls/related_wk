Title: Efficient Knowledge Transfer in Federated Recommendation for Joint Venture Ecosystem
Abstract: The current Federated Recommendation System (FedRS) focuses on personalized recommendation services and assumes clients are personalized IoT devices (e.g., mobile phones). In this paper, we deeply dive into new but practical FedRS applications within the joint venture ecosystem. Subsidiaries engage as participants with their users and items. However, in such a situation, merely exchanging item embedding is insufficient, as user bases always exhibit both overlaps and exclusive segments, demonstrating the complexity of user information. Meanwhile, directly uploading user information is a violation of privacy and unacceptable. To tackle the above challenges, we propose an efficient and privacy-enhanced Federated Recommendation for the Joint Venture Ecosystem (FR-JVE) that each client transfers more common knowledge from other clients with a distilled user's rating preference from the local dataset. More specifically, we first transform the local data into a new format and apply model inversion techniques to distill the rating preference with frozen user gradients before the federated training. Then, a bridge function is employed on each client side to align the local rating preference and aggregated global preference in a privacy-friendly manner. Finally, each client matches similar users to make a better prediction for overlapped users. From a theoretical perspective, we analyze how effectively FR-JVE can guarantee user privacy. Empirically, we show that FR-JVE achieves superior performance compared to state-of-the-art methods.

Section: Introduction
Recommendation systems have emerged as crucial tools and products, significantly impacting daily lives by offering tailored suggestions of new items that may interest users. These systems fundamentally rely on centralized servers to consolidate user data, digital behaviors, and preferences, thereby training models for precise recommendation generation [9,39,8]. However, transmitting local user information to central servers poses considerable privacy and security concerns. Moreover, recent stringent government regulations on privacy protection underscore the necessity of storing user data locally on devices rather than uploading it to a global server. As a potential remedy to this dilemma, federated learning (FL) emerges as a promising approach, enforcing data localization and enabling the distributed training of a globally shared model [18,12,37,13]. This framework has achieved remarkable success and has been applied to various fields, such as recommendation systems [38,27,14] and smart healthcare [6,34].
In recent years, researchers have studied federated recommendation systems (FedRS), which enable different clients to yield optimal recommendations without breaching user privacy. FCF proposed in [1] is the first FL-based collaborative filtering method, which employs the stochastic gradient approach to update the local model, and FedAvg is adopted to update the global model. The authors in [4] propose to adapt distributed matrix factorization to the FL setting and introduce the homomorphic encryption technique on gradients before uploading to the server. FedNCF [10] adapts neural collaborative filtering to the federated setting which introduces neural network to learn user-item interaction function to enhance model learning ability. To better protect user privacy, the authors in [27] offer a bipartite personalization mechanism for personalized recommendations via keeping the user embedding local. Based on it, FedRAP [21] further emphasizes the differences between user clients with additive personalization techniques. Each participant in the study represents a distinct business group, encompassing a subset of users and items. Notably, while Client 1 and Client 2 share a partial overlap in their user and item sets, they each maintain a unique set of personalized users and items. The primary objective of FedRS is to leverage collaborative training to develop a comprehensive global model, especially enhancing prediction accuracy specifically for those users who are common across the participating clients.
While these methods achieve great success in user-level federated recommendation in which the user engages as the participant in the federated training, we focus on another more challenging and realistic scenario of federated recommendation, specifically within a joint venture ecosystem, where each participant is no longer an individual user but rather a subsidiary or business group. These subsidiaries, each possessing their own users and items, are managed under a unified controlling corporation. For instance, Alibaba [23], as the parent company, oversees various businesses such as Alipay [26], Tmall Shopping [11], and Digital Media and Entertainment Group [31]. These business groups need to collaborate while also protecting their respective user privacy information due to potential conflicts of interest. Each group will engage in federated training as a participant, sharing partially overlapping user and item information (e.g., credits, red envelopes, discounts) while maintaining their own exclusive user bases and item offerings. The intuitive explanation can also be found in a simple 3-client example illustrated in Figure 1.
Compared to traditional federated recommendation scenarios, our ecosystem poses additional challenges. Firstly, there is a significantly larger volume of data involved, as the participants are no longer individual users but rather business groups. Secondly, partial overlap exists between users and items, necessitating selective knowledge sharing among participants to enhance recommendation performance for both shared and unique users. Lastly, there is a trade-off between the privacy of user information and the performance of the recommendation model. Some existing methods upload user embedding to improve model performance, inadvertently leaking private information [10,4]. Others exclusively utilize item embedding while keeping user embedding locally private [27,21]. While this approach can yield satisfactory results in traditional federated recommendation settings, in our scenario, where a large data volume is involved, users exhibit partial overlap, resulting in intricate and overlapping user information. Relying solely on the sharing of item embedding falls short of achieving the expected performance.
To address these challenges, we in this paper investigate both a privacy-friendly and efficient method for the joint venture ecosystem. Inspired by [28], most cross-domain recommendation research transfers the common knowledge in the latent space between source and target domains to enhance recommendation performance. To facilitate this transfer, bridge functions are trained to map the embedding of the user and item from the source domain onto the target domain. In our ecosystem, we consider that the local data in each client constitutes a specific domain and study how to more effectively and securely transfer user and item information between clients.
To explore this idea, we first follow the hypothesis of sharing item embedding across clients in a FedAvg manner. While directly sharing user embedding with other clients is unacceptable due to the privacy leakage, we attempt to apply the differential privacy technique [35] to the user embedding to protect user privacy and employ a bridge function for each client to identify similar users between clients. However, the common understanding of differential privacy is that as the privacy budget decreases, the effectiveness of the model decreases, whereas increasing the budget results in increased privacy risks. Then, we propose an efficient and privacy-enhanced federated recommendation framework for the joint venture ecosystem dubbed FR-JVE that constructs the rating preference for each client as the filtering signal to transfer the common knowledge from other clients. Specifically, before the federated training, each client will transform the local dataset into a new data format. Based on it, a secure user's rating preference will be constructed locally via the model inversion technique with frozen user gradients, which will not disclose user privacy. Then, the fixed rating preference will be uploaded, and the central server will transmit the aggregated rating preferences of other clients to each client. During the training process, each client only needs to update and communicate with the item embedding. A bridge function is trained locally to map the downloaded aggregated rating preference to the local rating preference. At the inference stage, each client searches the top-k users well-matched on the rating preference mapped by the bridge function for recommendation.
Through extensive experiments on various datasets with rating tasks, we show that FR-JVE significantly improves the recommendation performance compared to state-of-the-art approaches. The major contributions of this paper are summarized as follows:
• We study the problem of federated recommendation for joint venture ecosystem, in which the main challenges are the massive data volume and complex user/item distribution. Different from traditional user-level federated recommendation, the subsidiary or business group with its own users and items participates in the federated training. The balance between performance and privacy does matter here.
• Then, to address this problem, we propose a novel framework named FR-JVE which can be seen as an off-the-shelf personalization add-on for standard item embedding transmissionbased federated recommendation system and it involves the user information with a secure rating preference in a privacy-friendly manner to improve the performance.
• Finally, we theoretically show that FR-JVE can efficiently guarantee user privacy and empirically conduct extensive experiments on various datasets. Experimental results illustrate that our proposed model outperforms the state-of-the-art methods on rating tasks.
this section cite: ['b8', 'b38', 'b7', 'b17', 'b11', 'b36', 'b12', 'b37', 'b26', 'b13', 'b5', 'b33', 'b0', 'b3', 'b9', 'b26', 'b20', 'b22', 'b25', 'b10', 'b30', 'b9', 'b3', 'b26', 'b20', 'b27', 'b34']

Section: Related Work
Federated Learning Federated Learning (FL) aims to train a global model collaboratively by aggregating locally trained models from multiple clients, each utilizing its own private dataset [17,16,15,19]. FedAvg [29] is a widely known FL framework that enhances the global model by combining parameters from locally trained models. Methods like [25] employ knowledge distillation with unlabeled samples as a proxy dataset. Recently, data-free knowledge distillation techniques leveraging adversarial methods for data generation have garnered attention [45,43,36]. In this manuscript, the FL framework has been adopted to the recommendation system to provide privacy protection for user information and recommendation services.
Federated Recommendation System Federated recommendation systems have garnered significant interest lately, fueled by heightened privacy concerns [40,24,22]. Recent efforts have primarily concentrated on leveraging the interaction matrix, the cornerstone of basic recommendation scenarios [42,2]. FCF [1], a pioneering collaborative filtering method under federated learning (FL), utilizes stochastic gradients to update local models and FedAvg for global model aggregation. To protect user privacy, FedMF [4] integrates distributed matrix factorization into the FL framework, encrypting gradients before transmission to the server. Additionally, federated recommendation methods utilizing diverse data sources have emerged, accounting for multiple information streams. FedFast [33] enhances FedAvg with an active aggregation strategy to accelerate convergence, while Efficient-FedRec partitions the model into a server-side news model and a client-side user model, minimizing computational and communication overheads. Both approaches go beyond the interaction matrix, incorporating user features and new attributes. Meanwhile, PFedRec [27] offers a bipartite personalization mechanism for personalized recommendations. Existing research usually focuses on the user-level federated recommendation, where each user is involved as a participant. FedCORE [20] proposes a cross-organization federated recommendation framework for cold-start users. In this
this section cite: ['b16', 'b15', 'b14', 'b18', 'b28', 'b24', 'b44', 'b42', 'b35', 'b39', 'b23', 'b21', 'b41', 'b1', 'b0', 'b3', 'b32', 'b26', 'b19']

Section: Rating Preference

this section cite: []

Section: At Inference Stage

this section cite: []

Section: For each client:
Privacy-Friendly
this section cite: []

Section: ?

this section cite: []

Section: User Item

this section cite: []

Section: Rating

this section cite: []

Section: Transferred Preference
Top-K Matching paper, we study a novel federated recommendation training framework for a joint venture ecosystem and better enhance user privacy while improving recommendations.
this section cite: []

Section: Problem Formulation
Federated Recommendation. We aim to collaboratively train a global model for K total clients in FedRS. We consider clients to have a partially shared user set and item set. Each client k can only access to his local private dataset
D k = {U k , V k , R k }, where U k = {u 1 k , u 2 k , • • • , u i k } has i users, V k = {v 1 k , v 2 k , • • • , v j k } has j items and R k = {r 11 , r 12 , • • • , r ij }, r ij ∈ R
denotes the interaction of user u i to the item v j . When original local data {U k , V k , R k } enters the forward process, matrix factorization will transform it into the representation (u k , v k )E k = (u k,e , v k,e ). The global dataset is considered as the composition of all local datasets
D = {D 1 , D 2 , • • • , D K } = K k=1 D k .
The objective of the FedRS is to learn a global embedding model E that minimizes the total empirical loss over the entire dataset D:
min E L(E) := K k=1 |D k | |D| L(E k ).(1)
L(E k ) = (i,j)∈R k 1 D k (r ij -rij ) 2 , rij = [u i k,e ] T v j k,e . 1 ⃝ (i,j)∈R k 1 D k -(r ij log rij + (1 -r ij ) log(1 -rij )). 2 ⃝ (2
)
where L(E k ) is the loss in the k-th client. Here we define two loss functions for 1 ⃝ rating prediction and 2
⃝ top-k recommendation.
Cross-Domain Recommendation. Cross-Domain Recommendation (CDR) focuses on transferring preference knowledge from an auxiliary source domain to a target domain, effectively addressing the data sparsity issue inherent in conventional recommendation systems. Given the necessity of knowledge transfer in CDR, recent research has predominantly emphasized user embedding and mapping relationships, leveraging latent space bridge functions to transfer user preferences. Suppose that {u s,e , v s,e } denotes the user/item embedding of the source domain and {u t,e , v t,e } denotes the user embedding of the target domain. Embedding bridge functions {ϕ u , ϕ v } are tasked with mapping the user/item embedding from the source domain to the target domain. The learning process is formalized as a supervised regression problem that minimizes the following loss:
min ϕu ū L map (ϕ(u s,e ), u t,e ), min ϕv v L map (ϕ(v s,e ), v t,e ).(3)
where the ū denotes the shared users and v denotes the shared items, the bridge function is defined with a linear layer, and the loss function L map with the mean square error loss. During the local inference stage, the final rate is predicted for the user u i and item v j by averaging the ratings:
rij = |ū| n=1 1 |ū| [ϕ u (u i s,e )] T ϕ v (v j s,e ).(4)
where rij denotes the predicted rating for the user u i t,e and item v j t,e .
this section cite: []

Section: Federated Recommendation for the Joint Venture Ecosystem
In this section, we first analyze the knowledge transfer techniques used in the centralized cross-domain recommendation and examine their application in our ecosystem with partially overlapped user bases and item offerings. We then discuss an important issue: directly uploading the user embedding will breach user privacy. We adopt differential privacy to user embedding, ensuring privacy. Unfortunately, it inevitably results in an unbalance between performance and privacy. Motivated by these findings, we finally present our both efficient and privacy-enhanced FR-JVE method.
this section cite: []

Section: Exploration of Cross-Domain Recommendation for Our Ecosystem
In traditional user-based federated recommendation systems, each user participates in federated training to obtain better recommendation results tailored to themselves. In such scenarios, user information is relatively scarce, and promising results can be achieved solely through interactions based on item information. However, in our joint venture ecosystem, where participants act as subsidiaries, there exist vast numbers of users and items. It necessitates federated training to recommend suitable products to all users, where user information plays a pivotal role.
this section cite: []

Section: ML 100K ML 1M
1.00
1.25 1.50 1.75 2.00 2.25 2.50 MAE FedCDR FedRAP EMCDR -w/FL ML 100K ML 1M 1.0 1.5 2.0 2.5 3.0 3.5 RMSE FedCDR FedRAP EMCDR -w/FL Inspired by [28], the crossdomain recommendation has achieved great success in transferring knowledge from other domains to enhance recommendation performance. Here we try to adopt this technique into the user-based FedRS. Since the client can communicate with the item embedding without breaching user privacy, we just employ a single bridge function to map the user embedding. Suppose that the client k implements an embedding bridge function ϕ k which maps the local user embedding with user uid. The learning process is formalized as a supervised regression problem that aims to minimize the following loss:
min ϕ k u i k ∈U L map ϕ k (u k,t ), u k,e ,(5)
where where the u k,t denotes the global user embedding calculated by weighing other user embedding in a FedAvg manner and U denotes the shared user. For simplicity, we define the bridge function with a linear layer and the loss function L map with the mean square error loss. During the local inference stage, each client makes the final rate prediction for the user u i k and item v j k by averaging the ratings by similar users:
u k,t = K/{k} i=1 |D i | |D|/|D k | • u k,e .
rij = |U | k=1 1 |U | [ϕ k (u k,t )] T v j k,e .(6)
where rij denotes the predicted rating. As shown in Fig. 3, we test three methods on two datasets, and EMCDR -w/FL achieves a significant improvement on both metrics that verifies the effectiveness of traditional CDR methods in our scenario. Although precise matching through user-embedding can ensure the maximum transfer of common knowledge for the shared user, directly exposing user-embedding to clients in federated learning severely violates user privacy and is undesirable.
this section cite: ['b27']

Section: Why does Differential Privacy Fail to Work?
To better protect user privacy, we try to adopt differential privacy to the uploaded user embedding. First, we conduct experiments with the LDP technique on MovieLens-100K in Table 1. As expected, with a smaller privacy budget, user privacy can be guaranteed and when the privacy budget exceeds a certain threshold, the effectiveness of privacy information protection will no longer be achieved. However, due to the relatively small number of parameters in the embedding, when perturbation is applied to the model parameters to effectively protect user information, the recommendation performance decreases significantly. Achieving a balance between user privacy and recommendation performance is challenging. To address this issue, we next propose a novel FedRS framework with enhanced privacy and satisfying performance.
this section cite: []

Section: FR-JVE: Secure Preference-based Framework
To better balance the privacy and recommendation performance, we avoid directly exposing the user embedding and construct a secure rating preference for each client, serving as the transferable collaborative filtering signals to match similar users rather than the user embedding. Intuitively, a user's rating preference is suitable to represent the main characteristics of that user. For example, in [44], the rating preference of the user is the weighted sum of item embedding that is rated by this user.
To ensure user privacy, here we apply model inversion techniques to distill the rating preference from both the user and item information for each client. Furthermore, we observe that the data format of each user and item is indifferentiable one-hot representation, impeding the optimization of the model inversion training. The framework of our proposed method is illustrated in Fig. 2. Consequently, we first apply the normal distribution and softmax function to initialize the data format for model inversion. Suppose that the user u i k and item v j k in client k is recognized as :
u i k = (u 1 , u 2 , • • • , u |U k | ), v j k = (v 1 , v 2 , • • • , v |V k | ).(7)
where u i , v j ∈ {0, 1}. Then we transform the one-hot representation with normal distribution and softmax function:
u i = exp(û i ) |U k | i=1 exp(û i ) , v j = exp(v j ) |V k | j=1 exp(v j )
, where ûi , vj ∼ U(0, 1).
With the new format defined in Eq.( 8), each client can distill the rating preference for the user u i k with the model f k :
(p i k |u i k , r i ) = min v k L(f k (u i k , v k ), r i ).(9)
During the training process, the gradient of u i k is frozen. Each element of the rating preference
p i k = (p 1 , p 2 , • • • , p |V k | )
is a weight for the whole item embedding V k . Then, the client k will train the matrix factorization with the new data format like Eq.( 1) to obtain the item embedding. It may raise concerns that an individual user might overlook numerous items. However, many other users involved in the federated training have rated these neglected items. As a result, these ratings contribute to well-generalized item embeddings. The overall item embedding embodies enriched knowledge, thereby facilitating the informative user characteristics. Inspired by the attention mechanism, which enables each component to contribute distinctively when compressing various elements into a singular representation, we obtain the final rating preference P i k for the user u i k as follows:
P i k = |V k | j=1 p j v j k,e .(10)
In this work, the rating preference will be used to transfer the common knowledge from other clients, which acts as a collaborative filtering signal enhanced with secure user information. Like Eq.( 5), each client locally trains a bridge function to map the aggregated rating preferences from other clients to the local rating preference.
However, simply employing the rating preference cannot match similar users like Eq.( 6). Here we search the top-k similar users with the rating preference for better prediction:
S i k = top_k |r -min u i k,e [u i k,e ] T ϕ k (P i k,t )| . (11
)
Where S i k is the matrix where each column is a searched user embedding, K and r are hyperparameters that denote the number of collected users and the expected recommendation score, ϕ(•) is a bridge function defined in Eq.( 5) to map the rating preference here. Then, the final prediction for the user u i k will be implemented like Eq.( 10). The workflow and privacy analysis of FR-JVE are provided in Appendix A and B.
this section cite: ['b43']

Section: Experiments

this section cite: []

Section: Datasets & Baselines
A thorough experimental study has been conducted to assess the performance of the FR-JVE in two popular scenarios with four recommendation datasets: (1) Rating Prediction: MovieLens-100K (ML-100K) [7] and MovieLens-1M (ML-1M). (2) Top-K Recommendation: LastFM-2K(LastFM) [3] and QB-article [41]. The details of these datasets are outlined in Appendix D.
For a fair comparison with other works, we follow the same protocols proposed by [29] to stimulate FL settings. We evaluate our method with the following baselines. 1. User-based Federated Recommendation: FedMF [4], FedNCF [10], PFedRec [27], FedRAP [21]; 2. Customed Methods: FedMF [4]-w/DP, EMCDR [28]-w/FL+DP; 3. Federated Cross-Domain Recommendation: FedCDR [30], P2FCDR [5]. More details are provided in Appendix C.
this section cite: ['b6', 'b2', 'b40', 'b28', 'b3', 'b9', 'b26', 'b20', 'b27', 'b29', 'b4']

Section: Configurations
Unless otherwise mentioned, to build the system with partially overlapped users and items, we first assign each client with random private users and items and then use the Dirichlet distribution Dir(α = 10) [32] to distribute shared users to yield data heterogeneity for all datasets where a smaller α indicates higher data heterogeneity. Here, we report the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) as the metrics for the rating prediction [28] and Hit Rate (HR@K) and Normalized Discounted Cumulative Gain (NDCG@K) for the top-k recommendation [27,21]. In this work, we set K = 10. We illustrate all the settings with all the benchmark parameters in Appendix D.
this section cite: ['b31', 'b27', 'b26', 'b20']

Section: Performance Overview
Main Results. Table 2 and 3 comprehensively showcase the efficacy of various methods on two tasks. For the top-k recommendation task, when examining the HR@10 metric, FR-JVE outperforms other methods by significant margins. As the evaluation metrics extend to higher positions, FR-JVE continues to exhibit robust performance, indicating its ability to provide relevant recommendations even when considering a larger pool of candidates. Notably, PFedRec and FedRAP outperform the Federated CDR method across most datasets and achieve relatively good results due to the gain of the linear layer within their frameworks. The performance of EMCDR-w/FL+DP is closely related to the privacy budget, and the model undergoes a significant degradation under strict privacy requirements.
Here, we report the case with ϵ = 0.2, and the remaining analysis is provided in Table 1.
Communication Efficiency. Fig. 4 shows the evaluation of various methods in terms of convergence and communication efficiency. Here, we record evaluations for these methods for every two iterations. Under this joint venture ecosystem, the convergence of the FedMF method is extremely challenging, making it difficult to achieve effective recommendation performance. While FedRAP is more complex than other baselines, it needs more iterations to converge. FR-JVE achieves the best evaluations in less than 20 iterations and displays a fast convergence trend within a shorter time of 80 iterations.
this section cite: []

Section: Ablation Study.
As shown in Table 5, we evaluate the effects of each module in our model via ablation studies. The -w/o bridge function denotes the performance without using the bridge function but directly employing the rating preference from other clients. The -w/o rating preference means = 10% = 25% = 80% Overlapped User Ratio 1 2 3 4 5 6 7 MAE ML-100K = 10% = 25% = 80% Overlapped User Ratio 2 4 6 8 RMSE ML-100K FedMF -w/DP EMCDR -w/FL+DP FedCDR FedRAP FR-JVE Figure 5: Performance comparison of various methods w.r.t. ratio β between local unique users and shared overlapped users. the client communicates with the only item embedding without user information. Compared with FR-JVE, the performance of FR-JVE -w/o rating preference degrades evidently. Specifically, the relatively less prominent role of the bridge function module may be constrained by issues related to the dataset, specifically that different clients did not introduce many new items or items that are sparse among clients. Experiment results verify the effectiveness of all modules.  Data Distribution. Fig. 5 displays two metrics with different ratios between local unique users and overlapped users on MovieLens-100K. As shown in this figure, we select three different ratios to divide the dataset and experimental results show that all methods achieve an improvement with the decline in local unique users. The underlying reason may be that the more overlapped users there are, the more associations between users can be captured, and different clients can obtain useful information from other clients to improve recommendation performance. However, when the number of locally unique users is large, the differences in data distribution among different clients are relatively significant, leading to a decline in the effectiveness of federated learning. Parameter Sensitivity Analysis. Fig. 6 provides the two metrics under different ratios between active clients and total clients. FR-JVE performs best with different ratios, and the lower MAE and RMSE loss is achieved by applying more active clients. Furthermore, we conduct additional research on the number of searching users K. When the value of K is too small, the client fails to migrate sufficient knowledge from limited similar users, leading to insignificant improvement in performance. Conversely, if an excessive amount of user knowledge is migrated, it may introduce redundant information that impacts the model's judgment.
To discuss the potential extra time cost introduced by the top-k search, we conducted time tests for different k values while controlling other variables in Table 6. We assume other methods incur equal time costs with similar inference approaches. FR-JVE requires an extra step for a top-k search. The cost for "Other baselines" should theoretically be independent of the k value, but there are fluctuations. The experiments show that our top-k search does not significantly increase the cost. We use a linear function for the mapping since our experiments are conducted on standard datasets. Although real-world data may require more complex bridge functions, the results in Table 7 show limited performance gains from more complex architectures. Complex bridge functions risk overfitting, especially with sparse or noisy transferred knowledge, whereas a simple linear network effectively captures essential cross-client relationships while maintaining good generalization.
Privacy Enhancement. In our method, each client transfers privacy-friendly common knowledge from other clients without using the LDP algorithm, but with a distilled user's rating preference from the local dataset. Furthermore, we also provide experiments in Table 8 that apply differential privacy to our rating preferences for reference. However, our method itself can guarantee user information privacy. Considering that differential privacy can impact performance, we did not utilize it further.
this section cite: []

Section: Scalability and Efficiency.
In this paper, we focus on federated recommendation within a jointventure ecosystem, where each participant holds substantial user-item data with natural overlaps. This setting aligns with cross-silo FL, featuring few participants but complex data distributions. In contrast, cross-device FL (e.g., C > 20) involves numerous users with limited data, which differs from our target scenario. To validate our approach, we vary the number of participants C across three configurations on two datasets. Results in Table 9 confirm the effectiveness of our method.
this section cite: []

Section: Flexibility and Lightweight.
The key of FR-JVE is to allow each client to transfer more common knowledge from other clients with a distilled user's rating preference from the local dataset. Since this rating preference is unrelated to the backbone of federated recommendation algorithms, we can integrate it with other basic algorithms. In our paper, we employed the most common collaborative filtering. We will supplement the integration with well-known FedNCF and pFedRec algorithms. Experimental results 10 demonstrate that our algorithm can be combined with any basic federated recommendation algorithm to enhance performance.
this section cite: []

Section: Conclusion
In this paper, we delve into the challenges of developing a federated recommendation system tailored for joint venture ecosystems. We propose an innovative and privacy-enhancing framework, FR-JVE, which leverages the user's rating preferences as a filtering signal to transmit common knowledge, thereby enhancing recommendation performance while safeguarding user privacy. Extensive experiments conducted on various settings and baselines show that FR-JVE achieves significant improvement in recommendation performance.
this section cite: []

Section: References
Ref_id:b0 Title: Federated collaborative filtering for privacy-preserving personalized recommendation system Year: (2019)
Ref_id:b1 Title: Federated collaborative filtering for privacy-preserving personalized recommendation system Year: (2019)
Ref_id:b2 Title: Second workshop on information heterogeneity and fusion in recommender systems (hetrec2011) Year: (2011)
Ref_id:b3 Title: Secure federated matrix factorization Year: (2020)
Ref_id:b4 Title: Win-win: a privacy-preserving federated framework for dual-target cross-domain recommendation Year: (2023)
Ref_id:b5 Title: Federated classincremental learning Year: (2022)
Ref_id:b6 Title: The movielens datasets: History and context Year: (2015)
Ref_id:b7 Title: Ecat: A entire space continual and adaptive transfer learning framework for cross-domain recommendation Year: (2024)
Ref_id:b8 Title: Recommendation systems: Principles, methods and evaluation Year: (2015)
Ref_id:b9 Title: Fedncf: federated neural collaborative filtering for privacy-preserving recommender system Year: (2022)
Ref_id:b10 Title: Multi-interest network with dynamic routing for recommendation at tmall Year: (2019)
Ref_id:b11 Title: A review of applications in federated learning Year: (2020)
Ref_id:b12 Title: Towards efficient replay in federated incremental learning Year: (2024)
Ref_id:b13 Title: Personalized federated recommendation for cold-start users via adaptive knowledge fusion Year: (2025)
Ref_id:b14 Title: Re-fed+: A better replay strategy for federated incremental learning Year: (2025)
Ref_id:b15 Title: Unleashing the power of continual learning on non-centralized devices: A survey Year: (2024)
Ref_id:b16 Title: Fedssi: Rehearsalfree continual federated learning with synergistic synaptic intelligence Year: ()
Ref_id:b17 Title: Sr-fdil: Synergistic replay for federated domain-incremental learning Year: (2024)
Ref_id:b18 Title: Personalized federated domain-incremental learning based on adaptive knowledge matching Year: (2024)
Ref_id:b19 Title: Fedcore: Federated learning for cross-organization recommendation ecosystem Year: (2024)
Ref_id:b20 Title: Federated recommendation with additive personalization Year: (2024)
Ref_id:b21 Title: Fedrec++: Lossless federated recommendation with explicit feedback Year: (2021)
Ref_id:b22 Title: E-business ecosystem and its evolutionary path: the case of the alibaba group in china Year: (2009)
Ref_id:b23 Title: Fedrec: Federated recommendation with explicit feedback Year: (2020)
Ref_id:b24 Title: Ensemble distillation for robust model fusion in federated learning Year: (2020)
Ref_id:b25 Title: The role of alipay in china Year: (2015)
Ref_id:b26 Title: Continual federated learning based on knowledge distillation Year: ()
Ref_id:b27 Title: Cross-domain recommendation: An embedding and mapping approach Year: (2017)
Ref_id:b28 Title: Communication-efficient learning of deep networks from decentralized data Year: (2017)
Ref_id:b29 Title: Fedcdr: federated cross-domain recommendation for privacy-preserving rating prediction Year: (2022)
Ref_id:b30 Title: China's "national champions" alibaba, tencent, and huawei Year: (2019)
Ref_id:b31 Title: Estimating a dirichlet distribution Year: (2000)
Ref_id:b32 Title: Fedfast: Going beyond average for faster training of federated recommender systems Year: (2020)
Ref_id:b33 Title: The future of digital health with federated learning Year: (2020)
Ref_id:b34 Title: Ldp-fed: Federated learning with local differential privacy Year: (2020)
Ref_id:b35 Title: Dafkd: Domainaware federated knowledge distillation Year: (2023)
Ref_id:b36 Title: FedCDA: Federated learning with cross-rounds divergence-aware aggregation Year: (2024)
Ref_id:b37 Title: Amazon-kg: A knowledge graph enhanced cross-domain recommendation dataset Year: (2024)
Ref_id:b38 Title: Large language model-based generative retrieval in alipay search Year: (2024)
Ref_id:b39 Title: Federated recommendation systems Year: (2020)
Ref_id:b40 Title: Tenrec: A large-scale multipurpose benchmark dataset for recommender systems Year: (2022)
Ref_id:b41 Title: Lightfr: Lightweight federated recommendation with privacy-preserving matrix factorization Year: (2023)
Ref_id:b42 Title: Fine-tuning global model via data-free knowledge distillation for non-iid federated learning Year: (2022)
Ref_id:b43 Title: Personalized transfer of user preferences for cross-domain recommendation Year: (2022)
Ref_id:b44 Title: Data-free knowledge distillation for heterogeneous federated learning Year: (2021)
