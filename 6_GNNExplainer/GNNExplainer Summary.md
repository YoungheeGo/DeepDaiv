# GNNExplainer Summary

# Introduction

GNN의 가장 큰 장점은 “관계성”이다. Non-relational Data, 즉 우리가 일반적으로 하는 데이터들은 각 데이터가 Independent 하다는 가정을 가지고 있다. 하지만 Social Network, Chemical Data등 관계가 중요한 요소로 자리매김 하는 Relation data는 이전의 데이터와 차이가 있기에 Relation data는 다른 기법을 사용해 분석해야한다. GNN은 바로 이 관계성을 담을 수 있는 기법이고 사회, 정보, 화학 그리고 생명 도메인 등 현대 세계의 데이터를 잘 분석할 수 있는 데이터이다.

![img0](./Fig/fig0.png)

위 그림으로 GNN XAI에 대한 overview를 볼 수 있다. 먼저 input graph의 하나의 노드 embedding을 도출하고 classification을 하는 모델이 있다면, 해당 노드를 특정 class로 분류함에 있어 영향을 미친 노드들, 즉 subgraph를 찾는 것이 GNN XAI의 목적이다.

# Related Works

Explainability 기법 종류 : 기존의 ML/DL에서 사용했던 방식 정리 (GNN의 Explainability 선례X)

- Surrogate / Proxy Model 을 구성한 뒤, Explanation 찾기
- High level feature을 설명할 수 있는 연관변수 찾기
- Computation 과정에서 영향을 크게 주는 중요한 변수 찾는다.

→ 하지만, 이런 기법들을 GNN에 사용하기에 부적합한 기법인데, 인접 행렬 에서 문제가 증폭될 수 있기 때문이다.

- 인접행렬 (Adjacency Matrix)
    
    ![img1](./Fig/fig1.png)
    

### GNN Explainer

GNN Explainer는 GNN의 예측을 설명하기 위한 기법. 훈련된 GNN과 Prediction을 가지고 특정 노드나 Class에 대해 Subgraph를 구성하며, 이를 통해 어떤 노드가 하나의 노드 / Class를 구성하는 데 영향을 주었는 지 알 수 있습니다. 또한 영향 노드만을 구할 수 있는 것이 아니라, 노드의 어떤 Feature의 영향력이 큰 지에 대해서도 알 수 있어 심화적인 분석이 가능합니다.

- 하나의 노드를 구성함에 있어 어떤 이웃 노드들의 영향을 미쳤는지 (Single-instance explainations)
- 하나의 라벨을 갖는 노드들에 대해 어떤 이웃의 노드들이 영향을 미쳤는지 (Multi-instance explainations)
- 노드 내 어떤 feature가 영향을 미쳤는지

### 예시] basketball과 sailing 그래프에서 노드 분류

![img2](./Fig/fig2.png)

각 노드는 학생, label은 취미운동에 대한 그래프를 나타낸다. $\Phi$는 Node Classification을 진행하는 GNN Model이라고 했을 때, $v_i, v_j$노드가 어떤 운동을 선택할 것인지 판단하는 문제로 정의할 수 있다. 빨간색 모델이 $v_i$를 농구라고 분류하고 초록색 모델이 $v_j$를 sailing이라고 분류하는 경우이다. 훈련된 모델과 Prediction이 존재할 때, GNN Explainer는 각 노드가 판단을 하기 위한 요소들이 무엇인지 판단해 우측의 그림처럼 subgraph를 나타낸 것으로 볼 수 있다. 

# Formulating explanations for graph neural networks

![img3](./Fig/fig3.png)

GNN기법으로 Node $v$의 Embedding $z$를 판단하고자 한다.

- $G_c(v)$ : v를 z Embedding으로 구성하는 Computation Graph.
    - Computation Graph
        
        계산과정을 위해 필요한 노드들을 엮은 그래프. 위의 그림 A에서 색으로 표시된 그래프들이 Computation Graph이며, 초록색은 $\hat {y}$ 을 구성할 때 중요한 노드들이며, 노란색은 반대의 의미.
        
- $A_c(v) \in 0,1^{n*n}$ : Adjacency Matrix 는 0 또는 1 의 값으로 n*n 차원으로 이루어져있다.
- $X_c(v) = x_j | v_j \in G_c(v)$ : $v$를 구성하기 위한 Computation Graph내에 있는 노드들의 Feature. Neighbors Set
- $P_{\Phi} (Y|G_c,X_c)$: 노드들의 feature들과 Computation Graph를 사용했을 때, {1,...,C}종류의 label을 가질 확률

기존 모델에 대한 Notation이고 GNN Explainer은 추가적인 노테이션 필요

- $G_s$: Computer Graph $G_c(v)$의 Subgraph
- $X_s$: $G_s$에 포함되어 있는 Feature
- $X_s^F$: $X_s$중 일부의 featues, 위의 그림 B에서 빨간색 X를 제외한 feature, 삭제되지 않은 features

# GNNExplainer

### Single Instance Optimization

최적화를 위해 단일노드 v에 대해 Computation Graph $G_c$ 내 포함되어 있는 Subgraph $G_s$ 중, Mutual Information이 가장 큰 $G_s$를 선택한다.

![img4](./Fig/fig4.png)

노드 $v_i$의 Embedding을 구성하여 라벨을 정의하기 위해 모델에서 사용하는 모든 노드들을 $G_c$라고 했을 때, 전부를 활용하는 것도 가능하지만, 그 중 필요한 노드와 불필요한 노드는 존재하고, 불필요한 노드는 결과를 저해할 수 있다. 따라서 핵심적인 노드만을 판단하기 위해 $G_c$내에 존재하는 여러 경우의 $G_s$를 구해 구 중 Mutual Information이 최대인 $G_s$를 구하는 것이다. 위의 그림에서 $G_c$에서 $G_s$후보들을 나타내는 것을 볼 수 있다. Mutual Information(MI)에 대한 식은 다음과 같은 식으로 표현한다.

먼저, MI에 대한 일반적인 정의는 두 변수의 상호 종속여부라고 할 수 있는데, 두 변수가 독립이라고 했을 때와 종속이라고 생각했을 때의 두 변수의 결합확률의 차이로서, 이 값이 크게 되면 두 변수가 독립이 아니라 종속에 가까우므로 관계가 크다고 할 수 있다. 즉, 위 식에서는 $v_i$의 예측값인 Y와 Subgraph $G_s, X_s$의 연관성이 크다고 할 수 있다.

$\max_{G_s} MI(Y,(G_s,X_s)) = H(Y) - H(Y|G=G_s, X=X_s)$은 Subgraph

MI의 최적화 식에서 H(Y)는 모델에서 모든 Computation Graph를 사용했을 때의 Return값에 대한 Entropy이므로 고정된 상수입니다. 따라서 해당 최적화 식을 오른쪽의 $H(Y|G=G_s, X=X_s)$를 최소화 시키는 것으로 전환할 수 있게 된다.

$\min H(Y|G=G_s, X=X_s)$이

이 식은 subgraph $G_s$의 Entropy인데 원래 Entropy란 불확실성을 의미하기에 이를 최소화 한다는 뜻은 모델 $\Phi$의 불확실성을 줄인다는 뜻이 된다. 따라서 최적화 식의 의미가 목적과 결부하게 되기 때문에 $G_s$의 크기를 $K_M$으로 한정시켜 적당한 크기로 유지하게 된다.

하지만, 해당 최적화 식은 직접적으로 사용할 수 없다. 그 이유는 $G_c$내의 수많은 $G_s$가 존재하며 이들에 대해 모두 비교해 최적의 $G_s$를 판단하는 것은 불가능하기 때문이다. 하나의 $G_c$가 M개의 노드를 갖고 있다면, 포함되어 있는 $G_s$의 개수는 $2^M$개 이다. 수많은 $G_s$는 interactable하므로 이를 해결하기 위해 $G_s$에 대해 근사를 진행하며 이는 Random Graph를 Variational Distribution으로 정의하는 Variational Inference를 포함하고 있다. ($G_s \sim g$ 를 가정한다.) 또한 $G_s$들을 분포로 나타내기 위해 Continuous Relaxation을 진행하며, Adjacency에 초함되어 있는 각 셀을 0과 1사이의 확률값으로 나타낸다.

$\min_g E_{G_S \sim g} H(Y|G=G_s, X=X_s)$ 

그리고 이를 젠센 부등식으로 upper bound로 전환하면, 

$\min_g H(Y|G=E_g[G_s],X=X_s)$

로 표현할 수 있습니다. 이를 Upper Bound를 minimize하는 것이 원래의 식을 minimize하는 것과 같은 맥락을 갖게 된다.

따라서 이제 $E_g[G_s]$를 도출해야하므로, $G_s$를 계산하기 위해 Mean Field Variational Approximation을 사용한다. 이는 기대값을 구하기 편한 방법이며 단순히 결합분포를 구하기 위해 Marginal Distribution을 모두 독립으로 간주하고 곱한것이다. 

→ $P_g(G_s) = \Pi_{(j,k) \in G_c} A_s[j,k]$

$A_s$는 (j,k)번째의 edge $(v_j,v_k)$가 존재하는지에 대한 기대를 나타낸다. 경험적으로 이 근사치가 GNN의 non-convexity 에도 불구하고 규제항과 함께 국소 최소값으로 수렴한다는 것을 관측했다. 따라서 인접행렬의 computation graph의 마스킹을 함으로써 최적화 될 수 있는  $E_g[G_s]$값을 $A_c \circledcirc \sigma(M)$으로 대체한다.  여기서 $M \in \mathbb{R}^{n*n}$이고 이는 우리가 학습할 필요가 있는 마스크를 나타낸다. 그리고, $\sigma$는 마스크를 $[0,1]^{n*n}$에 매핑하는 시그모이드를 나타낸다.

- Masking
    
    $A_s[j,k]$의 값이 연속적으로 작을 경우, 0으로 수렴하기 때문에 Masking한다라는 표현을 사용한 것 같아요!
    

그리고 모델의 신뢰도의 관점에서 설명을 하려 한다면 $\min_g H(Y|G=E_g[G_s],X=X_s)$식을 다음과 같이 수정할 수 있다.

$\min_M -\sum_{c=1}^C \mathbb{1} [y=c] log P_\Phi (Y=y|G=A_c \circledcirc \sigma(M), X=X_c)$

### Joint learning of graph structural and node feature information

어떤 노드들이 $\hat{y}$를 예측하는데 있어 가장 중요한지 식별하기 위해 GNN Explainer은 $G_s$내 노드들의 feature selector F를 학습한다. 모든 노드 피쳐를 구성하는 $X_s= \{x_j | v_j \in G_s\}$를 정의하기보다 GNN XAI는 $G_s$의 노드들의 피쳐의 집합으로 $X_s^F$로 구성된다. 이것은 이진 피쳐 셀렉터 $F\in \{0,1\}^d$를 통해 정의된다.

$X_s^F = \{x_j^F| v_j \in G_s\}$ , $x_j^F = [x_{j,t_1}, ..., x_{j,t_k}]$ for $F_{t_i}=1$이

$x_j^F$는 F에 의해 masked out 되지 않는 노드 피쳐들이다. mutual $(G_s, X_s)$는 information objective를 최대화 하는데 있어서 결합적으로 최적화된다. (다음 식 참고)

$\max_{G_s,F} MI(Y,(G_s,F)) = H(Y) - H(Y|G=G_s, X=X_s^F)$이 식은 가장 처음에 나온 식을 수정한 것이다. $\hat{y}$를 예측하기 위해 설명을 하는 node feature information과 구조를 고려한다.

# Experiments

### Synthetic dataset

![img5](./Fig/fig5.png)

- BA-shapes
    - 300개의 노드를 가진 Barabasi-Albert(BA) Graph에서 5개씩 80개 노드로 구성된 House꼴의 Motifs를 랜덤하게 선택된 노드에 연결한다.
    - 0.1N(이 때, N은 노드의 개수)개의 Edge를 추가적으로 포함해 perturbed를 하는 그래프 결과를 생성한다.
    - 노드는 구조적 역할에 따라 4개의 클래스로 할당이 된다. (house의 top, middle, bottom과 house에 포함되지 않는 노드)
- BA-Comunity
    - 2개의 BA-shapes의 합집합이며, Community와 역할에 따라 8개의 클래스로 나뉘어짐.
- Tree-Cycles
    - 8-level의 balanced binary tree를 구성한 뒤에 6개씩 80개 노드 사이클 motifs를 가져다 붙힌다.
- Tree-Grid
    - Tree-Cycles와 비슷함.
    - 8-level의 tree를 구성한뒤 9개로 이루어진 3*3 grid를 붙힌다.

### Real-world dataset

- MUTAG
    - 4337개의 분자그래프, 해당 분자들이 Gram-Negative Bacterium S.typhimurium으로 인한 유전적 변화로 라벨링 되어 있음
    - 라벨은 1과 -1으로 나타남
- REDDIT-BINARY
    - 2000개의 그래프는 각 discussion thread를 나타냄
    - 노드 : User in Thread
    - Edge: 하나의 유저가 다른 유저에 답장했을 경우
    - 그래프는 Thread내 어떤 User interaction이 이루어져 있는지 판단한다.
        - Question-Answer Interaction : r/IAmA, r/AskReddit
        - Online-Discussion Interaction : r/TrollXChromosomes, r/athesim

### Alternative baseline approach

- GRAD
    - GNN loss function 의 gradient를 인접행렬에서 계산해 Saliency Map을 구성하는 것과 동일하다.
- ATT
    - GAT를 통해서 Computation Graph의 Edge들에 대해 가중치를 계산할 수 있지만 Feature에 대한 가중치는 알 수 없다.

### setup and implementation details

- 각 데이터 셋에 대해 먼저 GNN 적용해 훈련
- GRAD 와 GNN Explainer을 통해 GNN 예측값 설명
- ATT는 GAT를 통해 Attention Weight를 구성
- $K_M,K_F$ 는 각 subgraph와 Feature Explanation의 size를 나타낸다.
- Synthetic Dataset에서는 $K_M$을 ground truth로 설정
- Real Dataset에서는 $K_M=10$으로 나타내며, $K_F$는 모든 데이터 셋에서 5로 지정함
- ground truth
    
    Ground truth is information that is known to be real or true, provided by direct observation and measurement (i.e. empirical evidence) as opposed to information provided by inference.
    
    ref: [https://en.wikipedia.org/wiki/Ground_truth](https://en.wikipedia.org/wiki/Ground_truth)
    

### Result

실험을 통해 밝혀내고자 하는 질문은?

- GNN Exlainer가 꽤 괜찮은 설명을 할 수 있는가?
- Ground-Truth Knowledge와 Explanation을 어떻게 비교하는가?
- Graph기반 예측 task에서 GNN Explainer를 어떻게 사용하는가?
- 훈련된 GNN 말고 다른 GNN에도 바로 적용할 수 있는가? (생산성의 관점)

**1) Quantitative Analyses**

![img6](./Fig/fig6.png)

Node Classification Dataset에 대한 Accuracy를 위의 표에서 확인할 수 있다. 

- Predicted : Explainability Method로 예측해 낸 Important Weights
- Label : Ground-truth Explanation에 포함되어 있는 Edge

즉, Ground-Truth Label을 예측해나가는 Binary Classification으로 문제를 정의해 나갔다. 더 우수한 Explainability Method는 더 정확히 Ground Truth Edge들을 예측해냈기에 Accuracy가 더 높게 나타남

**2) Qialitative Analyses**

![img7](./Fig/fig7.png)

![img8](./Fig/fig8.png)

![img9](./Fig/fig9.png)

- 위 세개의 그림을 통해 GNN Explainer이 가장 Ground Truth의 그래프 형태와 비슷하다는 것을 질적 연구로 파악할 수 있다.

# Conclusion

- Relational structures (풍부한 노드)에 적용가능
- GNN 예측의 인터페이스 제공
- GNN 모델 디버깅에 도움
- 모델의 실수에 대한 패턴을 식별
- model - agnostic 방법
    - Model-agnositc
        
        학습에 사용된 모델이 무엇인지에 구애받지 않고 독립적으로 모델을 해석하는 방법, 즉 어떤 모델을 사용했더라도 동일한 방식으로 해석할 수 있다.
        
        ref: [https://elapser.github.io/machine-learning/2019/03/08/Model-Agnostic-Interpretation.html](https://elapser.github.io/machine-learning/2019/03/08/Model-Agnostic-Interpretation.html)
        
    

마지막으로 본 논문의 Contribution을 설명하자면

- 최초의 GNN XAI 및 Baseline을 구축
- GNN XAI의 evaluation Methods 정의

ref: 

[https://en.wikipedia.org/wiki/Ground_truth](https://en.wikipedia.org/wiki/Ground_truth)

[http://dsba.korea.ac.kr/seminar/?mod=document&uid=1443](http://dsba.korea.ac.kr/seminar/?mod=document&uid=1443)

[https://velog.io/@tobigs_xai/GNNExplainer-Generating-Explanations-for-Graph-Neural-Networks](https://velog.io/@tobigs_xai/5%EC%A3%BC%EC%B0%A8-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-GNNExplainer-Generating-Explanations-for-Graph-Neural-Networks)

[https://hoonst.github.io/2021/02/03/GNNExplainer/](https://hoonst.github.io/2021/02/03/GNNExplainer/)
