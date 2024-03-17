# Deep-Daiv

- 기간 : 2021.12.15 ~ 2022.03.24
- Personal Goal : 
  1. How to Explain Simply 
  2. Explore a variety field about deep learning
  3. Survive

# week 1 : OT
- False Positive (X)
- True Positivie (O)
- Goal: 10k
- Slack & Notion
- 10h per week

# week 2
- What is GA?
- LaTeX Writing
- Jekyll: Static Generator -> 컨텐츠를 html(static websites)로 변환시키기, github 내부에도 설치됨
  - markdown파일로 글 작성, git push로 업로드, Github pages 내부 Jekyll이 이를 인식한 후 html로 변환후 웹 호스팅  
- Static Website: 어떤 웹사이트 주소를 접속한다면 모든 사람들이 모든 결과물(html)이 동일

# week 3
- [paper] Character-level CNN
- https://arxiv.org/abs/1509.01626
- CNN=Vision이라는 편견 깨기 
- CNN을 NLP에 적용시키기 with code
- [paper] & [code]

# week 4
- [paper] Going Deeper with Convolutions
- https://arxiv.org/abs/1409.4842
- GoogleNet이라고도 불리는 모델 -> ILSVRC에서 SOTA 달성했지만, 레이어가 너무 깊어 공동 1위한 VGGNet에 좀 더 집중함.
- 하지만 그 다음 챌린지인 2015 ILSVRC에서 SOTA를 달성한 모델이 152개의 레이어를 갖는걸 보면, layer 22인 GoogleNet이 영향을 줬을 거라고 생각한다...!
- Deep Learning Architecture 자세히 샅샅이 뜯어봄! 
- 코드도 돌려봤지만, 이전 CNN 공부 방향과는 다르게 '코드 구현'을 집중적으로 하지 않고 Architecture에 집중하며, Parameter에 집중함.
- 1\*1 Conv layer
- Inception
- [paper] & [code]

# week 5
- Reinforcement Learning foundation
- not paper
- 통계학의 시선으로 강화학습 바라보기
- youtube lectures : chapter 1 ~ chapter 4

# week 6
- [paper] Play Atari with deep reinforcement learning
- https://arxiv.org/abs/1312.5602
- RL with CNN
- Deep Q Network (DQN)
- 즉 컴퓨터가 실제로 이미지 데이터를 보면서 스스로 학습해 게임 실행 
- Experience Replay memory
- Input data: pixel -> CNN 사용


# week 7
- [paper] Human-level control through deep reinforcement learning
- https://www.nature.com/articles/nature14236
- Week 6에서 읽은 Atari 게임이랑 강화학습이 비슷한 느낌이어서 찾아보니 같은 모델 DQN이었다.
- 두 논문 모두 딥마인드에서 작성한 paper였고, 2013년의 DQN과 다른점은 다음과 같다. (DQN의 개정판이라고 생각했다)
- 2013년의 DQN은 target vlaue 를 상수취급하여 학습 진행했지만, 2015년의 DQNdms target value를 네트워크로 구성해 performance를 향상시킴
  - 이 과정에서 새로운 파라미터 theta^- 나타남
  - 이터레이션 C 시간동안 Q_hat과 Q를 동일하게 취급함
 

# week 8
- [paper] GNNExplainer - Generating Explanations for Grapha Neural Networks
- https://arxiv.org/abs/1903.03894
- GNN(Graph Neural Network) + XAI (Explainable AI)
- GNN의 가장 중요한 점 : '관계'정보 -> 대부분의 XAI 방법론은 관계정보를 포함시키지 않음
- 따라서 본 논문에서는 Subgraph 와 Node Feature 을 추출하는 기법을 특징으로 볼 수 있음

# week 9 ~ week 10
- Write a article about projects
- First draft
- *와 너무 힘드러 차라리 논문 읽을래...*

# week 11
- Week 8에 이어서 GNNExplainer 논문리뷰 및 코드 리뷰
- GNN 코드 ref : https://github.com/pyg-team/pytorch_geometric
- Mutual Information 최대화하는 것이 본 논문의 핵심
  - 관련 내용을 수식으로 derivation 

# week 12
- CNN + GNN = GCN 기초 개념 정리 
- GNN first paper 읽고 싶었는데 엄두가 안나서 그나마 아는 CNN 과 결합된 논문을 선택했는데 GNN에 대한 이해없이 읽기 시작하니까 시간이 오래 걸렸다.
- 생각보다 어렵고 challenge한 분야였다~~
- 이거 어려워서 2주로 늘리고 week12는 기초 개념들 정리함

# Week 13
- [paper] Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
- https://arxiv.org/abs/1606.09375
- Graph는 다른 데이터에 비해 유클리디안 공간에서 정의되지 않는다. 따라서 이 논문에서는 그래프에 CNN을 적용시키기 위해 다음과 같은 두가지 방법을 제안한다.
1. 그래프의 신호 (Non-Euclidean에서 정의됨)를 유클리디안 공간으로 변환
    - 그래프의 라플라시안 행렬을 푸리에 급수에 따라 고유값 분해 
    - 그래프 행렬을 푸리에 변환(FT)해 유클리디안 공간으로 변환시킨 행렬과 CNN의 필터의 행렬과 element-wise 곱을 통해 CNN 수행한다.
    - 합성곱 된 그래프 결과값을 푸리에 역변환(IFT)를 통해 Non-Euclidean공간으로 다시 바꾼다.
2. CNN의 필터 행렬 변환
    - CNN 필터 행렬을 parameter화 하여 체비셰프 다항식을 이용해 행렬 분해
    - 그래프 행렬과의 Element-wise곱을 통해 CNN 수행
    - 이는 푸리에 급수 행렬이 필요 없기 때문에 계산량이 큰 고유값 분해를 사용할 수 없다.
    - 그래서 계산복잡도가 훨씬 낮아진다.

