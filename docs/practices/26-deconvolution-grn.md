# Deconvolution과 유전자 조절 네트워크 분석 실습

## 26.1 개요

이 장에서는 Bulk RNA-seq 데이터에서 세포 유형 비율을 추정하는 Deconvolution 기법과 유전자 조절 네트워크(Gene Regulatory Network, GRN) 분석을 파이썬으로 구현하는 실습을 다룬다. 차원 축소와 클러스터링 알고리즘의 수학적 원리에 대한 내용은 [11장 차원 축소와 데이터 분석](../theory/11-dimensionality-reduction-and-data-analysis.md)을 참조한다.

## 26.2 Cellular Deconvolution

### 26.2.1 Deconvolution의 개념

Deconvolution은 scRNA-seq 데이터로부터 정의된 세포 유형을 기반으로 Bulk RNA-seq 데이터에 혼합되어 있는 각 세포 유형의 비율을 추정하는 기법이다. Bulk RNA-seq는 조직 전체의 평균 발현량을 측정하므로, 어떤 세포 유형이 얼마나 포함되어 있는지 직접 알 수 없다. Deconvolution을 통해 이 문제를 해결할 수 있다.

### 26.2.2 수학적 모델

Deconvolution 문제는 다음과 같은 행렬곱으로 표현할 수 있다:

```
M = W × H
```

각 행렬의 의미는 다음과 같다:

| 행렬 | 차원 | 설명 |
|---|---|---|
| M | (Sample) × (Gene) | Bulk RNA-seq 발현 행렬 |
| W | (Sample) × (Cell type) | 각 샘플에서 세포 유형별 비율 행렬 |
| H | (Cell type) × (Gene) | 세포 유형별 유전자 발현 시그니처 행렬 |

### 26.2.3 해결 방법

이 문제는 Matrix Factorization 또는 Regression 문제로 해결할 수 있다:

| 방법 | 설명 |
|---|---|
| Non-negative Matrix Factorization (NMF) | 음이 아닌 값으로 행렬을 분해 |
| Generalized Linear Model (GLM) | 선형 회귀 기반 추정 |
| Support Vector Regression (SVR) | SVM 기반 회귀 |

### 26.2.4 CIBERSORT

CIBERSORT는 Support Vector Machine의 한 형태인 Nu-SVM을 사용하여 deconvolution 문제를 해결하는 대표적인 도구이다. 2015년 Nature Methods에 발표된 이후 1만 회 이상 인용되었다.

참고: https://cibersortx.stanford.edu/

## 26.3 유전자 조절 네트워크 (GRN)

### 26.3.1 GRN의 개념

유전자 조절 네트워크는 유전자들 간의 발현 조절 관계를 네트워크로 표현한 것이다. 한 유전자의 발현이 다른 유전자의 발현에 영향을 줄 때, 이를 화살표(활성화) 또는 막대(억제)로 표시한다.

### 26.3.2 GRN과 허브 유전자

네트워크에서 중요한 유전자일수록 GRN의 "허브(hub)"일 확률이 높다. 허브 유전자는 많은 다른 유전자와 연결되어 있어 세포 기능에 핵심적인 역할을 한다. 그래프 이론에서는 이러한 중요도를 Centrality로 측정한다:

| Centrality 유형 | 설명 |
|---|---|
| Degree Centrality | 연결된 노드의 수 |
| Betweenness Centrality | 최단 경로에 포함되는 빈도 |
| PageRank | 중요한 노드로부터의 연결 가중치 |

### 26.3.3 GRN 분석 도구

전사체 데이터로부터 GRN을 생성하고 허브 유전자를 찾는 대표적인 알고리즘은 다음과 같다:

| 데이터 유형 | 도구 |
|---|---|
| Bulk RNA-seq | WGCNA |
| scRNA-seq | SCENIC, TENET |

## 26.4 WGCNA

### 26.4.1 WGCNA 개요

WGCNA(Weighted Gene Co-expression Network Analysis)는 유전자 발현 사이의 연관성 분석을 통해 공동 발현 네트워크를 생성하는 방법이다. 2008년 BMC Bioinformatics에 발표된 이후 20,000회 이상 인용되었다.

### 26.4.2 인접 행렬 계산

WGCNA의 핵심은 유전자 쌍 간의 상관계수를 거듭제곱하여 인접 행렬을 구성하는 것이다:

```
a_ij = |cor(i, j)|^β
```

여기서 β는 soft-thresholding power로, 높은 상관관계는 강조하고 낮은 상관관계는 억제하는 역할을 한다.

### 26.4.3 TOM (Topological Overlap Matrix)

TOM은 두 유전자가 공유하는 이웃의 정도를 측정한다:

```
TOM_ij = (Σ_u a_iu × a_uj + a_ij) / (min(k_i, k_j) + 1 - a_ij)
```

여기서 k_i는 유전자 i의 연결성(connectivity)이다. DistTOM = 1 - TOM으로 거리 행렬을 계산하여 계층적 클러스터링에 사용한다.

## 26.5 실습 환경 구성

### 26.5.1 작업 디렉토리 생성

```bash
$ mkdir -p ~/week11
$ cd ~/week11
```

### 26.5.2 UV 가상환경 설정

```bash
$ uv venv --python 3.13
$ source .venv/bin/activate
$ uv pip install scanpy statsmodels scikit-learn networkx ipykernel seaborn matplotlib
```

### 26.5.3 Jupyter 커널 등록

```bash
$ python -m ipykernel install --user --name week11 --display-name "week11"
```

### 26.5.4 데이터 파일 준비

실습에 사용할 데이터 파일을 심볼릭 링크한다.

```bash
$ ln -s /bce/lectures/2025-bioinformatics/data/deconvolution/count-data-diaphragm-annotated.h5ad .
```

## 26.6 Deconvolution 실습

### 26.6.1 데이터 로드 및 전처리

```python
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

adata = sc.read_h5ad('count-data-diaphragm-annotated.h5ad')

# 원본 데이터 보존
adata.layers['raw'] = adata.X.copy()

# 전처리
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# 시각화
sc.pl.umap(adata, color='cell_type')
```

### 26.6.2 시그니처 행렬 생성

세포 유형별 유전자 발현 시그니처 행렬(H)을 생성한다.

```python
# 각 세포 유형별 유전자 발현 합계 계산
cell_types = adata.obs['cell_type'].cat.categories
sig = np.zeros((len(cell_types), adata.shape[1]))

for i, ct in enumerate(cell_types):
    idx = adata.obs['cell_type'] == ct
    sig[i] = adata.layers['raw'][idx].sum(axis=0)
    sig[i] /= sig[i].max()
```

### 26.6.3 Bulk 발현 데이터 시뮬레이션

모든 세포의 발현량을 합하여 가상의 bulk 데이터를 생성한다.

```python
v = np.asarray(adata.layers['raw'].sum(axis=0)).ravel()
v = v / v.max()
```

### 26.6.4 GLM을 이용한 Deconvolution

Generalized Linear Model을 사용하여 세포 유형 비율을 추정한다.

```python
import statsmodels.api as sm

X = sig.T  # (Gene) x (Cell type)
y = v      # (Gene,)

model = sm.GLM(y, X)
result = model.fit()
print(result.summary())
```

### 26.6.5 Nu-SVR을 이용한 Deconvolution (CIBERSORT 방식)

CIBERSORT에서 사용하는 Nu-SVR 방식으로 deconvolution을 수행한다.

```python
from sklearn.svm import NuSVR

X = sig.T
y = v

# CIBERSORT는 nu 값을 [0.25, 0.5, 0.75] 중 자동 선택
clf = NuSVR(nu=0.25, kernel='linear')
clf.fit(X, y)
```

### 26.6.6 결과 비교

GLM과 CIBERSORT 결과를 실제 세포 비율과 비교한다.

```python
# 실제 세포 비율 계산
true_proportions = adata.obs['cell_type'].value_counts(normalize=True)
true_proportions = true_proportions[cell_types].values

# 결과 정규화
glm_result = result.params / result.params.max()
svr_result = clf.coef_.ravel()
svr_result = svr_result / svr_result.max()

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.barplot(x=list(cell_types), y=glm_result, ax=axes[0])
axes[0].set_title('GLM')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x=list(cell_types), y=svr_result, ax=axes[1])
axes[1].set_title('CIBERSORT')
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(x=list(cell_types), y=true_proportions, ax=axes[2])
axes[2].set_title('True')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
```

### 26.6.7 다중 샘플 시뮬레이션

랜덤하게 세포를 선택하여 여러 샘플을 시뮬레이션한다.

```python
samples = 100
np.random.seed(42)

sums = np.zeros((samples, adata.layers['raw'].shape[1]))
for i in range(samples):
    idx = np.random.choice(adata.layers['raw'].shape[0], 1000)
    sums[i] = np.asarray(adata.layers['raw'][idx].sum(axis=0)).ravel()
```

### 26.6.8 NMF를 이용한 Reference-free Deconvolution

세포 유형 시그니처 없이 NMF만으로 deconvolution을 수행한다.

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=6, random_state=42, max_iter=10000)
W = nmf.fit_transform(sums)  # 비율 행렬
H = nmf.components_          # 시그니처 행렬
```

### 26.6.9 NMF 결과 검증

추정된 시그니처가 실제 세포 유형과 얼마나 일치하는지 확인한다.

```python
from scipy.spatial.distance import cdist

# 코사인 거리로 가장 가까운 세포 유형 찾기
dist = cdist(H, sig, metric='cosine')
closest = np.argmin(dist, axis=1)

nmf_labels = [f"{i} (closest: {lbl})" for i, lbl in enumerate(cell_types[closest])]

# Heatmap 시각화
sns.heatmap(W, xticklabels=nmf_labels)
plt.xlabel('NMF Component')
plt.ylabel('Sample')
```

## 26.7 GRN 분석 실습

### 26.7.1 데이터 준비

특정 세포 유형만 선택하여 GRN 분석을 수행한다.

```python
adata = sc.read_h5ad('count-data-diaphragm-annotated.h5ad')
adata_endo = adata[adata.obs['cell_type'] == 'endothelial cell']

# 필터링
sc.pp.filter_genes(adata_endo, min_cells=3)
sc.pp.filter_cells(adata_endo, min_genes=200)

# 전처리
sc.pp.log1p(adata_endo)
sc.pp.highly_variable_genes(adata_endo, n_top_genes=500)

# HVG만 선택
adata_endo = adata_endo[:, adata_endo.var.highly_variable]
```

### 26.7.2 상관 행렬 및 인접 행렬 계산

WGCNA의 핵심인 가중 상관 행렬을 계산한다.

```python
beta = 3

# 유전자 간 상관 행렬 계산
correlation_matrix = np.corrcoef(adata_endo.X.T.todense())

# 양의 상관관계만 사용 (signed network)
connectivity_matrix = correlation_matrix > 0

# WGCNA 인접 행렬
correlation_matrix = np.abs(correlation_matrix) ** beta
```

### 26.7.3 TOM 계산

Topological Overlap Matrix를 계산한다.

```python
k = np.sum(correlation_matrix, axis=1)
tom = np.zeros_like(correlation_matrix)
n = correlation_matrix.shape[0]

for i in range(n):
    for j in range(i+1, n):
        shared = np.sum(correlation_matrix[i, :] * correlation_matrix[j, :])
        tom[i,j] = (shared + correlation_matrix[i,j]) / (min(k[i], k[j]) + 1 - correlation_matrix[i,j])
        tom[j,i] = tom[i,j]

disttom = 1 - tom
```

### 26.7.4 계층적 클러스터링

DistTOM을 사용하여 유전자를 모듈로 클러스터링한다.

```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(linkage='average', n_clusters=30).fit(disttom)
```

### 26.7.5 네트워크 시각화

NetworkX를 사용하여 특정 모듈의 네트워크를 시각화한다.

```python
import networkx as nx

# 모듈 0의 유전자 선택
nodes = np.where(clustering.labels_ == 0)[0]
nodes = np.sort(nodes)

# 연결 행렬에서 해당 노드만 추출
connectivity_matrix_0 = connectivity_matrix[nodes, :][:, nodes]
G = nx.Graph(connectivity_matrix_0)

# 자기 연결 제거
G.remove_edges_from(nx.selfloop_edges(G))

# 유전자 이름으로 라벨 변경
G = nx.relabel_nodes(G, dict(enumerate(adata_endo.var.index[nodes])))
```

### 26.7.6 Centrality 기반 시각화

Degree centrality를 계산하여 허브 유전자를 시각화한다.

```python
centrality = nx.degree_centrality(G)
pos = nx.spring_layout(G, weight='weight')

plt.figure(figsize=(20, 20))
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=1000,
                       node_color=[centrality[node] for node in G.nodes],
                       cmap='coolwarm')
nx.draw_networkx_labels(G, pos, font_size=8)
plt.show()
```

## 26.8 SCENIC과 TENET

### 26.8.1 SCENIC

SCENIC은 scRNA-seq 데이터를 위한 GRN 분석 도구로, WGCNA를 단일세포 데이터에 맞게 개선한 방법이다. 주요 단계는 다음과 같다:

1. **Co-expression 분석**: GENIE3/GRNBoost를 사용하여 공동 발현 모듈 발견
2. **Motif 분석**: RcisTarget으로 전사인자 결합 모티프 확인
3. **Cell scoring**: AUCell로 각 세포에서 regulon 활성도 계산
4. **클러스터링**: regulon 활성도 기반 세포 상태 분류

참고: https://www.nature.com/articles/nmeth.4463

### 26.8.2 TENET

TENET은 pseudotime 기반으로 세포를 정렬한 후 transfer entropy를 계산하여 유전자 간 인과관계를 추론하는 방법이다.

참고: https://academic.oup.com/nar/article/49/1/e1/5973444

## 26.9 실습 과제

### 실습 26.1: Deconvolution 분석

1. 본문에서 제시된 코드를 사용하여 deconvolution 실습을 수행한다.
2. GLM, Nu-SVR, NMF 세 가지 방법의 결과를 비교한다.
3. 결과를 bar plot으로 시각화하고 실제 세포 비율과 비교한다.

### 실습 26.2: WGCNA 기반 GRN 분석

1. endothelial cell 데이터에서 WGCNA 분석을 수행한다.
2. TOM 행렬을 계산하고 계층적 클러스터링으로 모듈을 찾는다.
3. 모듈 0의 네트워크를 시각화하고 허브 유전자를 확인한다.

### 실습 26.3: 다른 세포 유형 분석

1. satellite cell 또는 stromal cell에 대해 동일한 GRN 분석을 수행한다.
2. endothelial cell과 다른 점이 있는지 비교한다.
