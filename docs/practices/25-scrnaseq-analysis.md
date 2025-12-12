# 단일세포 전사체 분석 실습

## 25.1 개요

이 장에서는 단일세포 RNA 시퀀싱(scRNA-seq) 데이터의 분석 파이프라인을 파이썬으로 직접 구현하는 실습을 다룬다. 단일세포 전사체학의 이론적 배경, 기술적 원리, 세포 아틀라스 프로젝트에 대한 내용은 [10장 단일 세포 전사체학](../theory/10-single-cell-transcriptomics.md)을 참조한다. 차원 축소와 클러스터링 알고리즘의 수학적 원리에 대한 상세한 내용은 [11장 차원 축소와 데이터 분석](../theory/11-dimensionality-reduction-and-data-analysis.md)을 참조한다.

## 25.2 분석 파이프라인 개요

단일세포 전사체 데이터 분석은 다음과 같은 단계로 구성된다:

1. **정량화 및 데이터 생성**: Cell x Gene matrix 생성
2. **전처리**: 배치 효과 교정, 정규화, 로그 변환
3. **차원 축소**: PCA를 통한 초기 차원 축소
4. **그래프 구성**: k-최근접 이웃(KNN) 그래프 생성
5. **클러스터링**: Leiden 알고리즘을 이용한 세포 유형 분류
6. **시각화**: UMAP 임베딩을 통한 2차원 시각화
7. **차등 발현 분석**: 클러스터 간 마커 유전자 발견

## 25.3 실습 환경 구성

### 25.3.1 작업 디렉토리 생성

```bash
$ mkdir -p ~/week7
$ cd ~/week7
```

### 25.3.2 UV 가상환경 설정

UV를 사용하여 파이썬 가상환경을 생성하고 필요한 패키지를 설치한다.

```bash
$ uv venv --python 3.13
$ source .venv/bin/activate
$ uv pip install numpy pandas scikit-learn seaborn matplotlib
$ uv pip install umap-learn leidenalg igraph
$ uv pip install ipykernel
```

### 25.3.3 Jupyter 커널 등록

```bash
$ python -m ipykernel install --user --name week7 --display-name "week7"
```

## 25.4 데이터 시뮬레이션

실제 scRNA-seq 데이터를 사용하기 전에, 데이터의 특성을 이해하기 위해 시뮬레이션 데이터를 생성한다.

### 25.4.1 기본 패키지 및 상수 설정

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import nbinom

np.random.seed(42)

n_celltypes = 20
n_cells_per_celltype = 100
n_genes = 500
n_total_cells = n_celltypes * n_cells_per_celltype
```

### 25.4.2 Negative Binomial 기반 카운트 생성

scRNA-seq 데이터는 음이항 분포(Negative Binomial distribution)를 따르는 것으로 알려져 있다. 이를 시뮬레이션하는 함수를 정의한다.

```python
def simulate_counts(mu0, mu1):
    mu = np.random.uniform(mu0, mu1)
    r = np.random.uniform(2, 4)
    p = r / (r + mu)
    return nbinom.rvs(r, p, size=n_cells_per_celltype)
```

### 25.4.3 세포 유형별 발현 데이터 생성

각 세포 유형에 대해 마커 유전자와 배경 유전자를 구분하여 발현량을 생성한다.

```python
total_counts = []
for i in range(n_celltypes):
    n_marker_genes = np.random.randint(5, 50)
    marker_gene_indices = np.random.choice(n_genes, n_marker_genes, replace=False)
    counts = np.zeros((n_cells_per_celltype, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        if gene_idx in marker_gene_indices:
            counts[:, gene_idx] = simulate_counts(10, 1000)
        else:
            counts[:, gene_idx] = simulate_counts(1, 5)
    total_counts.append(counts)

mat = np.vstack(total_counts)
np.random.shuffle(mat)

df = pd.DataFrame(mat,
    columns=[f"Gene_{i}" for i in range(n_genes)],
    index=[f"Cell_{i}" for i in range(n_total_cells)]
)
```

## 25.5 데이터 전처리

### 25.5.1 Mean-Variance 관계 확인

scRNA-seq 데이터의 특징적인 mean-variance 관계를 확인한다.

```python
gene_means = df.mean(axis=0)
gene_vars = df.var(axis=0)

sns.scatterplot(x=gene_means, y=gene_vars, s=2)
plt.xlabel('Mean Expression')
plt.ylabel('Variance of Expression')
plt.ylim(0, None)
```

### 25.5.2 라이브러리 크기 정규화

세포마다 시퀀싱 깊이가 다르므로, 라이브러리 크기로 정규화한다.

```python
cell_summed = df.sum(axis=1).values
df_normalized = df / cell_summed.reshape(-1, 1) * np.median(cell_summed)
```

### 25.5.3 로그 변환

정규화된 데이터에 로그 변환을 적용하여 분산을 안정화한다.

```python
df_log_normalized = np.log1p(df_normalized)
```

로그 변환 전후의 유전자 발현 분포를 비교한다.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['Gene_100'], bins=50, ax=axes[0])
axes[0].set_title('Raw counts')
sns.histplot(df_log_normalized['Gene_100'], bins=50, ax=axes[1])
axes[1].set_title('Log-normalized')
```

## 25.6 차원 축소

### 25.6.1 PCA 수행

scikit-learn의 PCA를 사용하여 차원을 축소한다.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca_result = pca.fit_transform(df_log_normalized)

sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], s=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
```

### 25.6.2 KNN 그래프 생성

UMAP 패키지의 nearest_neighbors 함수를 사용하여 KNN 그래프를 생성한다.

```python
from umap.umap_ import nearest_neighbors

knn = nearest_neighbors(pca_result,
                        n_neighbors=15,
                        metric="euclidean",
                        metric_kwds=None,
                        angular=False,
                        random_state=None)
```

### 25.6.3 UMAP 임베딩

KNN 그래프를 기반으로 UMAP 임베딩을 수행한다.

```python
from umap import UMAP

umap_model = UMAP(n_components=2, min_dist=0.5, spread=1.0, precomputed_knn=knn)
umap_result = umap_model.fit_transform(pca_result)

sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], s=1)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
```

## 25.7 클러스터링

### 25.7.1 Leiden 클러스터링을 위한 그래프 구성

KNN 인덱스를 사용하여 igraph 그래프 객체를 생성한다.

```python
import leidenalg
import igraph as ig

indices = knn[0]

edges = []
for i in range(indices.shape[0]):
    for j in range(1, indices.shape[1]):
        edges.append((i, indices[i, j]))

g = ig.Graph(edges=edges, directed=False)
g.simplify()
```

### 25.7.2 Leiden 클러스터링 수행

Leiden 알고리즘을 사용하여 세포를 클러스터링한다.

```python
partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    resolution_parameter=1.0
)

labels = np.array(partition.membership)
```

### 25.7.3 클러스터링 결과 시각화

UMAP 좌표에 클러스터 라벨을 색상으로 표시한다.

```python
sns.scatterplot(
    x=umap_result[:, 0], y=umap_result[:, 1], s=1,
    hue=labels, palette='tab20', legend=None
)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
```

### 25.7.4 클러스터별 Heatmap

클러스터링 결과로 세포를 정렬하여 발현 패턴을 확인한다.

```python
sorted_indices = np.argsort(labels)
df_sorted = df_log_normalized.iloc[sorted_indices, :]

sns.heatmap(df_sorted, cmap='viridis', cbar=False)
```

## 25.8 차등 발현 분석

### 25.8.1 Wilcoxon Rank-Sum 검정

두 클러스터 간의 차등 발현 유전자를 찾기 위해 Wilcoxon rank-sum 검정을 수행한다.

```python
from scipy.stats import ranksums

cluster_0_indices = np.where(labels == 0)[0]
cluster_1_indices = np.where(labels == 1)[0]

deg_genes = []
p_values = []
adjusted_p_values = []
log2fc = []

for gene in df_log_normalized.columns:
    stat, p_value = ranksums(
        df_log_normalized.iloc[cluster_0_indices][gene],
        df_log_normalized.iloc[cluster_1_indices][gene]
    )
    adjusted_p_value = p_value * n_genes  # Bonferroni correction
    if adjusted_p_value < 0.01:
        deg_genes.append(gene)
        p_values.append(p_value)
        adjusted_p_values.append(adjusted_p_value)
        log2fc.append(
            np.log2(df_log_normalized.iloc[cluster_0_indices][gene].mean() + 1) -
            np.log2(df_log_normalized.iloc[cluster_1_indices][gene].mean() + 1)
        )
```

### 25.8.2 Volcano Plot

차등 발현 분석 결과를 volcano plot으로 시각화한다.

```python
sns.scatterplot(x=log2fc, y=-np.log10(adjusted_p_values), s=5)
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 Adjusted P-Value')
plt.axhline(y=2, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
```

### 25.8.3 상위 DEG Heatmap

상위 차등 발현 유전자의 발현 패턴을 heatmap으로 확인한다.

```python
top_deg_indices = np.argsort(np.abs(log2fc))[-50:]
top_deg_genes = [deg_genes[i] for i in top_deg_indices]

df_top_degs = df_sorted[top_deg_genes].iloc[:200]
sns.heatmap(df_top_degs, cmap='viridis', cbar=True)
```

## 25.9 Scanpy를 이용한 분석

Scanpy는 단일세포 데이터 분석을 위한 파이썬 패키지로, 앞서 다룬 모든 분석 단계를 통합적으로 제공한다.

### 25.9.1 Scanpy 설치

```bash
$ uv pip install scanpy
```

### 25.9.2 데이터 로드

h5ad 형식의 데이터를 로드한다.

```python
import scanpy as sc

adata = sc.read_h5ad("data.h5ad")
```

### 25.9.3 품질 관리

```python
# 미토콘드리아 유전자 비율 계산
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# QC plot
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4)
```

### 25.9.4 전처리 및 차원 축소

```python
# 필터링
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 정규화
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 고변이 유전자 선택
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]

# 스케일링
sc.pp.scale(adata, max_value=10)

# PCA
sc.tl.pca(adata, svd_solver='arpack')
```

### 25.9.5 클러스터링 및 시각화

```python
# 이웃 그래프 구성
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# UMAP
sc.tl.umap(adata)

# Leiden 클러스터링
sc.tl.leiden(adata, resolution=1.5)

# 시각화
sc.pl.umap(adata, color=['leiden'])
```

### 25.9.6 마커 유전자 발견

```python
# 클러스터별 마커 유전자 찾기
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Dot plot
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)
```

## 25.10 실습 과제

### 실습 25.1: 시뮬레이션 데이터 분석

1. 본문에서 제시된 코드를 사용하여 시뮬레이션 데이터를 생성한다.
2. 정규화, PCA, UMAP, Leiden 클러스터링을 순차적으로 수행한다.
3. resolution 파라미터를 0.5, 1.0, 2.0으로 변경하면서 클러스터 수의 변화를 관찰한다.

### 실습 25.2: Scanpy 분석 파이프라인

1. UV로 새로운 가상환경을 생성하고 Scanpy를 설치한다.
2. 제공된 h5ad 파일을 로드한다.
   - 데이터 경로: `/bce/lectures/2025-bioinformatics/data/scrnaseq/brain_small.h5ad`
3. Scanpy 튜토리얼을 참조하여 다음을 수행한다:
   - 기본 QC plot 및 고변이 유전자(HVG) 선택 plot
   - Leiden 클러스터링 결과(resolution 1.5)로 색이 표현된 PCA 및 UMAP
   - 각 클러스터별 DEG 5개씩 포함된 dot plot

참고 튜토리얼: https://scverse-tutorials.readthedocs.io/en/latest/notebooks/basic-scrna-tutorial.html
