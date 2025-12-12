# 공간 전사체학 분석 실습

## 28.1 개요

이 장에서는 Squidpy를 사용하여 공간 전사체학 데이터를 분석하는 실습을 다룬다. 공간 전사체학의 기본 원리, 다양한 기술 플랫폼(Visium, MERSCOPE, Xenium 등)에 대한 이론적 내용은 [12장 조직 병리학과 공간체학](../theory/12-pathology-and-spatial-omics.md)을 참조한다.

## 28.2 Squidpy

Squidpy는 공간 오믹스 데이터 분석을 위한 Python 라이브러리이다. Scanpy와 AnnData 생태계와 완전히 통합되어 있으며, 공간 그래프 분석, 이미지 분석, 공간 통계 계산 등의 기능을 제공한다.

Squidpy의 주요 기능은 다음과 같다:

| 기능 | 설명 |
|---|---|
| 공간 그래프 구축 | 세포 간 공간 관계를 그래프로 표현 |
| 공간 통계 | Moran's I, Ripley's L 등 공간 자기상관 분석 |
| 이웃 농축 분석 | 세포 유형 간 공간적 상호작용 분석 |
| 동시 발생 분석 | 특정 거리에서 세포 유형의 동시 발생 확률 계산 |
| 이미지 분석 | 조직 이미지에서 형태학적 특성 추출 |

참고: https://squidpy.readthedocs.io/

## 28.3 실습 환경 구성

### 28.3.1 작업 디렉토리 생성

```bash
$ mkdir -p ~/spatial
$ cd ~/spatial
```

### 28.3.2 UV 가상환경 설정

```bash
$ uv venv --python 3.11
$ source .venv/bin/activate
```

### 28.3.3 필수 라이브러리 설치

```bash
$ uv pip install squidpy scanpy spatialdata spatialdata-io ipykernel
```

### 28.3.4 Jupyter 커널 등록

```bash
$ python -m ipykernel install --user --name spatial --display-name "spatial"
```

## 28.4 10x Genomics Xenium 데이터 분석

### 28.4.1 데이터 로드

Xenium은 10x Genomics의 제자리 시퀀싱(In Situ Sequencing) 기반 플랫폼이다. 이 실습에서는 인간 폐암 데이터셋을 사용한다.

```python
import squidpy as sq
import scanpy as sc
import spatialdata_io
import matplotlib.pyplot as plt

# Xenium 데이터 로드 (SpatialData 형식으로 변환)
sdata = spatialdata_io.xenium(
    "path/to/xenium_output",
    cells_as_shapes=True
)

# AnnData 객체 추출
adata = sdata.tables["table"]
```

### 28.4.2 품질 관리(QC) 지표 계산

```python
# QC 지표 계산
sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

# QC 지표 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(adata.obs["total_counts"], bins=50)
axes[0].set_xlabel("Total counts")
axes[0].set_ylabel("Number of cells")

axes[1].hist(adata.obs["n_genes_by_counts"], bins=50)
axes[1].set_xlabel("Number of genes")
axes[1].set_ylabel("Number of cells")

# 세포 면적 분포 (Xenium에서 제공하는 경우)
if "cell_area" in adata.obs.columns:
    axes[2].hist(adata.obs["cell_area"], bins=50)
    axes[2].set_xlabel("Cell area")
    axes[2].set_ylabel("Number of cells")

plt.tight_layout()
plt.show()
```

### 28.4.3 데이터 전처리

단일세포 전사체 분석과 동일한 전처리 파이프라인을 적용한다. 자세한 내용은 [25장 단일세포 전사체 분석](./25-scrnaseq-analysis.md)을 참조한다.

```python
# 세포 및 유전자 필터링
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=5)

# 정규화 및 로그 변환
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# 고변이 유전자 선택
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# 차원 축소
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)

# 클러스터링
sc.tl.leiden(adata, resolution=0.5)
```

### 28.4.4 공간 시각화

```python
# UMAP 시각화
sc.pl.umap(adata, color="leiden", legend_loc="on data")

# 공간 좌표에서 클러스터 시각화
sq.pl.spatial_scatter(
    adata,
    color="leiden",
    shape=None,
    size=0.5
)
```

### 28.4.5 공간 이웃 그래프 구축

공간 분석을 위해서는 먼저 세포 간 공간 관계를 그래프로 구축해야 한다.

```python
# Delaunay 삼각분할 기반 공간 그래프 구축
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

# 또는 고정 반경 기반 이웃 정의
sq.gr.spatial_neighbors(adata, coord_type="generic", radius=100)
```

### 28.4.6 이웃 농축 분석

이웃 농축 분석(Neighborhood Enrichment)은 특정 세포 유형이 다른 세포 유형 근처에 예상보다 많이 또는 적게 분포하는지를 분석한다.

```python
# 이웃 농축 분석
sq.gr.nhood_enrichment(adata, cluster_key="leiden")

# 결과 시각화
sq.pl.nhood_enrichment(
    adata,
    cluster_key="leiden",
    method="average",
    cmap="coolwarm",
    vmin=-50,
    vmax=50
)
```

결과 히트맵에서 양수 값(빨간색)은 두 클러스터가 공간적으로 함께 분포하는 경향을, 음수 값(파란색)은 서로 떨어져 분포하는 경향을 나타낸다.

### 28.4.7 동시 발생 분석

동시 발생 분석(Co-occurrence Analysis)은 특정 세포 유형이 다른 세포 유형 근처에서 발견될 조건부 확률을 거리에 따라 분석한다.

```python
# 동시 발생 분석
sq.gr.co_occurrence(
    adata,
    cluster_key="leiden",
    spatial_key="spatial",
    interval=[0, 100, 200, 300, 400, 500]
)

# 결과 시각화
sq.pl.co_occurrence(
    adata,
    cluster_key="leiden",
    clusters=["0", "1", "2"]  # 관심 클러스터 선택
)
```

### 28.4.8 중심성 점수 계산

그래프 중심성 지표를 계산하여 각 클러스터의 공간적 특성을 분석한다.

```python
# 중심성 점수 계산
sq.gr.centrality_scores(adata, cluster_key="leiden")

# 결과 시각화
sq.pl.centrality_scores(adata, cluster_key="leiden")
```

### 28.4.9 Moran's I를 이용한 공간 자기상관 분석

Moran's I는 유전자 발현이 공간적으로 클러스터링되어 있는지(양의 자기상관), 무작위로 분포하는지, 또는 분산되어 있는지(음의 자기상관)를 측정하는 통계량이다.

```python
# 공간 자기상관 계산
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    n_perms=100,
    n_jobs=4
)

# 결과 확인
adata.uns["moranI"].head(20)
```

### 28.4.10 Ripley's L 함수

Ripley's L 함수는 특정 클러스터의 점 패턴이 클러스터링되어 있는지, 무작위인지, 분산되어 있는지를 분석한다.

```python
# Ripley's L 계산
sq.gr.ripley(adata, cluster_key="leiden", mode="L")

# 결과 시각화
sq.pl.ripley(adata, cluster_key="leiden", mode="L")
```

## 28.5 Vizgen MERSCOPE 데이터 분석

### 28.5.1 데이터 로드

MERSCOPE는 Vizgen의 MERFISH 기반 플랫폼이다. 이 실습에서는 마우스 뇌 수용체 데이터셋을 사용한다.

```python
import squidpy as sq

# Vizgen 데이터 로드
adata = sq.read.vizgen(
    path="path/to/vizgen_output",
    counts_file="cell_by_gene.csv",
    meta_file="cell_metadata.csv",
    transformation_file="micron_to_mosaic_pixel_transform.csv"
)
```

### 28.5.2 품질 관리

```python
# QC 지표 계산
sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

# 세포 부피 분포 확인 (MERSCOPE에서 제공)
if "volume" in adata.obs.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(adata.obs["volume"], bins=50)
    ax.set_xlabel("Cell volume")
    ax.set_ylabel("Number of cells")
    plt.show()
```

### 28.5.3 전처리 및 클러스터링

```python
# 필터링
sc.pp.filter_cells(adata, min_counts=50)
sc.pp.filter_genes(adata, min_cells=10)

# 정규화
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# 고변이 유전자 선택
sc.pp.highly_variable_genes(adata, n_top_genes=500)

# 차원 축소 및 클러스터링
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8)
```

### 28.5.4 공간 분석

```python
# 공간 그래프 구축
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

# 이웃 농축 분석
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(adata, cluster_key="leiden")

# 동시 발생 분석
sq.gr.co_occurrence(
    adata,
    cluster_key="leiden",
    interval=[0, 50, 100, 150, 200]
)

# Moran's I
sq.gr.spatial_autocorr(adata, mode="moran")

# Ripley's L
sq.gr.ripley(adata, cluster_key="leiden", mode="L")
```

### 28.5.5 특정 유전자의 공간 발현 시각화

```python
# 특정 유전자 발현의 공간 분포
sq.pl.spatial_scatter(
    adata,
    color=["Gad1", "Slc17a7", "Aqp4"],  # 마커 유전자
    shape=None,
    size=0.5
)
```

## 28.6 공간 통계 분석의 해석

### 28.6.1 Moran's I 해석

| Moran's I 값 | 해석 |
|---|---|
| I > 0 | 양의 공간 자기상관 (클러스터링) |
| I ≈ 0 | 무작위 분포 |
| I < 0 | 음의 공간 자기상관 (분산) |

p-value가 유의수준(일반적으로 0.05) 미만일 때 공간 패턴이 통계적으로 유의하다고 판단한다.

### 28.6.2 이웃 농축 z-score 해석

| z-score | 해석 |
|---|---|
| z > 2 | 유의한 양의 농축 (함께 분포) |
| -2 < z < 2 | 무작위 분포 |
| z < -2 | 유의한 음의 농축 (분리되어 분포) |

### 28.6.3 Ripley's L 해석

Ripley's L 함수에서 관측값이 기대값(무작위 분포)보다 높으면 클러스터링을, 낮으면 분산을 나타낸다.

## 28.7 SpatialData 형식

SpatialData는 공간 오믹스 데이터를 위한 표준화된 데이터 형식이다. 다양한 공간 오믹스 플랫폼의 데이터를 통합된 형식으로 저장하고 분석할 수 있다.

### 28.7.1 SpatialData 구조

| 구성요소 | 설명 |
|---|---|
| Tables | AnnData 형식의 유전자 발현 데이터 |
| Points | 개별 전사체의 공간 좌표 |
| Shapes | 세포 경계 등의 다각형 |
| Labels | 세포 분할 마스크 |
| Images | 조직 이미지 |

### 28.7.2 SpatialData 사용

```python
import spatialdata as sd
import spatialdata_io

# Xenium 데이터 로드
sdata = spatialdata_io.xenium("path/to/data")

# 구성요소 확인
print(sdata)

# 테이블 접근
adata = sdata.tables["table"]

# 이미지 접근
images = sdata.images

# 포인트 접근 (전사체 위치)
points = sdata.points
```

## 28.8 napari를 이용한 대화형 시각화

napari는 다차원 이미지 데이터를 대화형으로 시각화하는 도구이다. SpatialData와 통합하여 공간 전사체학 데이터를 탐색할 수 있다.

```python
# napari 설치 (별도 설치 필요)
# uv pip install napari-spatialdata napari[all]

import napari
from napari_spatialdata import Interactive

# napari 뷰어 시작
viewer = napari.Viewer()
Interactive(sdata, viewer)
```