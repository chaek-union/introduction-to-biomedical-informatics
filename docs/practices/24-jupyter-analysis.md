# 주피터 노트북 기반 차등 발현 분석

## 개요

이 장에서는 VS Code의 주피터 노트북 환경에서 PyDESeq2를 사용하여 차등 발현 분석을 수행하는 방법을 다룬다. 차등 발현 분석의 이론적 배경과 통계적 원리에 대한 내용은 [9장 전사체학 기초](../theory/9-basic-transcriptomics.md)를 참조한다.

## 실습 환경 구성

### VS Code 주피터 노트북 설정

VS Code에서 주피터 노트북을 사용하려면 Jupyter 확장을 설치해야 한다.

1. VS Code 왼쪽 사이드바에서 Extensions 아이콘을 클릭한다.
2. 검색창에 "Jupyter"를 입력한다.
3. Microsoft에서 제공하는 Jupyter 확장을 설치한다.

### Python 패키지 설치

차등 발현 분석에 필요한 패키지를 설치한다.

```bash
$ conda activate bioinfo
$ pip install pydeseq2
$ conda install pandas numpy matplotlib seaborn
```

PyDESeq2는 R의 DESeq2를 Python으로 구현한 패키지로, RNA-seq 데이터의 차등 발현 분석에 사용된다.

### 작업 디렉토리 생성

```bash
$ mkdir -p ~/week6/notebooks
$ cd ~/week6
```

### 노트북 파일 생성

VS Code에서 새 파일을 생성하고 `.ipynb` 확장자로 저장한다.

1. File → New File 선택
2. 파일명을 `deseq2_analysis.ipynb`로 저장
3. 커널을 bioinfo conda 환경으로 선택

## 데이터 준비

### 카운트 행렬 로드

STAR에서 생성된 ReadsPerGene.out.tab 파일들을 읽어 카운트 행렬을 생성한다.

```python
import pandas as pd
import numpy as np
from pathlib import Path

# 샘플 목록 정의
samples = ["SRR4420293", "SRR4420294", "SRR4420295",
           "SRR4420296", "SRR4420297", "SRR4420298"]

# 카운트 데이터 로드
counts_dict = {}
for sample in samples:
    filepath = f"star_output/{sample}/ReadsPerGene.out.tab"
    df = pd.read_csv(filepath, sep="\t", header=None, skiprows=4,
                     names=["gene_id", "unstranded", "forward", "reverse"])
    # Unstranded 열 사용 (라이브러리 유형에 따라 조정)
    counts_dict[sample] = df.set_index("gene_id")["unstranded"]

# 카운트 행렬 생성
counts_df = pd.DataFrame(counts_dict)
print(counts_df.head())
```

출력 예시:

```
                SRR4420293  SRR4420294  SRR4420295  SRR4420296  SRR4420297  SRR4420298
gene_id
gene-AT1G01010        1234        1156        1289         987        1023        1045
gene-AT1G01020         567         534         589         423         445         467
gene-AT1G01030         234         256         245         312         298         305
gene-AT1G01040         789         812         756         654         678         689
gene-AT1G01050          45          52          48          67          71          65
```

### 메타데이터 생성

샘플의 실험 조건 정보를 담은 메타데이터 DataFrame을 생성한다.

```python
# 메타데이터 정의
metadata = pd.DataFrame({
    "sample": samples,
    "condition": ["WT", "WT", "WT", "atrx1", "atrx1", "atrx1"]
})
metadata = metadata.set_index("sample")
print(metadata)
```

출력 예시:

```
           condition
sample
SRR4420293        WT
SRR4420294        WT
SRR4420295        WT
SRR4420296     atrx1
SRR4420297     atrx1
SRR4420298     atrx1
```

### 데이터 필터링

발현량이 너무 낮은 유전자는 분석에서 제외한다.

```python
# 최소 발현량 필터링: 모든 샘플에서 총 10개 이상의 리드가 있는 유전자만 유지
min_counts = 10
genes_to_keep = counts_df.sum(axis=1) >= min_counts
counts_filtered = counts_df[genes_to_keep]

print(f"필터링 전 유전자 수: {len(counts_df)}")
print(f"필터링 후 유전자 수: {len(counts_filtered)}")
```

## PyDESeq2를 이용한 차등 발현 분석

### DESeq2 객체 생성

PyDESeq2의 DeseqDataSet 객체를 생성한다.

```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# DESeq2 데이터셋 생성
dds = DeseqDataSet(
    counts=counts_filtered.T,  # 샘플이 행, 유전자가 열
    metadata=metadata,
    design="~condition"  # 실험 설계 공식
)
```

design 파라미터는 R의 formula 문법을 따르며, `~condition`은 condition 열을 기준으로 그룹 간 차이를 분석한다는 의미이다.

### 정규화 및 분산 추정

DESeq2 분석의 핵심 단계를 수행한다.

```python
# 크기 인자 계산 및 분산 추정
dds.fit()
```

fit() 메서드는 내부적으로 다음 단계를 수행한다:

1. 크기 인자(size factor) 계산: 라이브러리 크기 차이 보정
2. 분산 추정: 유전자별 분산 추정 및 shrinkage 적용
3. 음이항 분포 모델 피팅

### 통계 검정

그룹 간 차등 발현을 검정한다.

```python
# 통계 검정 수행
stat_res = DeseqStats(dds, contrast=["condition", "atrx1", "WT"])
stat_res.summary()

# 결과 추출
results_df = stat_res.results_df
print(results_df.head())
```

contrast 파라미터는 비교할 그룹을 지정한다. `["condition", "atrx1", "WT"]`는 atrx1 대비 WT의 차이를 분석한다는 의미이다.

출력 예시:

```
               baseMean  log2FoldChange     lfcSE      stat    pvalue      padj
gene_id
gene-AT1G01010   1122.33           0.234     0.089     2.629    0.0086    0.0342
gene-AT1G01020    504.17          -0.156     0.112    -1.393    0.1636    0.4521
gene-AT1G01030    275.00           0.312     0.145     2.152    0.0314    0.0987
gene-AT1G01040    729.67          -0.089     0.078    -1.141    0.2538    0.5834
gene-AT1G01050     58.00           0.523     0.234     2.235    0.0254    0.0856
```

결과 DataFrame의 주요 열:

| 열 | 설명 |
|---|---|
| baseMean | 모든 샘플에서의 평균 정규화 카운트 |
| log2FoldChange | log2 배수 변화 (양수: 상향 조절, 음수: 하향 조절) |
| lfcSE | log2FoldChange의 표준 오차 |
| stat | Wald 검정 통계량 |
| pvalue | 원시 p-value |
| padj | 다중 검정 보정된 p-value (Benjamini-Hochberg) |

## 결과 분석 및 시각화

### 유의한 유전자 필터링

통계적으로 유의한 차등 발현 유전자를 추출한다.

```python
# 유의성 기준 설정
padj_threshold = 0.05
lfc_threshold = 1.0  # |log2FC| > 1

# 유의한 유전자 필터링
significant = results_df[
    (results_df["padj"] < padj_threshold) &
    (abs(results_df["log2FoldChange"]) > lfc_threshold)
]

print(f"유의한 차등 발현 유전자 수: {len(significant)}")
print(f"  - 상향 조절: {sum(significant['log2FoldChange'] > 0)}")
print(f"  - 하향 조절: {sum(significant['log2FoldChange'] < 0)}")
```

### 화산 그림 (Volcano Plot)

화산 그림은 차등 발현 분석 결과를 시각화하는 대표적인 방법이다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
results_plot = results_df.dropna()
x = results_plot["log2FoldChange"]
y = -np.log10(results_plot["padj"])

# 색상 지정
colors = []
for idx, row in results_plot.iterrows():
    if row["padj"] < padj_threshold and abs(row["log2FoldChange"]) > lfc_threshold:
        if row["log2FoldChange"] > 0:
            colors.append("red")  # 상향 조절
        else:
            colors.append("blue")  # 하향 조절
    else:
        colors.append("gray")  # 비유의

# 그래프 생성
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=colors, alpha=0.5, s=10)

# 임계선 추가
plt.axhline(y=-np.log10(padj_threshold), color="black", linestyle="--", linewidth=0.5)
plt.axvline(x=lfc_threshold, color="black", linestyle="--", linewidth=0.5)
plt.axvline(x=-lfc_threshold, color="black", linestyle="--", linewidth=0.5)

# 레이블
plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(adjusted p-value)")
plt.title("Volcano Plot: atrx1 vs WT")

plt.tight_layout()
plt.savefig("volcano_plot.png", dpi=150)
plt.show()
```

화산 그림에서 x축은 발현량 변화의 크기(log2FC)를, y축은 통계적 유의성(-log10 p-value)을 나타낸다. 오른쪽 상단의 점들은 유의하게 상향 조절된 유전자를, 왼쪽 상단의 점들은 유의하게 하향 조절된 유전자를 나타낸다.

### MA Plot

MA plot은 발현량과 변화량의 관계를 보여준다.

```python
plt.figure(figsize=(10, 8))

# MA plot 생성
x = np.log10(results_plot["baseMean"] + 1)
y = results_plot["log2FoldChange"]

plt.scatter(x, y, c=colors, alpha=0.5, s=10)
plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

plt.xlabel("log10(mean expression)")
plt.ylabel("log2 Fold Change")
plt.title("MA Plot: atrx1 vs WT")

plt.tight_layout()
plt.savefig("ma_plot.png", dpi=150)
plt.show()
```

### 히트맵

유의한 유전자들의 발현 패턴을 히트맵으로 시각화한다.

```python
import seaborn as sns

# 상위 50개 유의한 유전자 선택
top_genes = significant.nlargest(50, "baseMean").index

# 정규화된 카운트 추출 (log2 변환)
normalized_counts = np.log2(counts_filtered.loc[top_genes] + 1)

# Z-score 정규화 (행 기준)
z_scores = (normalized_counts.T - normalized_counts.T.mean()) / normalized_counts.T.std()
z_scores = z_scores.T

# 히트맵 생성
plt.figure(figsize=(10, 12))
sns.heatmap(z_scores, cmap="RdBu_r", center=0,
            xticklabels=True, yticklabels=True)
plt.title("Expression Heatmap of Top 50 DEGs")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.show()
```

## 결과 저장

### 전체 결과 저장

분석 결과를 CSV 파일로 저장한다.

```python
# 전체 결과 저장
results_df.to_csv("deseq2_results_all.csv")

# 유의한 유전자만 저장
significant.to_csv("deseq2_results_significant.csv")

print("결과가 저장되었습니다.")
```

### 유전자 목록 추출

후속 분석(GO 분석, KEGG pathway 분석 등)을 위해 유전자 목록을 추출한다.

```python
# 상향 조절 유전자 목록
upregulated = significant[significant["log2FoldChange"] > 0].index.tolist()
with open("upregulated_genes.txt", "w") as f:
    f.write("\n".join(upregulated))

# 하향 조절 유전자 목록
downregulated = significant[significant["log2FoldChange"] < 0].index.tolist()
with open("downregulated_genes.txt", "w") as f:
    f.write("\n".join(downregulated))

print(f"상향 조절 유전자: {len(upregulated)}개")
print(f"하향 조절 유전자: {len(downregulated)}개")
```

## 전체 분석 코드

다음은 전체 분석 과정을 하나의 스크립트로 정리한 것이다.

```python
# 필요한 패키지 로드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 1. 데이터 로드
samples = ["SRR4420293", "SRR4420294", "SRR4420295",
           "SRR4420296", "SRR4420297", "SRR4420298"]

counts_dict = {}
for sample in samples:
    filepath = f"star_output/{sample}/ReadsPerGene.out.tab"
    df = pd.read_csv(filepath, sep="\t", header=None, skiprows=4,
                     names=["gene_id", "unstranded", "forward", "reverse"])
    counts_dict[sample] = df.set_index("gene_id")["unstranded"]

counts_df = pd.DataFrame(counts_dict)

# 2. 메타데이터 생성
metadata = pd.DataFrame({
    "sample": samples,
    "condition": ["WT", "WT", "WT", "atrx1", "atrx1", "atrx1"]
}).set_index("sample")

# 3. 필터링
counts_filtered = counts_df[counts_df.sum(axis=1) >= 10]

# 4. DESeq2 분석
dds = DeseqDataSet(counts=counts_filtered.T, metadata=metadata, design="~condition")
dds.fit()

stat_res = DeseqStats(dds, contrast=["condition", "atrx1", "WT"])
stat_res.summary()
results_df = stat_res.results_df

# 5. 유의한 유전자 추출
significant = results_df[
    (results_df["padj"] < 0.05) &
    (abs(results_df["log2FoldChange"]) > 1)
]

# 6. 결과 저장
results_df.to_csv("deseq2_results_all.csv")
significant.to_csv("deseq2_results_significant.csv")

print(f"분석 완료: {len(significant)}개의 유의한 차등 발현 유전자 발견")
```

## 실습 과제

### 실습 24.1: 환경 설정

1. VS Code에 Jupyter 확장을 설치한다.
2. PyDESeq2와 필요한 패키지들을 설치한다.
3. 새로운 Jupyter 노트북 파일을 생성한다.

### 실습 24.2: 차등 발현 분석

1. 23장에서 생성한 STAR 출력 파일들을 로드하여 카운트 행렬을 생성한다.
2. 메타데이터를 작성하고 PyDESeq2로 분석을 수행한다.
3. 유의한 차등 발현 유전자를 추출한다.

### 실습 24.3: 결과 시각화

1. 화산 그림을 생성하고 해석한다.
2. MA plot을 생성하고 발현량에 따른 fold change 분포를 확인한다.
3. 상위 차등 발현 유전자들의 히트맵을 생성한다.

### 실습 24.4: 결과 해석

다음 질문에 답하시오:

1. padj < 0.05이고 |log2FC| > 1인 유전자는 몇 개인가?
2. 가장 유의한 상향 조절 유전자 5개는 무엇인가?
3. 가장 유의한 하향 조절 유전자 5개는 무엇인가?
4. baseMean이 가장 높은 유의한 유전자는 무엇인가?
