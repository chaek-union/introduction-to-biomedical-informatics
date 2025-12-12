# 전사체 분석 파이프라인 구축

## 개요

이 장에서는 Snakemake를 활용하여 전사체 분석 파이프라인을 구축하는 방법을 다룬다. 전사체 분석의 이론적 배경, RNA-seq의 원리, 정량화 및 차등 발현 분석에 대한 내용은 [9장 전사체학 기초](../theory/9-basic-transcriptomics.md)를 참조한다.

## Snakemake 개요

### Snakemake란

Snakemake는 파이썬 기반의 워크플로우 관리 도구로, 생명정보학 분석에서 재현 가능한 파이프라인을 구축하는 데 널리 사용된다. Snakemake의 주요 특징은 다음과 같다:

- 파이썬 문법을 기반으로 하여 사용법이 간단하다.
- 여러 도구를 순차적 또는 병렬로 실행할 수 있다.
- 파일 의존성을 자동으로 추적하여 필요한 작업만 수행한다.
- 쉘 명령뿐만 아니라 파이썬 코드나 Jupyter 노트북도 실행할 수 있다.

### Snakemake 설치

conda를 사용하여 Snakemake를 설치한다.

```bash
$ conda activate bioinfo
$ conda install snakemake
```

### 기본 문법

Snakemake 워크플로우는 `Snakefile`에 정의되며, 각 작업은 `rule`로 표현된다.

```python
rule 규칙명:
    input:
        "인풋 파일명"
    output:
        "아웃풋 파일명"
    shell:
        "쉘 명령"
```

실제 예시로 BWA-MEM을 이용한 매핑 규칙을 살펴보면:

```python
rule bwa_map:
    input:
        "data/genome.fa",
        "data/samples/A.fastq"
    output:
        "mapped_reads/A.bam"
    shell:
        "bwa mem {input} | samtools view -Sb - > {output}"
```

위 규칙을 실행하면 `bwa mem` 명령이 수행되어 `mapped_reads/A.bam` 파일이 생성된다.

### 와일드카드 사용

Snakemake에서는 와일드카드를 사용하여 여러 샘플에 대해 동일한 규칙을 적용할 수 있다.

```python
rule bwa_map:
    input:
        "data/genome.fa",
        "data/samples/{sample}.fastq"
    output:
        "mapped_reads/{sample}.bam"
    shell:
        "bwa mem {input} | samtools view -Sb - > {output}"
```

다음과 같이 실행하면 A와 B 샘플 모두에 대해 매핑이 수행된다:

```bash
$ snakemake --cores 1 mapped_reads/A.bam mapped_reads/B.bam
```

### 실행 블록 유형

Snakemake는 세 가지 실행 블록을 지원한다:

**shell 블록**: 쉘 명령을 실행한다.
```python
rule example_shell:
    input: "input.txt"
    output: "output.txt"
    shell: "cat {input} > {output}"
```

**script 블록**: 외부 파이썬 스크립트를 실행한다.
```python
rule example_script:
    input: "input.txt"
    output: "output.txt"
    script: "scripts/process.py"
```

**run 블록**: 인라인 파이썬 코드를 실행한다.
```python
rule example_run:
    input: "input.txt"
    output: "output.txt"
    run:
        with open(input[0]) as f:
            data = f.read()
        with open(output[0], 'w') as f:
            f.write(data.upper())
```

### 규칙 연결

여러 규칙을 정의하면 Snakemake가 파일 의존성을 자동으로 추적하여 순차적으로 실행한다.

```python
rule rule1:
    input: "파일1"
    output: "파일2"
    shell: "명령1"

rule rule2:
    input: "파일2"
    output: "파일3"
    shell: "명령2"
```

`파일3`을 생성하려면 먼저 `파일2`가 필요하므로, Snakemake는 `rule1`을 먼저 실행한 후 `rule2`를 실행한다:

```bash
$ snakemake --cores 1 파일3
```

## RNA-seq 분석 파이프라인

### 파이프라인 구조

전사체 분석 파이프라인은 다음과 같은 단계로 구성된다:

1. **품질 관리 (FastQC)**: 원시 FASTQ 파일의 품질을 평가한다.
2. **어댑터 제거 (Cutadapt)**: 시퀀싱 어댑터와 poly-A 꼬리를 제거한다.
3. **정렬 (STAR)**: 스플라이스 인식 정렬을 수행한다.
4. **정량화**: 유전자별 리드 카운트를 계산한다.

### 실습 환경 구성

작업 디렉토리를 생성하고 필요한 도구를 설치한다.

```bash
$ mkdir -p ~/week6
$ cd ~/week6
$ conda activate bioinfo
$ conda install star cutadapt fastqc
```

### STAR 인덱스 생성

STAR 정렬을 위해서는 참조 유전체 인덱스가 필요하다. 인덱스 생성에는 참조 유전체 FASTA 파일과 유전자 주석 GTF 파일이 필요하다.

```bash
STAR --runThreadN 4 \
     --runMode genomeGenerate \
     --genomeDir star_index \
     --genomeFastaFiles /path/to/reference.fa \
     --sjdbGTFfile /path/to/annotation.gtf
```

주요 옵션 설명:

| 옵션 | 설명 |
|---|---|
| --runThreadN | 사용할 CPU 코어 수 |
| --runMode | 실행 모드 (genomeGenerate: 인덱스 생성) |
| --genomeDir | 인덱스가 저장될 디렉토리 |
| --genomeFastaFiles | 참조 유전체 FASTA 파일 |
| --sjdbGTFfile | 유전자 주석 GTF 파일 |

본 실습에서는 애기장대(Arabidopsis thaliana) 참조 유전체를 사용한다:

```bash
$ STAR --runThreadN 4 \
       --runMode genomeGenerate \
       --genomeDir star_index \
       --genomeFastaFiles /bce/omics/references/arabidopsis/GCF_000001735.4/GCF_000001735.4_TAIR10.1_genomic.fna \
       --sjdbGTFfile /bce/omics/references/arabidopsis/GCF_000001735.4/GCF_000001735.4_TAIR10.1_genomic.gtf
```

### Snakefile 작성

전체 파이프라인을 정의하는 Snakefile을 작성한다.

```python
# 샘플 목록 정의
SAMPLES = ["SRR4420293", "SRR4420294", "SRR4420295",
           "SRR4420296", "SRR4420297", "SRR4420298"]

# 참조 유전체 경로
REFERENCE_DIR = "/bce/omics/references/arabidopsis/GCF_000001735.4"
STAR_INDEX = "star_index"

# 기본 규칙: 모든 샘플에 대해 파이프라인 실행
rule all:
    input:
        expand("fastqc/{sample}_1_fastqc.html", sample=SAMPLES),
        expand("fastqc/{sample}_2_fastqc.html", sample=SAMPLES),
        expand("star_output/{sample}/ReadsPerGene.out.tab", sample=SAMPLES)

# FastQC 품질 관리
rule fastqc:
    input:
        r1 = "data/{sample}_1.fastq.gz",
        r2 = "data/{sample}_2.fastq.gz"
    output:
        html1 = "fastqc/{sample}_1_fastqc.html",
        html2 = "fastqc/{sample}_2_fastqc.html",
        zip1 = "fastqc/{sample}_1_fastqc.zip",
        zip2 = "fastqc/{sample}_2_fastqc.zip"
    shell:
        """
        fastqc {input.r1} {input.r2} -o fastqc/
        """

# Cutadapt 어댑터 제거
rule cutadapt:
    input:
        r1 = "data/{sample}_1.fastq.gz",
        r2 = "data/{sample}_2.fastq.gz"
    output:
        r1 = "trimmed/{sample}_1.trimmed.fastq.gz",
        r2 = "trimmed/{sample}_2.trimmed.fastq.gz"
    params:
        adapter = "AGATCGGAAGAGC",  # Illumina Universal Adapter
        polya = "A" * 20  # poly-A tail
    shell:
        """
        cutadapt -a {params.adapter} -A {params.adapter} \
                 -a {params.polya} -A {params.polya} \
                 -o {output.r1} -p {output.r2} \
                 {input.r1} {input.r2} \
                 --minimum-length 20
        """

# STAR 정렬 및 정량화
rule star_align:
    input:
        r1 = "trimmed/{sample}_1.trimmed.fastq.gz",
        r2 = "trimmed/{sample}_2.trimmed.fastq.gz"
    output:
        bam = "star_output/{sample}/Aligned.sortedByCoord.out.bam",
        counts = "star_output/{sample}/ReadsPerGene.out.tab"
    params:
        index = STAR_INDEX,
        outdir = "star_output/{sample}/"
    threads: 4
    shell:
        """
        STAR --genomeDir {params.index} \
             --runThreadN {threads} \
             --readFilesIn {input.r1} {input.r2} \
             --readFilesCommand zcat \
             --outSAMtype BAM SortedByCoordinate \
             --quantMode GeneCounts \
             --outFileNamePrefix {params.outdir} \
             --outTmpDir /tmp/{wildcards.sample}_star
        """
```

### 파이프라인 실행

파이프라인을 실행하기 전에 dry-run으로 실행 계획을 확인한다:

```bash
$ snakemake --dry-run
```

실제 실행:

```bash
$ snakemake --cores 4
```

특정 파일만 생성하려면:

```bash
$ snakemake --cores 4 star_output/SRR4420293/ReadsPerGene.out.tab
```

### 주요 Snakemake 옵션

| 옵션 | 설명 |
|---|---|
| --dry-run (-n) | 실행하지 않고 계획만 출력 |
| --cores (-j) | 사용할 CPU 코어 수 |
| --forceall (-F) | 모든 규칙을 강제로 재실행 |
| --printshellcmds (-p) | 실행되는 쉘 명령 출력 |
| --dag | 의존성 그래프를 DOT 형식으로 출력 |

## STAR 출력 파일

### ReadsPerGene.out.tab 파일

STAR의 `--quantMode GeneCounts` 옵션을 사용하면 유전자별 리드 카운트가 포함된 파일이 생성된다.

```bash
$ head star_output/SRR4420293/ReadsPerGene.out.tab
```

파일 구조:

| 열 | 설명 |
|---|---|
| 1열 | 유전자 ID |
| 2열 | Unstranded 카운트 |
| 3열 | Forward strand 카운트 |
| 4열 | Reverse strand 카운트 |

파일 처음 4줄은 통계 정보를 포함한다:

| 행 | 설명 |
|---|---|
| N_unmapped | 매핑되지 않은 리드 수 |
| N_multimapping | 다중 매핑 리드 수 |
| N_noFeature | 유전자 영역에 해당하지 않는 리드 수 |
| N_ambiguous | 여러 유전자에 걸친 리드 수 |

### Strandedness 확인

RNA-seq 라이브러리는 방향성(strandedness)에 따라 unstranded, forward stranded, reverse stranded로 구분된다. 실험에 사용된 라이브러리 유형에 맞는 열을 선택해야 한다.

방향성을 확인하려면 2열, 3열, 4열의 합계를 비교한다. 대부분의 카운트가 특정 열에 집중되어 있다면 해당 방향의 stranded 라이브러리이다.

## 고급 Snakemake 기능

### expand 함수

`expand` 함수는 와일드카드를 여러 값으로 확장한다.

```python
expand("data/{sample}/processed.txt", sample=["A", "B", "C"])
# 결과: ["data/A/processed.txt", "data/B/processed.txt", "data/C/processed.txt"]
```

### multiext 함수

`multiext` 함수는 하나의 기본 경로에 여러 확장자를 추가한다.

```python
multiext("results/sample1", ".pdf", ".txt", ".png")
# 결과: ["results/sample1.pdf", "results/sample1.txt", "results/sample1.png"]
```

### 설정 파일 사용

복잡한 파이프라인에서는 설정을 별도의 YAML 파일로 분리할 수 있다.

config.yaml:
```yaml
samples:
  - SRR4420293
  - SRR4420294
  - SRR4420295
reference_dir: /bce/omics/references/arabidopsis/GCF_000001735.4
threads: 4
```

Snakefile에서 설정 파일 사용:
```python
configfile: "config.yaml"

SAMPLES = config["samples"]
REFERENCE_DIR = config["reference_dir"]
```

## 실습 과제

### 실습 23.1: Snakemake 설치 및 기본 실행

1. week6 디렉토리를 생성하고 Snakemake를 설치한다.
2. 간단한 Snakefile을 작성하여 텍스트 파일을 복사하는 규칙을 테스트한다.

### 실습 23.2: RNA-seq 파이프라인 구축

1. 본문에서 제시된 Snakefile을 작성한다.
2. 다음 샘플들에 대해 FastQC, Cutadapt, STAR 정렬을 수행한다:

| 조건 | 반복 1 | 반복 2 | 반복 3 |
|---|---|---|---|
| WT | SRR4420293 | SRR4420294 | SRR4420295 |
| atrx-1 | SRR4420296 | SRR4420297 | SRR4420298 |

3. 생성된 ReadsPerGene.out.tab 파일의 내용을 확인한다.

### 실습 23.3: Strandedness 확인

1. STAR 출력 파일 ReadsPerGene.out.tab의 2열, 3열, 4열 합계를 비교한다.
2. 라이브러리가 stranded인지 unstranded인지 판단한다.
