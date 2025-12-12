# 시퀀스 정렬 실습

## 20.1 개요

이 장에서는 NGS 데이터를 참조 유전체에 정렬하는 실습을 다룬다. 정렬 알고리즘의 원리(BWT, FM Index 등)에 대한 이론적 내용은 [3장 차세대 시퀀싱](../theory/3-ngs.md)을 참조한다.

## 20.2 실습 환경 구성

conda를 사용하여 BWA-MEM2를 설치한다.

```bash
$ conda activate bioinfo
$ conda install bwa-mem2
```

## 20.3 참조 유전체 준비

### 20.3.1 참조 유전체 다운로드

인간 참조 유전체는 UCSC, NCBI, Ensembl 등에서 다운로드할 수 있다. 일반적으로 GRCh38(hg38)을 사용한다.

```bash
$ wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
$ gunzip hg38.fa.gz
```

### 20.3.2 인덱스 생성

정렬을 수행하기 전에 참조 유전체의 인덱스를 생성해야 한다. 인덱스는 BWT와 FM Index 구조를 기반으로 하며, 한 번 생성하면 동일한 참조 유전체에 대해 재사용할 수 있다.

```bash
$ bwa-mem2 index hg38.fa
```

인덱스 생성에는 시간이 소요된다. 완료되면 다음과 같은 파일들이 생성된다:

```
hg38.fa.0123
hg38.fa.amb
hg38.fa.ann
hg38.fa.bwt.2bit.64
hg38.fa.pac
```

## 20.4 BWA-MEM2를 이용한 정렬

### 20.4.1 기본 정렬 실행

Paired-end Fastq 파일을 참조 유전체에 정렬한다.

```bash
$ bwa-mem2 mem hg38.fa R1.fastq.gz R2.fastq.gz > aligned.sam
```

출력은 SAM 형식으로 생성된다. SAM 파일의 구조와 처리 방법은 [21장 SAM/BAM 파일 처리](21-sam-bam.md)에서 다룬다.

### 20.4.2 Read Group 추가

다운스트림 분석을 위해 Read Group 정보를 추가하는 것이 권장된다. `-R` 옵션을 사용한다.

```bash
$ bwa-mem2 mem -R '@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA' hg38.fa R1.fastq.gz R2.fastq.gz > aligned.sam
```

Read Group 필드:

| 필드 | 설명 |
|---|---|
| ID | Read Group 식별자 |
| SM | 샘플 이름 |
| PL | 시퀀싱 플랫폼 (ILLUMINA, PACBIO 등) |
| LB | 라이브러리 식별자 |
| PU | 플랫폼 유닛 (flowcell-barcode.lane) |

### 20.4.3 스레드 수 지정

`-t` 옵션으로 사용할 스레드 수를 지정하여 정렬 속도를 높일 수 있다.

```bash
$ bwa-mem2 mem -t 8 -R '@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA' hg38.fa R1.fastq.gz R2.fastq.gz > aligned.sam
```

### 20.4.4 파이프라인 구성

SAM 파일은 용량이 크므로, 일반적으로 정렬 결과를 바로 BAM으로 변환하고 정렬한다.

```bash
$ bwa-mem2 mem -t 8 -R '@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA' hg38.fa R1.fastq.gz R2.fastq.gz | samtools sort -o aligned.sorted.bam
```

## 20.5 정렬 통계 확인

정렬이 완료되면 samtools를 사용하여 기본 통계를 확인할 수 있다.

```bash
$ samtools flagstat aligned.sorted.bam
```

주요 확인 항목:
- **mapped**: 참조 유전체에 정렬된 read 비율
- **properly paired**: 올바르게 짝지어진 paired-end read 비율
- **duplicates**: 중복 read 수

## 20.6 실습 과제

### 실습 20.1: 인덱스 생성

1. 제공된 참조 유전체 파일에 대해 BWA-MEM2 인덱스를 생성한다.
2. 생성된 인덱스 파일들을 확인한다.

### 실습 20.2: 정렬 수행

1. Fastq 파일을 참조 유전체에 정렬한다.
2. Read Group 정보를 포함하여 정렬을 수행한다.
3. 정렬 결과를 BAM 형식으로 저장한다.

### 실습 20.3: 정렬 품질 확인

1. samtools flagstat으로 정렬 통계를 확인한다.
2. 정렬률(mapping rate)을 계산한다.
3. properly paired 비율을 확인한다.
