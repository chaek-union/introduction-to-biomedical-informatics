# NGS 시퀀싱 데이터 처리 실습

## 19.1 개요

이 장에서는 차세대 시퀀싱(NGS) 데이터의 실제 처리 방법을 다룬다. NGS 기술의 원리와 Fastq 파일 형식에 대한 이론적 내용은 [3장 차세대 시퀀싱](../theory/3-ngs.md)을 참조한다.

## 19.2 실습 환경 구성

conda를 사용하여 실습에 필요한 도구들을 설치한다.

```bash
$ conda create -n bioinfo
$ conda activate bioinfo
$ conda install fastqc cutadapt
```

## 19.3 Fastq 파일 확인

### 19.3.1 압축 파일 내용 확인

Fastq 파일은 일반적으로 gzip으로 압축되어 `.fastq.gz` 형태로 제공된다. `zcat`과 `less` 명령을 사용하여 압축된 상태로 내용을 확인할 수 있다.

```bash
$ zcat R1.fastq.gz | less
```

Fastq 파일의 구조와 각 필드의 의미는 [3장 Fastq 파일 형식과 구조](../theory/3-ngs.md#fastq-파일-형식과-구조)를 참조한다.

### 19.3.2 Paired-end 파일

Paired-end 시퀀싱의 경우 두 개의 Fastq 파일이 생성된다. 파일명에 R1과 R2로 구분되며, 각각 Read 1과 Read 2를 포함한다.

```
Liver_SeqScope_2nd_1_R1.fastq.gz
Liver_SeqScope_2nd_1_R2.fastq.gz
```

두 파일의 줄 수는 동일하며, 같은 위치의 read는 동일한 DNA 조각에서 유래한 짝이다.

## 19.4 FastQC를 이용한 품질 확인

FastQC는 Fastq 파일의 품질을 확인하는 도구이다. 다음과 같은 정보를 제공한다.

- 위치별 품질 점수 분포
- 서열 품질 점수 분포
- 위치별 염기 조성
- GC 함량 분포
- 서열 길이 분포
- 중복 서열 수준
- 과다 표현 서열
- 어댑터 오염도

### 19.4.1 FastQC 실행

단일 파일 실행:

```bash
$ fastqc R1.fastq.gz
```

Paired-end 파일 동시 실행:

```bash
$ fastqc R1.fastq.gz R2.fastq.gz
```

for 루프를 사용한 여러 파일 처리:

```bash
$ for i in `seq 1 2`; do fastqc R${i}.fastq.gz; done
```

### 19.4.2 결과 확인

FastQC 실행이 완료되면 HTML 형식의 보고서가 생성된다. 웹 브라우저에서 열어 품질을 확인한다.

주요 확인 항목:
- **Per base sequence quality**: 각 위치별 품질 점수 분포. 일반적으로 read 끝으로 갈수록 품질이 낮아진다.
- **Per sequence quality scores**: 전체 read의 평균 품질 분포.
- **Adapter Content**: 어댑터 서열 오염도. 높은 비율이 나타나면 트리밍이 필요하다.

## 19.5 Cutadapt를 이용한 전처리

Cutadapt는 어댑터 서열 제거와 품질 기반 트리밍을 수행하는 도구이다.

### 19.5.1 어댑터 오염

어댑터 오염은 DNA insert가 read 길이보다 짧을 때 발생한다. 시퀀싱이 insert를 지나 반대쪽 어댑터 서열까지 읽어버리기 때문이다. 이러한 어댑터 서열은 분석 전에 반드시 제거해야 한다.

Illumina Universal Adapter 서열은 `AGATCGGAAGAG`로 시작한다. 사용하는 시퀀싱 키트에 따라 전체 어댑터 서열이 다를 수 있으므로, 정확한 서열은 Illumina Adapter Sequences 문서를 참조한다.

Cutadapt는 어댑터 서열 외에도 poly-A tail을 제거하는 데 사용할 수 있다. RNA-seq 데이터에서 poly-A tail이 포함된 경우 `-a "A{100}"` 옵션으로 제거할 수 있다.

### 19.5.2 Cutadapt 옵션

주요 옵션:

| 옵션 | 설명 |
|---|---|
| `-a` | Read 1에서 제거할 3' 어댑터 서열 |
| `-A` | Read 2에서 제거할 3' 어댑터 서열 |
| `-o` | Read 1 출력 파일 |
| `-p` | Read 2 출력 파일 |
| `-q` | 품질 기반 트리밍 임계값 |

### 19.5.3 어댑터 제거 실행

Paired-end 파일에서 어댑터 제거:

```bash
$ cutadapt -a AGATCGGAAGAG -A AGATCGGAAGAG -o trimmed_R1.fastq.gz -p trimmed_R2.fastq.gz R1.fastq.gz R2.fastq.gz
```

품질 기반 트리밍을 함께 수행하는 경우:

```bash
$ cutadapt -a AGATCGGAAGAG -A AGATCGGAAGAG -q 20 -o trimmed_R1.fastq.gz -p trimmed_R2.fastq.gz R1.fastq.gz R2.fastq.gz
```

### 19.5.4 결과 확인

처리 후 FastQC를 다시 실행하여 어댑터가 제거되었는지 확인한다.

```bash
$ fastqc trimmed_R1.fastq.gz trimmed_R2.fastq.gz
```

## 19.6 실습 과제

### 실습 19.1: Fastq 파일 구조 확인

1. 제공된 Fastq 파일을 `zcat`과 `less`를 사용하여 확인한다.
2. 첫 번째 read의 각 줄이 무엇을 의미하는지 설명한다.
3. 품질 점수 문자를 Phred 점수로 변환한다.

### 실습 19.2: FastQC 분석

1. R1.fastq.gz와 R2.fastq.gz에 대해 FastQC를 실행한다.
2. 생성된 HTML 보고서를 확인한다.
3. 어댑터 오염이 있는지 확인한다.

### 실습 19.3: 어댑터 제거

1. Cutadapt를 사용하여 어댑터를 제거한다.
2. 처리 전후의 FastQC 결과를 비교한다.
3. 어댑터 제거 효과를 확인한다.
