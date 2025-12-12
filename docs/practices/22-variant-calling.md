# 변이 찾기 실습

## 개요

이 장에서는 BAM 파일로부터 유전 변이를 찾는 실습을 다룬다. 변이 찾기(Variant Calling)의 이론적 배경과 VCF 파일 형식에 대한 내용은 [4장 변이와 진화](../theory/4-variant-and-evolution.md)를 참조한다.

## 실습 환경 구성

### 작업 디렉토리 생성

```bash
$ mkdir -p ~/week5
$ cd ~/week5
```

### Octopus 설치

conda를 사용하여 Octopus를 설치한다.

```bash
$ conda activate bioinfo
$ conda install octopus
```

Octopus는 최근 벤치마크에서 높은 성능을 보이는 variant caller로, local realignment 기능을 내장하고 있어 indel 주변에서의 정확한 변이 탐지가 가능하다.

## 입력 파일 준비

### 심볼릭 링크 생성

이전 실습에서 생성한 파일들을 week5 디렉토리로 심볼릭 링크한다.

```bash
$ ln -s ../week4/reference.fa.gz .
$ ln -s ../week4/aligned.sorted.markdup.bam .
$ ln -s ../week4/aligned.sorted.markdup.bam.bai .
```

### 참조 유전체 준비

Octopus는 압축 해제된 참조 유전체와 faidx 파일을 필요로 한다.

```bash
$ zcat reference.fa.gz > reference.fa
$ samtools faidx reference.fa
```

faidx 명령은 참조 유전체의 인덱스 파일(.fai)을 생성한다. 이 인덱스는 특정 유전체 영역에 빠르게 접근할 수 있게 해준다.

## Octopus를 이용한 변이 찾기

### 기본 실행

다음 명령어로 variant calling을 수행한다.

```bash
$ octopus -R reference.fa -I aligned.sorted.markdup.bam -o variants.vcf.gz
```

주요 옵션 설명:

| 옵션 | 설명 |
|---|---|
| -R | 참조 유전체 파일 |
| -I | 입력 BAM 파일 |
| -o | 출력 VCF 파일 |

실행 시간은 데이터 크기에 따라 다르며, 예제 데이터의 경우 약 4분이 소요된다.

### 멀티스레드 실행

대용량 데이터의 경우 스레드 수를 지정하여 속도를 높일 수 있다.

```bash
$ octopus -R reference.fa -I aligned.sorted.markdup.bam -o variants.vcf.gz --threads 8
```

## VCF 파일 확인

### 파일 내용 확인

생성된 VCF 파일의 내용을 확인한다.

```bash
$ zcat variants.vcf.gz | less
```

VCF 파일은 헤더(##로 시작)와 본문(#CHROM으로 시작하는 컬럼 헤더 이후)으로 구성된다.

### VCF 필드 이해

VCF 파일의 주요 필드는 다음과 같다:

| 필드 | 설명 |
|---|---|
| CHROM | 염색체 번호 |
| POS | 유전체 상의 위치 (1-based) |
| ID | dbSNP ID (있는 경우) |
| REF | 참조 유전체의 염기 |
| ALT | 변이된 염기 |
| QUAL | 변이 호출 품질 점수 |
| FILTER | 필터 상태 (PASS 또는 필터 실패 이유) |
| INFO | 변이에 대한 추가 정보 |
| FORMAT | 샘플별 데이터 형식 |

VCF 형식에 대한 자세한 내용은 [4장 변이와 진화](../theory/4-variant-and-evolution.md)를 참조한다.

### 변이 유형 구분

REF와 ALT 필드를 비교하여 변이 유형을 판단할 수 있다:

- **SNV (Single Nucleotide Variant)**: REF와 ALT의 길이가 같은 경우 (예: A → T)
- **Insertion**: ALT가 REF보다 긴 경우 (예: T → TGGA)
- **Deletion**: REF가 ALT보다 긴 경우 (예: TGG → T)
- **MNV (Multiallelic Nucleotide Variant)**: ALT에 쉼표로 구분된 여러 대립유전자가 있는 경우

## ANNOVAR를 이용한 변이 주석

### ANNOVAR 다운로드

ANNOVAR는 변이에 대한 기능적 주석을 추가하는 도구이다. ANNOVAR는 학술 목적으로 무료로 사용할 수 있으며, 다운로드를 위해서는 사용자 등록이 필요하다.

1. https://annovar.openbioinformatics.org/en/latest/user-guide/download/ 페이지에 접속한다.
2. 페이지 하단의 등록 양식을 작성한다 (이름, 이메일, 소속 기관 등).
3. 등록 후 이메일로 다운로드 링크를 받는다.
4. 다운로드한 파일을 week5 디렉토리에 압축 해제한다.

```bash
$ tar -xvzf annovar.latest.tar.gz
```

압축 해제 후 annovar 디렉토리가 생성된다.

### ANNOVAR 실행 환경 설정

ANNOVAR 실행에 필요한 Perl 모듈을 설치한다.

```bash
$ conda install perl-pod-usage
```

### 예제 파일 준비

ANNOVAR 예제 파일을 심볼릭 링크한다.

```bash
$ ln -s annovar/example/ex2.vcf .
```

### ANNOVAR 실행

table_annovar.pl을 사용하여 변이 주석을 수행한다.

```bash
$ annovar/table_annovar.pl ex2.vcf annovar/humandb \
    -buildver hg19 \
    -out myanno \
    -remove \
    -protocol refGene,cytoBand,exac03,avsnp147,dbnsfp30a \
    -operation g,r,f,f,f \
    -nastring . \
    -vcfinput \
    -polish
```

### ANNOVAR 옵션 설명

주요 옵션:

| 옵션 | 설명 |
|---|---|
| -buildver | 참조 유전체 버전 (hg19, hg38 등) |
| -out | 출력 파일 접두사 |
| -protocol | 사용할 데이터베이스 목록 |
| -operation | 각 프로토콜의 작업 유형 |
| -vcfinput | VCF 형식 입력 사용 |

operation 유형:

| 유형 | 설명 |
|---|---|
| g (Gene-based) | 변이 위치(Exonic, Intronic, UTR)와 아미노산 변화 분석 |
| r (Region-based) | 알려진 유전체 영역(TF binding sites, 반복 영역 등)과의 중첩 확인 |
| f (Filter-based) | 데이터베이스(dbSNP, gnomAD 등)에서 계산된 점수 참조 |

### 결과 확인

주석이 추가된 VCF 파일을 확인한다.

```bash
$ less myanno.hg19_multianno.vcf
```

## 유전변이 데이터베이스

### 주요 데이터베이스

변이 분석에 활용되는 주요 데이터베이스:

| 데이터베이스 | 설명 |
|---|---|
| dbSNP | NCBI의 단일염기다형성 데이터베이스 |
| gnomAD | 대규모 인구 집단의 변이 빈도 데이터베이스 |
| 1000 Genomes | 전 세계 인구 집단의 유전 변이 데이터베이스 |
| OMIM | 인간 유전자와 유전 질환 카탈로그 |
| COSMIC | 암 체세포 돌연변이 카탈로그 |

### Blacklist 영역

중복 영역이나 반복 서열 등 분석에서 제외해야 하는 영역을 blacklist로 관리한다. 이러한 영역에서 발견된 변이는 허위 양성(false positive)일 가능성이 높다.

## 분석 파이프라인 개요

### 개인/약물 유전체 분석

개인 유전체나 약물 유전체 분석의 전형적인 파이프라인:

1. Illumina Sequencer → BCL 파일
2. bcl2fastq → FASTQ 파일
3. FastQC → 품질 관리
4. Cutadapt → 어댑터 제거
5. BWA-MEM2 → BAM 파일
6. Octopus 등 → VCF 파일
7. dbSNP, OMIM 등 → 변이 주석

### 질병 유전체 분석

종양-정상 쌍 분석의 경우 체세포 변이(somatic variant)를 찾기 위해 종양 샘플과 정상 샘플(주로 혈액)을 함께 분석한다. 이후 GSEA, KEGG, COSMIC 등의 데이터베이스를 활용하여 기능 분석을 수행한다.

## 실습 과제

### 실습 22.1: Octopus 실행

1. week5 디렉토리를 생성하고 필요한 파일들을 심볼릭 링크한다.
2. 참조 유전체를 압축 해제하고 faidx 인덱스를 생성한다.
3. Octopus로 variant calling을 수행한다.
4. 생성된 VCF 파일의 내용을 확인한다.

### 실습 22.2: VCF 파일 해석

다음 변이의 의미를 설명하시오:

1. CHROM=3, POS=23, REF=A, ALT=T
2. CHROM=5, POS=53, REF=T, ALT=TGGA
3. CHROM=4, POS=4, REF=TGG, ALT=T

### 실습 22.3: ANNOVAR 실행

1. ANNOVAR를 설정하고 예제 파일을 준비한다.
2. table_annovar.pl을 실행하여 변이 주석을 수행한다.
3. 출력된 multianno.vcf 파일의 INFO 필드에 추가된 주석을 확인한다.