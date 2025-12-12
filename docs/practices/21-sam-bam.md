# SAM/BAM 파일 처리 실습

## 21.1 개요

이 장에서는 시퀀스 정렬 결과인 SAM/BAM 파일의 구조를 이해하고 samtools를 사용한 처리 방법을 다룬다.

## 21.2 실습 환경 구성

conda를 사용하여 samtools를 설치한다.

```bash
$ conda activate bioinfo
$ conda install samtools
```

## 21.3 SAM/BAM 파일 형식

### 21.3.1 SAM 파일 구조

SAM(Sequence Alignment/Map)은 정렬 결과를 저장하는 텍스트 형식이다. 헤더와 정렬 레코드로 구성된다.

헤더 예시:
```
@HD	VN:1.6	SO:coordinate
@SQ	SN:chr1	LN:248956422
@RG	ID:sample1	SM:sample1	PL:ILLUMINA
@PG	ID:bwa-mem2	PN:bwa-mem2	VN:2.2.1
```

헤더 태그:

| 태그 | 설명 |
|---|---|
| @HD | 헤더 정보 (버전, 정렬 순서) |
| @SQ | 참조 서열 정보 (이름, 길이) |
| @RG | Read Group 정보 |
| @PG | 프로그램 정보 |

### 21.3.2 정렬 레코드

각 read의 정렬 정보는 탭으로 구분된 11개의 필수 필드로 구성된다.

| 필드 | 이름 | 설명 |
|---|---|---|
| 1 | QNAME | Read 이름 |
| 2 | FLAG | 비트 플래그 |
| 3 | RNAME | 참조 서열 이름 |
| 4 | POS | 정렬 시작 위치 (1-based) |
| 5 | MAPQ | 맵핑 품질 |
| 6 | CIGAR | 정렬 정보 |
| 7 | RNEXT | 짝 read의 참조 서열 |
| 8 | PNEXT | 짝 read의 위치 |
| 9 | TLEN | 템플릿 길이 |
| 10 | SEQ | 서열 |
| 11 | QUAL | 품질 점수 |

### 21.3.3 FLAG 필드

FLAG 필드는 read의 상태를 비트 플래그로 표현한다.

| 값 | 의미 |
|---|---|
| 1 | paired-end 시퀀싱 |
| 2 | properly paired |
| 4 | unmapped |
| 8 | mate unmapped |
| 16 | reverse strand |
| 32 | mate reverse strand |
| 64 | first in pair (R1) |
| 128 | second in pair (R2) |
| 256 | secondary alignment |
| 512 | failed QC |
| 1024 | duplicate |
| 2048 | supplementary alignment |

여러 상태가 조합되면 값을 더한다. 예를 들어 FLAG=99는 paired(1) + properly paired(2) + mate reverse(32) + first in pair(64) = 99이다.

### 21.3.4 CIGAR 문자열

CIGAR는 정렬 상태를 압축하여 표현한다.

| 문자 | 의미 |
|---|---|
| M | 매치 또는 미스매치 |
| I | 삽입 (read에만 존재) |
| D | 삭제 (참조에만 존재) |
| N | 스킵 영역 (RNA-seq 인트론) |
| S | 소프트 클리핑 |
| H | 하드 클리핑 |

예시: `50M2I30M`은 50bp 매치, 2bp 삽입, 30bp 매치를 의미한다.

### 21.3.5 BAM 형식

BAM은 SAM의 바이너리 압축 형식이다. 파일 크기가 작고 처리 속도가 빠르다. 일반적으로 BAM 형식으로 저장하고 필요시 SAM으로 변환하여 확인한다.

## 21.4 samtools 기본 명령

### 21.4.1 파일 변환

SAM을 BAM으로 변환:

```bash
$ samtools view -bS aligned.sam > aligned.bam
```

BAM을 SAM으로 변환하여 확인:

```bash
$ samtools view aligned.bam | less
```

헤더와 함께 출력:

```bash
$ samtools view -h aligned.bam | less
```

### 21.4.2 정렬 (sorting)

좌표 기준 정렬:

```bash
$ samtools sort -o aligned.sorted.bam aligned.bam
```

Read 이름 기준 정렬:

```bash
$ samtools sort -n -o aligned.namesorted.bam aligned.bam
```

### 21.4.3 인덱싱

정렬된 BAM 파일에 인덱스를 생성하면 특정 영역을 빠르게 조회할 수 있다.

```bash
$ samtools index aligned.sorted.bam
```

인덱스 파일 `.bai`가 생성된다.

### 21.4.4 통계 확인

기본 통계:

```bash
$ samtools flagstat aligned.sorted.bam
```

상세 통계:

```bash
$ samtools stats aligned.sorted.bam > stats.txt
```

## 21.5 BAM 파일 필터링

### 21.5.1 특정 영역 추출

인덱스가 있는 BAM 파일에서 특정 영역만 추출:

```bash
$ samtools view aligned.sorted.bam chr1:1000000-2000000
```

### 21.5.2 FLAG 기반 필터링

mapped read만 추출 (`-F 4`는 unmapped 제외):

```bash
$ samtools view -F 4 aligned.sorted.bam > mapped.bam
```

properly paired read만 추출:

```bash
$ samtools view -f 2 aligned.sorted.bam > proper_pairs.bam
```

주요 옵션:

| 옵션 | 설명 |
|---|---|
| `-f` | 해당 FLAG가 설정된 read만 포함 |
| `-F` | 해당 FLAG가 설정된 read 제외 |
| `-q` | 최소 맵핑 품질 |

### 21.5.3 품질 기반 필터링

맵핑 품질 20 이상인 read만 추출:

```bash
$ samtools view -q 20 aligned.sorted.bam > high_quality.bam
```

## 21.6 중복 표시

PCR 중복은 라이브러리 준비 과정에서 발생하며, 분석 전에 표시하거나 제거해야 한다.

```bash
$ samtools markdup aligned.sorted.bam marked.bam
```

중복 제거:

```bash
$ samtools markdup -r aligned.sorted.bam dedup.bam
```

## 21.7 IGV를 이용한 시각화

IGV(Integrative Genomics Viewer)는 BAM 파일을 시각적으로 확인할 수 있는 도구이다.

### 21.7.1 IGV 설치

IGV는 https://igv.org/doc/desktop/ 에서 다운로드할 수 있다.

### 21.7.2 BAM 파일 로드

1. 참조 유전체를 선택한다 (Genomes > Load Genome).
2. BAM 파일을 로드한다 (File > Load from File).
3. 관심 영역으로 이동하여 정렬 결과를 확인한다.

IGV에서 각 read는 막대로 표시되며, 색상은 방향이나 매핑 품질을 나타낸다. 변이가 있는 위치는 다른 색으로 표시된다.

## 21.8 전체 파이프라인 예시

Fastq에서 분석 가능한 BAM까지의 전체 과정:

```bash
# 정렬 및 BAM 변환
$ bwa-mem2 mem -t 8 -R '@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA' \
    hg38.fa R1.fastq.gz R2.fastq.gz | samtools sort -o aligned.sorted.bam

# 인덱싱
$ samtools index aligned.sorted.bam

# 중복 표시
$ samtools markdup aligned.sorted.bam marked.bam
$ samtools index marked.bam

# 통계 확인
$ samtools flagstat marked.bam
```

## 21.9 실습 과제

### 실습 21.1: SAM 파일 구조 이해

1. SAM 파일을 열어 헤더와 정렬 레코드를 확인한다.
2. 특정 read의 FLAG 값을 해석한다.
3. CIGAR 문자열이 의미하는 정렬 상태를 설명한다.

### 실습 21.2: samtools 기본 사용

1. SAM 파일을 BAM으로 변환한다.
2. BAM 파일을 좌표 기준으로 정렬한다.
3. 인덱스를 생성한다.
4. flagstat으로 통계를 확인한다.

### 실습 21.3: 필터링과 시각화

1. 특정 염색체 영역의 read만 추출한다.
2. 맵핑 품질 20 이상인 read만 추출한다.
3. IGV에서 BAM 파일을 로드하여 정렬 결과를 확인한다.
