# 리눅스

## 리눅스의 중요성

대부분의 생명정보학 도구는 리눅스용으로 출시된다. 따라서 의생명정보학 연구를 위해서는 리눅스 환경에서의 작업이 필수적이다. 이는 리눅스 환경이 고성능 컴퓨팅과 대용량 데이터 처리에 최적화되어 있기 때문이며, 주로 다음과 같은 이유로 생명정보학 분야에서 널리 사용된다:

1. 다양한 오픈소스 도구의 가용성
2. 명령줄 인터페이스를 통한 효율적인 데이터 처리
3. 분산 컴퓨팅 환경과의 호환성
4. 배치 처리를 통한 대규모 분석 자동화 가능성

생명정보학을 공부하고 실무에 적용하기 위해서는 리눅스를 공부하는 것이 매우 중요하다.

## 리눅스의 역사와 철학

### 자유 소프트웨어 재단 (Free Software Foundation, FSF)

자유 소프트웨어 재단은 1985년 리처드 스톨만(Richard Stallman, 일명 "RMS")에 의해 창설되었다. 이 재단의 핵심 철학은 "자유(Free) 소프트웨어"의 개념이다. 여기서 말하는 "자유"란 누구나 자유롭게 소프트웨어를 공부하고, 변경하고, 배포할 수 있도록 그 소스 코드가 공개된 소프트웨어를 의미한다. 

이는 소프트웨어를 무료로 제공해야 한다는 의미가 아니다. 리처드 스톨만의 유명한 말에 따르면 "자유 소프트웨어는 언론의 자유와 같은 자유를 의미하는 것이지, 무료 맥주를 의미하는 것이 아니다(Free in the sense of free speech, not free beer)."

자유 소프트웨어 재단은 일반 공중 사용 허가서(General Public License, GPL)를 통해 자유 소프트웨어의 권리를 보호한다. GPL은 자유 소프트웨어를 사용하여 개발한 소프트웨어도 반드시 자유 소프트웨어가 되어야 한다는 조건을 포함한다.

### GNU 프로젝트

GNU(GNU is not UNIX!)는 자유 소프트웨어 재단의 주요 프로젝트 중 하나이다. 이 프로젝트의 목표는 당시 가장 인기 있었던 운영체제인 유닉스(UNIX)의 "클론"을 제작하는 것이었다. 유닉스는 커널(Kernel)과 다양한 유틸리티로 구성되어 있었다.

리처드 스톨만은 이를 위해 수많은 유틸리티를 개발했다:
- Emacs (텍스트 에디터)
- GCC (GNU Compiler Collection, C 컴파일러)
- GDB (디버거)
- GNU make (빌드 자동화 도구)
등 다양한 도구들이 개발되었다.

### 리눅스 커널의 탄생

한편, 지구 반대편 핀란드에서는 리누스 토발즈(Linus Torvalds)가 "재미로(just for fun)" 운영체제 커널을 개발하고 있었다. 이 커널은 후에 GNU 프로젝트의 일부로 편입되게 된다.

![리눅스 발표](../assets/images/linus-releases-linux.png)

**Figure 18.1** 리누스 토발즈가 리눅스를 처음 발표한 역사적 순간 (1991년)

### GNU/Linux 시스템

GNU 프로젝트의 유틸리티와 리눅스 커널의 결합으로 GNU/Linux 시스템이 탄생했다. 이 시스템은 유닉스와 완전히 호환되는 POSIX 준수 시스템으로, 현재 99% 이상의 서버 시스템이 리눅스를 채택하고 있다.

![스톨만과 토발즈](../assets/images/richard-and-linus.png)

**Figure 18.2** 리처드 스톨만과 리누스 토발즈

현재는 이 시스템을 간단히 "리눅스"라고 부르는 것이 일반적이다. 이는 시스템에 포함된 모든 소프트웨어가 GPL을 따르는 것은 아니기 때문이다. 물론 리처드 스톨만은 리눅스라는 명칭보다는 GNU/Linux라는 명칭을 선호한다.

## 리눅스 배포판

### 배포판의 개념

리눅스 배포판은 각자의 취향과 목적에 맞는 소프트웨어를 기본으로 포함하는 리눅스 운영체제이다. 다양한 배포판이 존재하며, 이들은 웹사이트 https://distrowatch.com/ 에서 확인할 수 있다.

### 주요 배포판

가장 널리 사용되는 두 카테고리의 배포판은 다음과 같다:
- 데비안 계열: 우분투 리눅스(Ubuntu Linux)가 대표적이다.
- 레드햇 계열: CentOS가 대표적이다.

### 패키지 매니저

배포판의 주요 차이점 중 하나는 패키지 매니저이다:
- 데비안 계열: APT (Advanced Package Tool)
- 레드햇 계열: YUM (Yellowdog Updater Modified), 최근에는 DNF (Dandified YUM)

이들은 비슷한 활용법을 가지고 있다:
- `apt install 패키지명` (데비안 계열)
- `yum install 패키지명` 또는 `dnf install 패키지명` (레드햇 계열)

패키지 매니저는 "루트(root)" 권한이 있는 사용자만 활용할 수 있다.

## 리눅스 기본 개념

### 루트 (Root)

리눅스에서 "루트"는 두 가지 의미를 가진다:
1. 시스템 관리자 (슈퍼유저)
2. 디렉토리 구조에서 가장 첫 번째 노드 (/)

### 디렉토리

디렉토리는 파일 및 다른 디렉토리를 포함할 수 있는 논리적 공간을 의미한다. 윈도우의 "폴더"와 같은 개념이다.

![파일 시스템 경로](../assets/images/linux-paths.png)

**Figure 18.3** 리눅스 파일 시스템의 경로 구조와 절대경로/상대경로

### 쉘 (Shell)

쉘은 커널과 사용자 간의 인터페이스 역할을 한다. 주로 CUI(Command-Line User Interface) 기반으로, 키보드를 통해 명령어를 입력하고 화면에 그 결과를 확인할 수 있다.

주요 쉘의 종류:
- sh: Bourne shell (UNIX 표준 쉘)
- bash: GNU Bourne-Again Shell (GNU 프로젝트 표준 쉘)
- csh, zsh, fish 등: 편의기능이 추가된 쉘

## 가상 환경 구축: Conda

Conda는 프로젝트별로 분리된 가상 환경을 구축할 수 있게 해주는 도구이다. 이는 서로 다른 프로젝트에서 서로 다른 버전의 소프트웨어를 사용해야 할 때 특히 유용하다.

예를 들어, `project1`에서는 Python 3.4가 필요하고, `project2`에서는 Python 3.10이 필요한 경우, Conda를 사용하면 두 프로젝트의 환경을 분리하여 서로 다른 버전의 파이썬을 설치할 수 있다.

### Conda 설치

Conda는 처리 속도가 느린 편이므로, 더 빠른 속도로 동작하는 Conda의 클론인 Micromamba를 설치하는 것을 추천한다. Micromamba는 https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html 에서 설치할 수 있다.

### 가상환경 생성 예제

다음은 두 가지 가상환경을 생성하고, 서로 다른 버전의 Python을 설치하는 예제이다:

```bash
# project1 환경 생성 (Python 3.9)
$ micromamba create -n project1 python=3.9
$ micromamba activate project1
(project1) $ python --version
Python 3.9.x

# 필요한 패키지 설치
(project1) $ micromamba install -c conda-forge numpy pandas matplotlib
(project1) $ python -c "import numpy; print(numpy.__version__)"
1.xx.x

# 환경 비활성화
(project1) $ micromamba deactivate

# project2 환경 생성 (Python 3.13)
$ micromamba create -n project2 python=3.13
$ micromamba activate project2
(project2) $ python --version
Python 3.13.x

# 필요한 패키지 설치 (동일한 패키지, 다른 버전)
(project2) $ micromamba install -c conda-forge numpy pandas matplotlib
(project2) $ python -c "import numpy; print(numpy.__version__)"
2.xx.x

# 환경 비활성화
(project2) $ micromamba deactivate

# 설치된 환경 목록 확인
$ micromamba env list
```

이 예제에서는 micromamba를 사용하여 두 개의 독립된 가상환경(project1과 project2)을 생성하고, 각각 Python 3.9와 Python 3.13을 설치했다. 각 환경에서는 동일한 패키지(numpy, pandas, matplotlib)를 설치했지만, Python 버전이 다르기 때문에 패키지 버전도 다를 수 있다.
각 환경은 완전히 독립적이므로, 한 환경에서의 변경이 다른 환경에 영향을 주지 않는다. 이는 서로 다른 버전의 패키지가 필요한 여러 프로젝트를 동시에 진행할 때 매우 유용하다.

## 생명정보학 문제 해결: Rosalind

Rosalind는 생명정보학 문제풀이 웹사이트(https://rosalind.info/)로, 생명정보학을 코딩테스트처럼 공부할 수 있게 해준다. 생명정보학 개념을 실습을 통해 배우고 싶다면 Rosalind의 문제를 풀어보는 것이 좋은 방법이다.

## 리눅스 필수 명령어

### 파일 시스템 탐색

- **pwd** (Print Working Directory): 현재 디렉토리 경로 확인
  - 경로가 /로 시작하는 경우 "절대경로", 그렇지 않은 경우 "상대경로"

![pwd 명령어](../assets/images/pwd.png)

**Figure 18.4** pwd 명령어를 통한 현재 작업 디렉토리 확인

- **cd** (Change Directory): 디렉토리 변경
  - 탭키를 활용하여 긴 경로를 쉽게 입력할 수 있다.
- **ls** (List): 디렉토리 내용 표시

### 파일 및 디렉토리 관리

- **chmod** (Change Mode): 파일/디렉토리 권한 변경
- **groups**: 현재 내가 소속된 그룹 확인
- **cp** (Copy): 파일/디렉토리 복사
  - 디렉토리 복사 시에는 -r 옵션이 필요하다.
  - `cp 소스파일 {대상파일 또는 대상디렉토리}`
  - `cp -r 소스디렉토리 대상디렉토리`
- **mv** (Move): 파일/디렉토리 이동 또는 이름 변경
  - `mv 소스파일 {대상파일 또는 대상디렉토리}`
- **mkdir** (Make Directory): 디렉토리 생성
  - `mkdir 디렉토리명`
- **rmdir** (Remove Directory): 디렉토리 삭제 (비어있는 디렉토리만 삭제 가능)
  - `rmdir 디렉토리명`
- **rm** (Remove): 파일/디렉토리 삭제
  - 디렉토리 삭제를 위해서는 -r 옵션을 사용해야 한다.
  - `rm 파일명`
  - `rm -r 디렉토리명`
  - 한번 삭제된 파일은 복구할 수 없으므로 신중히 사용해야 한다.

### 시스템 모니터링

- **htop**: 현재 실행 중인 프로세스 확인

![htop 화면](../assets/images/htop.png)

**Figure 18.5** htop을 이용한 시스템 리소스 모니터링

### 파일 조작

- **cat / less**: 파일 내용 확인
- **zcat / zless**: gz 압축된 파일 내용 확인
- **nano**: 간단한 텍스트 편집기

![nano 에디터](../assets/images/nano.png)

**Figure 18.6** nano 텍스트 에디터의 인터페이스와 기본 사용법

- **wget**: 인터넷에서 파일 다운로드

## 생명정보학에서의 리눅스 활용

리눅스는 생명정보학에서 필수적인 도구이다. 대부분의 생명정보학 소프트웨어와 데이터베이스는 리눅스 환경에서 개발되고 운영된다. 또한, 대용량 데이터 처리, 병렬 컴퓨팅, 자동화된 워크플로우 구축 등 생명정보학 연구에 필요한 다양한 작업을 리눅스 환경에서 효율적으로 수행할 수 있다.

리눅스의 명령줄 인터페이스(CLI)는 반복적인 작업을 자동화하고, 복잡한 데이터 처리 파이프라인을 구축하는 데 유용하다. 또한, 서버 환경에서의 원격 작업 및 배치 작업 실행에도 리눅스의 기능이 필수적이다.