# 딥러닝 기반 세포 유형 분류 실습

## 27.1 개요

이 장에서는 PyTorch Lightning을 사용하여 scRNA-seq 데이터로부터 세포 유형을 분류하는 딥러닝 모델을 구현하는 실습을 다룬다. 딥러닝의 기본 원리, 손실 함수, 신경망 아키텍처에 대한 이론적 내용은 [16장 인공지능과 의생명정보학의 미래](../theory/16-artificial-intelligence-and-biomedical-informatics.md)를 참조한다.

## 27.2 PyTorch와 PyTorch Lightning

### 27.2.1 PyTorch

PyTorch는 딥러닝 모델을 구축하고 훈련하기 위한 오픈소스 프레임워크이다. 동적 계산 그래프를 사용하여 직관적인 디버깅이 가능하며, GPU 가속을 통해 대규모 데이터셋에서도 효율적인 학습이 가능하다.

### 27.2.2 PyTorch Lightning

PyTorch Lightning은 PyTorch 코드를 더 정돈되고 간결한 방식으로 작성할 수 있게 해주는 고수준 라이브러리이다. "Focus on science, not engineering"이라는 철학 아래, 연구자가 모델 아키텍처와 학습 로직에 집중할 수 있도록 보일러플레이트 코드를 최소화한다.

PyTorch Lightning의 주요 특징은 다음과 같다:

| 특징 | 설명 |
|---|---|
| 코드 구조화 | 학습, 검증, 테스트 로직을 체계적으로 분리 |
| 자동 GPU 관리 | 멀티 GPU 학습을 자동으로 처리 |
| 로깅 통합 | TensorBoard 등 다양한 로깅 도구와 통합 |
| 체크포인트 | 모델 상태를 자동으로 저장 및 복원 |

참고: https://lightning.ai/pytorch-lightning

## 27.3 실습 환경 구성

### 27.3.1 작업 디렉토리 생성

```bash
$ mkdir -p ~/week12
$ cd ~/week12
```

### 27.3.2 UV 가상환경 설정

```bash
$ uv venv --python 3.13
$ source .venv/bin/activate
```

### 27.3.3 PyTorch 설치

PyTorch 공식 사이트(https://pytorch.org/get-started/locally/)에서 Linux, pip, Python, CUDA를 선택하고 제시된 명령어를 실행한다.

```bash
$ uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 27.3.4 추가 라이브러리 설치

```bash
$ uv pip install lightning scanpy tensorboard ipykernel scikit-learn
```

### 27.3.5 Jupyter 커널 등록

```bash
$ python -m ipykernel install --user --name week12 --display-name "week12"
```

### 27.3.6 데이터 파일 준비

```bash
$ ln -s /bce/lectures/2025-bioinformatics/data/deconvolution/count-data-diaphragm-annotated.h5ad .
```

## 27.4 딥러닝 모델 설계

### 27.4.1 LightningModule 구조

PyTorch Lightning에서는 LightningModule 클래스를 상속받아 딥러닝 모델을 정의한다. 구현해야 하는 주요 메소드는 다음과 같다:

| 메소드 | 설명 |
|---|---|
| `__init__` | 모델에 필요한 레이어 선언 |
| `forward` | 순전파 레이어 구조 정의 |
| `configure_optimizers` | 옵티마이저 설정 |
| `training_step` | 학습 시 호출되는 메소드 |
| `validation_step` | 검증 시 호출되는 메소드 |

### 27.4.2 레이어 구성

obs-by-feature 형식의 데이터를 처리할 때는 Fully Connected(Linear) 레이어를 사용한다. 일반적인 레이어 구성 순서는 다음과 같다:

```
Linear → ReLU → BatchNorm → Dropout
```

각 레이어의 역할은 다음과 같다:

| 레이어 | 역할 |
|---|---|
| Linear | 입력 차원을 출력 차원으로 선형 변환 |
| ReLU | 비선형 활성화 함수로 복잡한 패턴 학습 가능 |
| BatchNorm | 배치 정규화로 학습 안정화 |
| Dropout | 과적합 방지를 위해 일부 뉴런 비활성화 |

### 27.4.3 손실 함수 선택

분류 문제에서 사용하는 손실 함수는 문제 유형에 따라 다르다:

| 문제 유형 | 손실 함수 | PyTorch 구현 |
|---|---|---|
| 이진 분류 | Binary Cross Entropy | `nn.BCELoss()` |
| 다중 클래스 분류 | Categorical Cross Entropy | `nn.CrossEntropyLoss()` |
| 회귀 | Mean Squared Error | `nn.MSELoss()` |

세포 유형 분류는 다중 클래스 분류 문제이므로 CrossEntropyLoss를 사용한다.

## 27.5 scRNA-seq 분류기 구현

### 27.5.1 필요 라이브러리 임포트

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
```

### 27.5.2 모델 클래스 정의

```python
class SCRNAClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(0.3)
            )

        self.network = nn.Sequential(
            block(input_dim, 128),
            block(128, 64),
            nn.Linear(64, num_classes)
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.CrossEntropyLoss()(self(x), y)
        acc = (self(x).argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.CrossEntropyLoss()(self(x), y)
        acc = (self(x).argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

### 27.5.3 네트워크 구조

위 모델의 네트워크 구조는 다음과 같다:

```
Input (50) → Linear → ReLU → BatchNorm → Dropout (0.3)
         → Linear → ReLU → BatchNorm → Dropout (0.3)
         → Linear → Output (num_classes)
           (128)                    (64)
```

## 27.6 데이터 전처리

### 27.6.1 데이터 로드 및 전처리

```python
adata = sc.read_h5ad('count-data-diaphragm-annotated.h5ad')

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)
```

### 27.6.2 UMAP 시각화

```python
sc.pl.umap(adata, color='cell_type', legend_loc='on data')
```

### 27.6.3 학습/테스트 데이터 분리

```python
np.random.seed(42)
mask = np.random.rand(adata.n_obs) < 0.8
adata.obs['train_mask'] = mask
adata.obs['test_mask'] = ~mask
```

### 27.6.4 PyTorch 텐서 변환

```python
X = adata.obsm["X_pca"]
y = LabelEncoder().fit_transform(adata.obs['cell_type'])
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

X_train = X[adata.obs['train_mask'].values]
y_train = y[adata.obs['train_mask'].values]
X_test = X[adata.obs['test_mask'].values]
y_test = y[adata.obs['test_mask'].values]
```

### 27.6.5 데이터로더 생성

```python
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=128
)
```

## 27.7 모델 훈련

### 27.7.1 트레이너 설정

```python
trainer = pl.Trainer(
    max_epochs=50,
    logger=TensorBoardLogger('logs'),
    accelerator='auto',
    devices=[0]
)

model = SCRNAClassifier(50, len(set(y.numpy())))
```

### 27.7.2 모델 훈련 실행

```python
trainer.fit(model, train_loader, test_loader)
```

### 27.7.3 TensorBoard로 학습 과정 모니터링

```python
%load_ext tensorboard
%tensorboard --logdir logs --bind_all
```

TensorBoard를 통해 train_loss, train_acc, val_loss, val_acc 등의 지표를 실시간으로 확인할 수 있다.

## 27.8 모델 저장 및 로드

### 27.8.1 체크포인트 저장

```python
trainer.save_checkpoint("best_model.ckpt")
```

### 27.8.2 체크포인트 로드

```python
model = SCRNAClassifier.load_from_checkpoint("best_model.ckpt")
model.eval()
```

## 27.9 추론 및 결과 시각화

### 27.9.1 테스트 데이터 추론

```python
with torch.no_grad():
    y_infer = model(X_test)
```

### 27.9.2 결과 어노테이션

```python
adata_test = adata[~mask].copy()

y_infer_labels = y_infer.argmax(dim=1).numpy()
le = LabelEncoder().fit(adata.obs['cell_type'])
y_infer_labels = le.inverse_transform(y_infer_labels)

adata_test.obs['y_infer'] = y_infer_labels
```

### 27.9.3 결과 시각화

```python
sc.pl.umap(adata_test, color=['cell_type', 'y_infer'], legend_loc='on data')
```

## 27.10 scVI를 이용한 배치 효과 보정

### 27.10.1 scVI 개요

scVI(single-cell Variational Inference)는 변분 오토인코더(VAE)를 기반으로 하는 단일세포 데이터 분석 도구이다. scVI는 딥러닝을 활용하여 배치 효과를 보정하고, 세포를 저차원 잠재 공간에 임베딩한다.

scVI의 생성 모델은 다음과 같은 확률 분포를 가정한다:

| 변수 | 분포 | 설명 |
|---|---|---|
| z_n | N(0, I) | 잠재 변수 (세포 상태) |
| w_ng | Gamma | 유전자 발현 기댓값 |
| y_ng | Poisson | 관측된 카운트 |
| h_ng | Bernoulli | 드롭아웃 여부 |

### 27.10.2 scVI 설치 및 사용

```bash
$ uv pip install scvi-tools
```

```python
import scvi

adata = sc.read_h5ad('data.h5ad')
scvi.model.SCVI.setup_anndata(adata, batch_key='batch')

model = scvi.model.SCVI(adata)
model.train()

adata.obsm['X_scVI'] = model.get_latent_representation()
```

scVI로 얻은 잠재 표현은 배치 효과가 보정된 상태로, 클러스터링이나 시각화에 활용할 수 있다.

## 27.11 실습 과제

### 실습 27.1: 기본 분류기 구현

1. 본문에서 제시된 코드를 사용하여 scRNA-seq 분류기를 구현한다.
2. 50 에폭 동안 훈련하고 TensorBoard로 학습 과정을 모니터링한다.
3. 테스트 데이터에 대한 예측 결과를 UMAP으로 시각화한다.

### 실습 27.2: 모델 아키텍처 개선

1. 레이어를 추가하여 input_dim → 512 → 256 → 128 → num_classes 구조로 수정한다.
2. 레이어 순서를 Linear → BatchNorm → ReLU → Dropout으로 변경한다.
3. 클래스별 가중치를 적용하여 불균형 데이터 문제를 해결한다.

클래스별 가중치 적용 예시:

```python
# 클래스별 샘플 수 계산
class_counts = np.bincount(y_train.numpy())
class_weights = 1.0 / class_counts
class_weights = torch.FloatTensor(class_weights / class_weights.sum())

# CrossEntropyLoss에 가중치 적용
loss = nn.CrossEntropyLoss(weight=class_weights)(self(x), y)
```

### 실습 27.3: 다른 데이터셋 적용

1. `/bce/lectures/2025-bioinformatics/data/scrnaseq/brain_small.h5ad` 파일을 사용한다.
2. 개선된 모델 구조로 훈련을 수행한다.
3. 분류 정확도와 세포 유형별 성능을 분석한다.
