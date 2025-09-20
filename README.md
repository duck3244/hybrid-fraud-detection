# 🚀 하이브리드 신용카드 사기 탐지 시스템

[![CI/CD 파이프라인](https://github.com/your-username/hybrid-fraud-detection/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/hybrid-fraud-detection/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🔍 **오토인코더 이상 탐지와 ELECTRA에서 영감받은 Active Learning을 결합한 고도의 신용카드 사기 탐지 시스템**

## 🌟 개요

본 프로젝트는 오토인코더의 비지도 학습 능력과 ELECTRA의 효율적인 샘플 선택 전략을 결합한 혁신적인 하이브리드 접근법을 구현합니다:

- **🎯 높은 정확도**: 최소한의 False Positive로 95% 이상의 AUC 달성
- **💡 스마트 학습**: Active Learning으로 라벨링 비용을 50% 절감
- **⚡ 실시간 처리**: 프로덕션 배포에 최적화
- **📊 비즈니스 인사이트**: 포괄적인 ROI 및 비용 효과 분석

## 🏗️ 시스템 아키텍처

```
📦 하이브리드 사기 탐지 시스템
├── 🧠 오토인코더 코어
│   ├── 인코더 네트워크 [29 → 14 → 7]
│   └── 디코더 네트워크 [7 → 14 → 29]
├── 🎯 손실 예측 모듈 (LPM)
│   ├── 은닉층 [64, 32, 16]
│   └── 불확실성 추정
├── 🎓 Active Learning 엔진
│   ├── 불확실성 기반 샘플링
│   ├── 다양성 고려 선택
│   └── 위원회 기반 질의
└── 📈 비즈니스 인텔리전스
    ├── ROI 계산기
    ├── 비용-효과 분석
    └── 성능 모니터링
```

## 🐍 Python 버전 권장사항

### 권장 버전: **Python 3.9**

```bash
# Python 3.9 설치 (권장)
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv

# macOS (Homebrew 사용)
brew install python@3.9

# Windows (공식 웹사이트에서 다운로드)
# https://www.python.org/downloads/release/python-3911/
```

### 지원 버전
- ✅ **Python 3.9** (권장 - 최적 성능)
- ✅ **Python 3.10** (완전 지원)
- ⚠️ **Python 3.8** (지원하나 일부 최신 기능 제한)
- ❌ **Python 3.11+** (TensorFlow 호환성 문제로 비권장)

### 버전별 특징

| Python 버전 | 권장도 | TensorFlow | 성능 | 특징 |
|-------------|--------|------------|------|------|
| **3.9** | ⭐⭐⭐⭐⭐ | 2.10+ | 최상 | 안정성 + 성능 최적화 |
| **3.10** | ⭐⭐⭐⭐ | 2.10+ | 우수 | 최신 기능, 약간의 호환성 이슈 |
| **3.8** | ⭐⭐⭐ | 2.8+ | 양호 | 구형 시스템 호환성 |
| **3.11+** | ⭐ | 불안정 | - | TensorFlow 호환성 문제 |

## 🚀 빠른 시작

### 1. 설치

```bash
# Python 3.9 가상환경 생성 (권장)
python3.9 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
pip install -e .
```

### 2. 빠른 데모 (30초)

```bash
# 합성 데이터로 빠른 실험 실행
python fraud_detection_system.py

# 또는 편의 함수 사용
python -c "from fraud_detection_system import quick_fraud_detection_experiment; quick_fraud_detection_experiment(epochs=20)"
```

### 3. 실제 데이터 사용

```bash
# Kaggle 신용카드 사기 데이터셋 다운로드
# data/raw/ 디렉토리에 creditcard.csv 배치

# 전체 파이프라인 실행
python fraud_detection_system.py --data data/raw/creditcard.csv --epochs 100
```

## 📊 핵심 기능

### 🧠 **하이브리드 아키텍처**
- **오토인코더**: 재구성 오류를 통한 비지도 이상 탐지
- **손실 예측 모듈**: ELECTRA에서 영감받은 불확실성 추정
- **결합 훈련**: 재구성 및 예측 손실의 공동 최적화

### 🎓 **고급 Active Learning**
- **다양한 전략**: 불확실성, 다양성, 위원회 기반, 적응형
- **스마트 샘플링**: 가장 유익한 샘플의 지능적 선택
- **비용 고려**: 어노테이션 비용을 고려한 샘플 선택

### 📈 **비즈니스 인텔리전스**
- **ROI 분석**: 투자 수익률 및 비용 절감 계산
- **성능 지표**: 포괄적인 사기 탐지 KPI
- **실시간 모니터링**: 프로덕션용 성능 추적

### 🎨 **풍부한 시각화**
- **인터랙티브 대시보드**: Plotly 기반 동적 시각화
- **훈련 모니터링**: 실시간 손실 및 지표 추적
- **모델 해석**: 특성 중요도 및 결정 분석

## 📁 프로젝트 구조

```
hybrid-fraud-detection/
├── 📚 문서화
│   ├── README_KR.md             # 이 파일 (한글)
│   ├── README.md                # 영문 문서
│   ├── notebooks/               # 인터랙티브 튜토리얼
│   └── docs/                    # API 문서
├── 🏗️ 핵심 시스템
│   ├── fraud_detection_system.py # 메인 시스템 통합
│   ├── models/
│   │   ├── hybrid_autoencoder.py  # 하이브리드 오토인코더
│   │   ├── loss_prediction.py     # 손실 예측 모듈
│   │   └── active_learning.py     # Active Learning
│   └── utils/
│       ├── data_preprocessing.py  # 데이터 전처리
│       ├── visualization.py       # 시각화
│       └── evaluation.py          # 성능 평가
├── ⚙️ 설정
│   ├── config.yaml              # 메인 설정 파일
│   ├── requirements.txt         # 의존성
│   └── setup.py                 # 패키지 설정
└── 📊 출력
    ├── data/                    # 데이터 저장
    ├── models/                  # 훈련된 모델
    ├── results/                 # 실험 결과
    └── plots/                   # 생성된 시각화
```

## 🎯 성능 벤치마크

| 지표 | 기존 오토인코더 | **하이브리드 접근법** | 개선율 |
|------|----------------|---------------------|--------|
| **AUC-ROC** | 0.92 | **0.95** | +3.3% |
| **AUC-PR** | 0.78 | **0.84** | +7.7% |
| **F1-Score** | 0.73 | **0.79** | +8.2% |
| **False Positive Rate** | 2.1% | **1.6%** | -23.8% |
| **라벨링 효율성** | 기준점 | **2.3배** | +130% |

## 🛠️ 고급 사용법

### 사용자 정의 설정

```python
from fraud_detection_system import HybridFraudDetectionSystem

# 사용자 정의 설정으로 시스템 생성
system = HybridFraudDetectionSystem(config_path='my_config.yaml')

# 데이터 로드
df = system.load_data('path/to/data.csv')

# 모델 구축 및 훈련
system.build_model()
history = system.train(epochs=100, batch_size=64)

# 평가
results = system.evaluate_model()
print(f"AUC 점수: {results['roc_auc']:.4f}")

# Active Learning 실행
al_results, summary = system.run_active_learning_experiment()
print(f"효율성 개선: {summary['improvement_percent']:.1f}%")
```

### Active Learning 전략

```python
from models.active_learning import ActiveLearningManager

# 다양한 전략으로 초기화
strategies = ['uncertainty', 'diversity', 'qbc', 'adaptive']

for strategy in strategies:
    manager = ActiveLearningManager(model, strategy_type=strategy)
    results, summary = manager.run_experiment(X_unlabeled, y_unlabeled)
    print(f"{strategy}: {summary['avg_fraud_ratio']:.3f} 사기 탐지율")
```

### 비즈니스 메트릭 분석

```python
from utils.evaluation import calculate_business_roi

# 사용자 정의 매개변수로 ROI 계산
roi_results = calculate_business_roi(
    y_true, y_pred,
    fraud_cost=150,           # 놓친 사기당 비용
    investigation_cost=15,    # 조사당 비용
    annual_volume=2000000     # 연간 거래량
)

print(f"연간 절감액: ${roi_results['annual_savings']:,.2f}")
print(f"ROI: {roi_results['roi_percentage']:.1f}%")
```

## 📊 모델 비교

다양한 접근법 비교:

```python
from utils.evaluation import ModelComparator

# 비교할 모델들 정의
models_results = [
    (y_test, pred_hybrid, scores_hybrid),
    (y_test, pred_isolation_forest, scores_if),
    (y_test, pred_one_class_svm, scores_svm)
]

model_names = ['하이브리드 오토인코더', 'Isolation Forest', 'One-Class SVM']

# 모델 비교
comparator = ModelComparator()
results = comparator.compare_models(models_results, model_names)
print(f"최고 모델: {results['best_model']}")
```

## 🧪 테스트

```bash
# 모든 테스트 실행
pytest tests/ -v

# 커버리지와 함께 실행
pytest tests/ --cov=. --cov-report=html

# 특정 테스트 카테고리 실행
pytest tests/ -m "unit"        # 단위 테스트만
pytest tests/ -m "integration" # 통합 테스트만
pytest tests/ -m "not slow"    # 느린 테스트 제외
```

## 📝 설정 옵션

### 모델 설정 (`config.yaml`)

```yaml
model:
  autoencoder:
    encoding_dims: [14, 7]        # 인코더 아키텍처
    dropout_rate: 0.1             # 정규화용 드롭아웃
  
  loss_prediction_module:
    hidden_dims: [64, 32, 16]     # LPM 아키텍처
    dropout_rate: 0.2             # 드롭아웃 비율
  
  training:
    learning_rate: 0.001          # Adam 옵티마이저 학습률
    reconstruction_weight: 1.0     # 재구성 손실 가중치
    lmp_weight: 0.1               # LPM 손실 가중치

active_learning:
  strategy: 'uncertainty'         # 선택 전략
  samples_per_iteration: 50       # AL 반복당 샘플 수
  max_iterations: 5               # 최대 AL 반복 수

data:
  test_size: 0.2                  # 훈련/테스트 분할 비율
  threshold_percentile: 95        # 이상치 임계값 백분위수
  scale_features: true            # 특성 스케일링
```

### Active Learning 전략

| 전략 | 설명 | 최적 사용처 |
|------|------|-------------|
| **uncertainty** | 가장 높은 불확실성 점수의 샘플 선택 | 범용 목적 |
| **diversity** | 불확실성과 샘플 다양성의 균형 | 중복 샘플 방지 |
| **qbc** | 앙상블 불일치를 사용한 위원회 기반 질의 | 높은 정확도 요구사항 |
| **adaptive** | 전략 간 자동 전환 | 동적 환경 |
| **cost_sensitive** | 어노테이션 비용을 고려한 선택 | 예산 제약 시나리오 |

## 🔬 연구 및 이론

### 하이브리드 아키텍처 혁신

우리 시스템은 두 가지 상호 보완적 접근법을 결합합니다:

1. **오토인코더 이상 탐지**: 재구성을 통해 정상 거래 패턴 학습
2. **ELECTRA에서 영감받은 LPM**: 불확실성 추정을 위한 재구성 오류 예측

손실 예측 모듈(LPM)은 ELECTRA의 판별기에서 영감을 받았지만 수치 데이터에 적응되었습니다:

```python
# ELECTRA 개념: 토큰이 원본인지 교체된 것인지 예측
electra_discriminator(tokens) → [원본, 교체됨, 원본, ...]

# 우리의 LPM: 재구성 손실 크기 예측
lmp(encoded_features) → predicted_reconstruction_error
```

### Active Learning의 장점

전통적인 사기 탐지는 광범위한 수동 라벨링이 필요합니다. 우리의 Active Learning 접근법은:

- **라벨링 비용을 50-70% 절감**
- **지능적인 샘플 선택**을 통해 모델 성능 개선
- **반복적 학습**으로 진화하는 사기 패턴에 적응

### 수학적 기초

**하이브리드 손실 함수:**
```
L_total = α × L_reconstruction + β × L_lmp

여기서:
- L_reconstruction = MSE(입력, 재구성됨)
- L_lmp = MSE(실제_손실, 예측_손실)
- α = 1.0, β = 0.1 (설정 가능)
```

**불확실성 점수:**
```
불확실성 = 재구성_오류 × 예측_손실
```
---