# quick_fix_test.py
"""
오류 수정 확인 및 빠른 테스트 스크립트
"""

import numpy as np
import pandas as pd
import os

def test_synthetic_data_generation():
    """합성 데이터 생성 테스트"""
    print("🧪 합성 데이터 생성 테스트")
    print("=" * 40)
    
    # 테스트 파라미터
    n_normal = 1000
    n_fraud = 50
    n_features = 29
    
    print(f"설정:")
    print(f"  정상 샘플: {n_normal}")
    print(f"  사기 샘플: {n_fraud}")
    print(f"  특성 수: {n_features}")
    
    try:
        # 정상 거래 데이터 생성
        X_normal = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features) * 0.5,
            size=n_normal
        )
        
        # 사기 거래 데이터 생성
        X_fraud = np.random.multivariate_normal(
            mean=np.random.uniform(-2, 2, n_features),
            cov=np.eye(n_features) * 1.5,
            size=n_fraud
        )
        
        # 데이터 결합
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
        
        print(f"\n✅ 데이터 생성 성공:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        # 올바른 feature 이름 생성 (수정된 버전)
        feature_names = [f'V{i}' for i in range(1, n_features + 1)]
        print(f"  feature_names 길이: {len(feature_names)}")
        print(f"  X 컬럼 수: {X.shape[1]}")
        
        # DataFrame 생성
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y
        
        # Time과 Amount 추가
        df['Time'] = np.random.uniform(0, 172800, len(df))
        
        # Amount 분리 생성
        normal_amounts = np.random.lognormal(3.0, 1.2, n_normal)
        fraud_amounts = np.random.lognormal(3.5, 1.5, n_fraud)
        all_amounts = np.hstack([normal_amounts, fraud_amounts])
        df['Amount'] = all_amounts
        
        print(f"\n✅ DataFrame 생성 성공:")
        print(f"  최종 shape: {df.shape}")
        print(f"  컬럼들: {list(df.columns)[:5]}... (총 {len(df.columns)}개)")
        print(f"  사기 비율: {df['Class'].mean():.4f}")
        
        return df
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_preprocessing():
    """데이터 전처리 테스트"""
    print("\n🧪 데이터 전처리 테스트")
    print("=" * 40)
    
    df = test_synthetic_data_generation()
    if df is None:
        print("❌ 합성 데이터 생성 실패")
        return None
    
    try:
        # Time 컬럼 제거
        data = df.drop(['Time'], axis=1, errors='ignore')
        print(f"✅ Time 제거 후: {data.shape}")
        
        # Amount 스케일링
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        if 'Amount' in data.columns:
            data['Amount'] = scaler.fit_transform(data[['Amount']])
            print(f"✅ Amount 스케일링 완료")
        
        # 훈련/테스트 분할
        from sklearn.model_selection import train_test_split
        
        if 'Class' in data.columns:
            X = data.drop(['Class'], axis=1)
            y = data['Class']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"✅ 데이터 분할 완료:")
            print(f"  훈련 세트: {X_train.shape}")
            print(f"  테스트 세트: {X_test.shape}")
            print(f"  특성 수: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ 전처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_creation():
    """모델 생성 테스트"""
    print("\n🧪 모델 생성 테스트")
    print("=" * 40)
    
    preprocessing_result = test_data_preprocessing()
    if preprocessing_result is None:
        print("❌ 전처리 실패")
        return None
    
    X_train, X_test, y_train, y_test = preprocessing_result
    
    try:
        # 정상 데이터만 선택 (오토인코더용)
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]
        
        print(f"✅ 정상 데이터 선택:")
        print(f"  정상 데이터: {X_train_normal.shape}")
        print(f"  입력 차원: {X_train_normal.shape[1]}")
        
        # 간단한 오토인코더 테스트
        import tensorflow as tf
        from tensorflow.keras import layers
        
        input_dim = X_train_normal.shape[1]
        
        # 간단한 오토인코더 구조
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(14, activation='tanh')(encoder_input)
        encoded = layers.Dense(7, activation='tanh')(encoded)
        
        decoded = layers.Dense(7, activation='tanh')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = tf.keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        print(f"✅ 모델 생성 성공:")
        print(f"  입력 차원: {input_dim}")
        print(f"  인코딩 차원: [14, 7]")
        
        # 간단한 훈련 테스트
        X_train_normal_scaled = X_train_normal.values.astype(np.float32)
        history = autoencoder.fit(
            X_train_normal_scaled, X_train_normal_scaled,
            epochs=3, batch_size=32, verbose=0
        )
        
        print(f"✅ 훈련 테스트 성공:")
        print(f"  최종 손실: {history.history['loss'][-1]:.6f}")
        
        # 예측 테스트
        X_test_scaled = X_test.values.astype(np.float32)
        predictions = autoencoder.predict(X_test_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_test_scaled - predictions), axis=1)
        
        print(f"✅ 예측 테스트 성공:")
        print(f"  재구성 오류 범위: {np.min(reconstruction_errors):.6f} - {np.max(reconstruction_errors):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_system_test():
    """전체 시스템 빠른 테스트"""
    print("\n🚀 전체 시스템 빠른 테스트")
    print("=" * 50)
    
    try:
        # 수정된 fraud_detection_system 임포트 시도
        from fraud_detection_system import HybridFraudDetectionSystem
        
        # 시스템 초기화
        system = HybridFraudDetectionSystem()
        print("✅ 시스템 초기화 성공")
        
        # 작은 합성 데이터로 테스트
        df = system.load_data(
            use_synthetic=True,
            n_normal=500,
            n_fraud=25,
            n_features=29
        )
        print(f"✅ 데이터 로드 성공: {df.shape}")
        
        # 모델 구축
        system.build_model()
        print("✅ 모델 구축 성공")
        
        # 짧은 훈련
        history = system.train(epochs=5, batch_size=32)
        print("✅ 훈련 성공")
        
        # 임계값 결정
        threshold = system.determine_threshold()
        print(f"✅ 임계값 결정: {threshold:.6f}")
        
        # 평가
        results = system.evaluate_model()
        print(f"✅ 평가 완료:")
        print(f"  정확도: {results['accuracy']:.4f}")
        print(f"  AUC: {results['roc_auc']:.4f}")
        
        print("\n🎉 전체 시스템 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🔧 하이브리드 사기 탐지 시스템 - 오류 수정 테스트")
    print("=" * 60)
    
    # 단계별 테스트
    tests = [
        ("합성 데이터 생성", test_synthetic_data_generation),
        ("데이터 전처리", test_data_preprocessing),
        ("모델 생성", test_model_creation),
        ("전체 시스템", run_quick_system_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if result is not False and result is not None:
                print(f"✅ {test_name} - 통과")
                passed += 1
            else:
                print(f"❌ {test_name} - 실패")
        except Exception as e:
            print(f"❌ {test_name} - 예외 발생: {e}")
    
    print(f"\n{'='*60}")
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 시스템이 정상 작동합니다.")
        print("\n다음 명령어로 실제 실행을 시도하세요:")
        print("python fraud_detection_system.py")
    else:
        print("⚠️ 일부 테스트 실패. 추가 수정이 필요합니다.")

if __name__ == "__main__":
    # 랜덤 시드 설정
    np.random.seed(42)
    
    # 테스트 실행
    main()