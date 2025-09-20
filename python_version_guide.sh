# Python 버전 가이드 및 환경 설정 스크립트

# ================================
# Python 버전 권장사항
# ================================

# 최적 권장: Python 3.9.18 (LTS)
# - TensorFlow 2.10+ 완벽 호환
# - 안정성과 성능의 최적 균형
# - 모든 의존성 라이브러리 지원

# 지원 버전:
# - Python 3.9.x (권장) ⭐⭐⭐⭐⭐
# - Python 3.10.x (우수) ⭐⭐⭐⭐
# - Python 3.8.x (제한적) ⭐⭐⭐

# 비권장:
# - Python 3.11+ (TensorFlow 호환성 이슈)
# - Python 3.7 이하 (EOL)

# ================================
# 운영체제별 Python 3.9 설치
# ================================

# Ubuntu/Debian 계열
install_python39_ubuntu() {
    echo "🐧 Ubuntu/Debian에서 Python 3.9 설치 중..."
    
    # 시스템 업데이트
    sudo apt update && sudo apt upgrade -y
    
    # Python 3.9 설치
    sudo apt install -y python3.9 python3.9-pip python3.9-venv python3.9-dev
    
    # 기본 Python으로 설정 (선택사항)
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
    
    # pip 업그레이드
    python3.9 -m pip install --upgrade pip
    
    echo "✅ Python 3.9 설치 완료"
    python3.9 --version
}

# CentOS/RHEL/Rocky Linux
install_python39_centos() {
    echo "🔴 CentOS/RHEL에서 Python 3.9 설치 중..."
    
    # EPEL 저장소 활성화
    sudo yum install -y epel-release
    
    # Python 3.9 설치
    sudo yum install -y python39 python39-pip python39-devel
    
    # 심볼릭 링크 생성
    sudo ln -sf /usr/bin/python3.9 /usr/local/bin/python3
    
    echo "✅ Python 3.9 설치 완료"
    python3.9 --version
}

# macOS (Homebrew)
install_python39_macos() {
    echo "🍎 macOS에서 Python 3.9 설치 중..."
    
    # Homebrew 설치 확인
    if ! command -v brew &> /dev/null; then
        echo "Homebrew를 먼저 설치해주세요: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        return 1
    fi
    
    # Python 3.9 설치
    brew install python@3.9
    
    # PATH 설정
    echo 'export PATH="/opt/homebrew/opt/python@3.9/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    
    echo "✅ Python 3.9 설치 완료"
    python3.9 --version
}

# Windows (Chocolatey)
install_python39_windows() {
    echo "🪟 Windows에서 Python 3.9 설치 가이드:"
    echo "1. 공식 사이트에서 다운로드: https://www.python.org/downloads/release/python-3918/"
    echo "2. 또는 Chocolatey 사용:"
    echo "   choco install python --version=3.9.18"
    echo "3. 또는 Windows Store에서 'Python 3.9' 검색"
}

# ================================
# 가상환경 설정 및 의존성 설치
# ================================

setup_virtual_environment() {
    echo "🌍 가상환경 설정 중..."
    
    # Python 버전 확인
    python_version=$(python3.9 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "Python 버전: $python_version"
    
    if [[ "$python_version" < "3.8" ]]; then
        echo "❌ Python 3.8 이상이 필요합니다. 현재 버전: $python_version"
        return 1
    fi
    
    # 가상환경 생성
    python3.9 -m venv venv
    
    # 가상환경 활성화
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # pip 업그레이드
    pip install --upgrade pip setuptools wheel
    
    echo "✅ 가상환경 설정 완료"
}

# 의존성 설치 및 검증
install_dependencies() {
    echo "📦 의존성 설치 중..."
    
    # 핵심 의존성 설치
    pip install tensorflow==2.10.1  # 안정적인 TensorFlow 버전
    pip install numpy==1.21.6       # TensorFlow 2.10과 호환
    pip install pandas==1.5.3       # 안정적인 pandas
    pip install scikit-learn==1.1.3 # 호환성 확인된 버전
    
    # 시각화 라이브러리
    pip install matplotlib==3.6.3
    pip install seaborn==0.12.2
    pip install plotly==5.17.0
    
    # 기타 라이브러리
    pip install PyYAML==6.0
    pip install scipy==1.9.3
    pip install imbalanced-learn==0.9.1
    pip install tqdm==4.64.1
    
    # 개발 도구 (선택사항)
    pip install jupyter==1.0.0
    pip install notebook==6.5.2
    pip install black==22.12.0
    pip install pytest==7.2.1
    
    echo "✅ 의존성 설치 완료"
}

# 환경 검증
verify_environment() {
    echo "🔍 환경 검증 중..."
    
    # Python 버전 확인
    python_version=$(python --version 2>&1)
    echo "Python: $python_version"
    
    # 핵심 라이브러리 임포트 테스트
    python -c "
import sys
print(f'Python 버전: {sys.version}')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow: {tf.__version__}')
    
    # GPU 사용 가능 여부 확인
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'✅ GPU 감지: {len(gpus)}개')
        for gpu in gpus:
            print(f'   - {gpu.name}')
    else:
        print('ℹ️  CPU 모드로 실행됩니다')
        
except ImportError as e:
    print(f'❌ TensorFlow 임포트 실패: {e}')
    exit(1)

try:
    import numpy as np
    import pandas as pd
    import sklearn
    print(f'✅ NumPy: {np.__version__}')
    print(f'✅ Pandas: {pd.__version__}')
    print(f'✅ Scikit-learn: {sklearn.__version__}')
except ImportError as e:
    print(f'❌ 핵심 라이브러리 임포트 실패: {e}')
    exit(1)

try:
    import matplotlib
    import seaborn
    import plotly
    print(f'✅ 시각화 라이브러리 준비 완료')
except ImportError as e:
    print(f'⚠️ 시각화 라이브러리 문제: {e}')

print('🎉 모든 환경 검증 완료!')
"
}

# 성능 최적화 설정
optimize_performance() {
    echo "⚡ 성능 최적화 설정 중..."
    
    # 환경 변수 설정
    export TF_CPP_MIN_LOG_LEVEL=2  # TensorFlow 경고 최소화
    export PYTHONHASHSEED=42       # 재현 가능한 결과
    
    # .bashrc 또는 .zshrc에 추가
    shell_rc="${HOME}/.bashrc"
    if [[ "$SHELL" == *"zsh"* ]]; then
        shell_rc="${HOME}/.zshrc"
    fi
    
    echo "# Hybrid Fraud Detection 환경 변수" >> "$shell_rc"
    echo "export TF_CPP_MIN_LOG_LEVEL=2" >> "$shell_rc"
    echo "export PYTHONHASHSEED=42" >> "$shell_rc"
    
    # GPU 메모리 증가 설정 (GPU 사용시)
    python -c "
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✅ GPU 메모리 증가 설정 완료')
    except:
        print('⚠️ GPU 설정 중 오류 발생')
else:
    print('ℹ️ GPU가 감지되지 않음')
"
    
    echo "✅ 성능 최적화 완료"
}

# ================================
# 메인 설치 스크립트
# ================================

main_setup() {
    echo "🚀 하이브리드 사기 탐지 시스템 환경 설정을 시작합니다..."
    
    # OS 감지
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            install_python39_ubuntu
        elif command -v yum &> /dev/null; then
            install_python39_centos
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        install_python39_macos
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        install_python39_windows
        return
    fi
    
    # 가상환경 및 의존성 설치
    setup_virtual_environment
    install_dependencies
    verify_environment
    optimize_performance
    
    echo "🎉 설치 완료! 다음 명령어로 프로젝트를 시작하세요:"
    echo "source venv/bin/activate  # 가상환경 활성화"
    echo "python fraud_detection_system.py  # 시스템 실행"
}

# ================================
# 문제 해결 가이드
# ================================

troubleshoot() {
    echo "🔧 일반적인 문제 해결 가이드:"
    echo ""
    
    echo "❌ TensorFlow 설치 실패:"
    echo "   해결: pip install --upgrade tensorflow==2.10.1"
    echo "   M1 Mac: pip install tensorflow-macos tensorflow-metal"
    echo ""
    
    echo "❌ NumPy 호환성 문제:"
    echo "   해결: pip install numpy==1.21.6"
    echo ""
    
    echo "❌ GPU 인식 안됨:"
    echo "   해결: pip install tensorflow-gpu==2.10.1"
    echo "   CUDA 11.2 및 cuDNN 8.1 설치 필요"
    echo ""
    
    echo "❌ 메모리 부족 오류:"
    echo "   해결: batch_size를 줄이거나 GPU 메모리 증가 설정"
    echo ""
    
    echo "❌ 의존성 충돌:"
    echo "   해결: pip install --force-reinstall -r requirements.txt"
}

# ================================
# 버전별 상세 정보
# ================================

version_details() {
    cat << EOF
🐍 Python 버전별 상세 정보

┌─────────────┬─────────────┬─────────────────┬──────────────┬─────────────────┐
│ Python      │ TensorFlow  │ 안정성          │ 성능         │ 권장도          │
├─────────────┼─────────────┼─────────────────┼──────────────┼─────────────────┤
│ 3.9.18      │ 2.10-2.13   │ 매우 높음       │ 최적         │ ⭐⭐⭐⭐⭐      │
│ 3.10.12     │ 2.10-2.13   │ 높음           │ 우수         │ ⭐⭐⭐⭐        │
│ 3.8.18      │ 2.8-2.11    │ 높음           │ 양호         │ ⭐⭐⭐          │
│ 3.11.x      │ 제한적      │ 불안정         │ 미지원       │ ❌              │
└─────────────┴─────────────┴─────────────────┴──────────────┴─────────────────┘

상세 권장사항:

🏆 Python 3.9.18 (최고 권장)
   - LTS 버전으로 장기 지원
   - TensorFlow 2.10+ 완벽 호환
   - 모든 ML 라이브러리 안정적 지원
   - 메모리 효율성 최적화

✅ Python 3.10.12 (권장)
   - 최신 언어 기능 지원
   - 대부분의 라이브러리 호환
   - 약간의 호환성 문제 가능성

⚠️  Python 3.8.18 (제한적 권장)
   - 구형 시스템용
   - 일부 최신 기능 미지원
   - TensorFlow 2.11+ 지원 제한

❌ Python 3.11+ (비권장)
   - TensorFlow 호환성 문제
   - 일부 ML 라이브러리 미지원
   - 안정성 이슈

EOF
}

# ================================
# 도커 환경 설정
# ================================

setup_docker_environment() {
    cat << EOF > Dockerfile.python39
# Python 3.9 기반 최적화된 Dockerfile
FROM python:3.9.18-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \\
    gcc g++ \\
    libhdf5-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 요구사항 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==23.3.2
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# 포트 노출
EXPOSE 8888 8000

# 기본 명령어
CMD ["python", "fraud_detection_system.py"]
EOF

    echo "✅ Python 3.9 최적화 Dockerfile 생성됨: Dockerfile.python39"
}

# ================================
# 실행 옵션
# ================================

case "${1:-setup}" in
    "setup")
        main_setup
        ;;
    "ubuntu")
        install_python39_ubuntu
        ;;
    "centos")
        install_python39_centos
        ;;
    "macos")
        install_python39_macos
        ;;
    "windows")
        install_python39_windows
        ;;
    "venv")
        setup_virtual_environment
        install_dependencies
        ;;
    "verify")
        verify_environment
        ;;
    "troubleshoot")
        troubleshoot
        ;;
    "versions")
        version_details
        ;;
    "docker")
        setup_docker_environment
        ;;
    *)
        echo "사용법: $0 [setup|ubuntu|centos|macos|windows|venv|verify|troubleshoot|versions|docker]"
        echo ""
        echo "옵션:"
        echo "  setup        - 전체 환경 설정 (기본값)"
        echo "  ubuntu       - Ubuntu/Debian Python 3.9 설치"
        echo "  centos       - CentOS/RHEL Python 3.9 설치"
        echo "  macos        - macOS Python 3.9 설치"
        echo "  windows      - Windows 설치 가이드"
        echo "  venv         - 가상환경 및 의존성만 설치"
        echo "  verify       - 환경 검증"
        echo "  troubleshoot - 문제 해결 가이드"
        echo "  versions     - 버전별 상세 정보"
        echo "  docker       - Docker 환경 설정"
        ;;
esac

# ================================
# requirements.txt (Python 3.9 최적화)
# ================================

cat << 'EOF' > requirements_python39.txt
# Python 3.9 최적화 의존성 목록
# 하이브리드 사기 탐지 시스템용

# 핵심 ML 프레임워크 (Python 3.9 최적화)
tensorflow==2.10.1
numpy==1.21.6
pandas==1.5.3
scikit-learn==1.1.3

# 시각화 라이브러리
matplotlib==3.6.3
seaborn==0.12.2
plotly==5.17.0

# 데이터 처리
scipy==1.9.3
PyYAML==6.0.1

# 불균형 데이터 처리
imbalanced-learn==0.9.1

# 유틸리티
tqdm==4.64.1
joblib==1.2.0

# Jupyter 환경
jupyter==1.0.0
notebook==6.5.2
ipywidgets==8.0.4

# 개발 도구
pytest==7.2.1
black==22.12.0
flake8==6.0.0
isort==5.11.4
mypy==0.991

# 배포 및 서빙
uvicorn==0.20.0
fastapi==0.88.0

# 성능 최적화
numba==0.56.4  # 수치 계산 가속
Cython==0.29.33  # C 확장 지원

# 보안
cryptography==38.0.4

# 메모리 프로파일링
memory-profiler==0.61.0
EOF

echo "✅ Python 3.9 최적화 requirements.txt 생성됨: requirements_python39.txt"

# ================================
# 환경 설정 검증 스크립트
# ================================

cat << 'EOF' > verify_setup.py
#!/usr/bin/env python3
"""
하이브리드 사기 탐지 시스템 환경 검증 스크립트
Python 3.9 최적화 확인
"""

import sys
import subprocess
import importlib
from packaging import version

def check_python_version():
    """Python 버전 확인"""
    py_version = sys.version_info
    version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    
    print(f"🐍 Python 버전: {version_str}")
    
    if py_version.major != 3:
        print("❌ Python 3가 필요합니다")
        return False
    
    if py_version.minor == 9:
        print("✅ Python 3.9 - 최적 버전!")
        return True
    elif py_version.minor == 10:
        print("✅ Python 3.10 - 권장 버전")
        return True
    elif py_version.minor == 8:
        print("⚠️ Python 3.8 - 지원하지만 업그레이드 권장")
        return True
    else:
        print(f"⚠️ Python {version_str} - 테스트되지 않은 버전")
        return False

def check_package(package_name, min_version=None):
    """패키지 설치 및 버전 확인"""
    try:
        module = importlib.import_module(package_name)
        pkg_version = getattr(module, '__version__', 'Unknown')
        
        if min_version and pkg_version != 'Unknown':
            if version.parse(pkg_version) < version.parse(min_version):
                print(f"⚠️ {package_name}: {pkg_version} (최소 {min_version} 권장)")
                return False
        
        print(f"✅ {package_name}: {pkg_version}")
        return True
        
    except ImportError:
        print(f"❌ {package_name}: 설치되지 않음")
        return False

def check_tensorflow_gpu():
    """TensorFlow GPU 지원 확인"""
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU 감지: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                
            # GPU 메모리 증가 설정
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            return True
        else:
            print("💻 CPU 모드로 실행됩니다")
            return False
            
    except Exception as e:
        print(f"⚠️ GPU 확인 중 오류: {e}")
        return False

def performance_benchmark():
    """간단한 성능 벤치마크"""
    import time
    import numpy as np
    
    print("\n⚡ 성능 벤치마크 실행 중...")
    
    # NumPy 벤치마크
    start_time = time.time()
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    numpy_time = time.time() - start_time
    
    print(f"NumPy 행렬 곱셈 (1000x1000): {numpy_time:.3f}초")
    
    # TensorFlow 벤치마크
    try:
        import tensorflow as tf
        start_time = time.time()
        x = tf.random.normal([1000, 1000])
        y = tf.random.normal([1000, 1000])
        z = tf.matmul(x, y)
        tf_time = time.time() - start_time
        
        print(f"TensorFlow 행렬 곱셈 (1000x1000): {tf_time:.3f}초")
        
        if tf_time < numpy_time:
            print("🚀 TensorFlow가 NumPy보다 빠릅니다!")
        else:
            print("💡 CPU 최적화를 고려해보세요")
            
    except Exception as e:
        print(f"TensorFlow 벤치마크 실패: {e}")

def main():
    """메인 검증 함수"""
    print("🔍 하이브리드 사기 탐지 시스템 환경 검증")
    print("=" * 50)
    
    # Python 버전 확인
    python_ok = check_python_version()
    
    print("\n📦 패키지 확인:")
    packages = [
        ("tensorflow", "2.10.0"),
        ("numpy", "1.21.0"),
        ("pandas", "1.5.0"),
        ("sklearn", "1.1.0"),
        ("matplotlib", "3.6.0"),
        ("seaborn", "0.12.0"),
        ("yaml", None),
        ("scipy", "1.9.0"),
    ]
    
    all_packages_ok = True
    for package_name, min_ver in packages:
        if package_name == "sklearn":
            package_name = "sklearn"
        elif package_name == "yaml":
            package_name = "yaml"
        
        if not check_package(package_name, min_ver):
            all_packages_ok = False
    
    print("\n🚀 GPU 확인:")
    gpu_available = check_tensorflow_gpu()
    
    if python_ok and all_packages_ok:
        performance_benchmark()
        
        print("\n" + "=" * 50)
        print("🎉 환경 검증 완료!")
        print("✅ 하이브리드 사기 탐지 시스템을 실행할 준비가 되었습니다.")
        print("\n다음 명령어로 시스템을 시작하세요:")
        print("python fraud_detection_system.py")
    else:
        print("\n" + "=" * 50)
        print("❌ 환경 설정에 문제가 있습니다.")
        print("requirements_python39.txt를 사용하여 의존성을 재설치하세요:")
        print("pip install -r requirements_python39.txt")

if __name__ == "__main__":
    main()
EOF

chmod +x verify_setup.py
echo "✅ 환경 검증 스크립트 생성됨: verify_setup.py"

echo ""
echo "🎯 Python 3.9 환경 설정 가이드가 생성되었습니다!"
echo "실행하려면: bash $(basename $0) setup"