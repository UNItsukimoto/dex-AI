import subprocess
import sys

def install_package(package):
    """パッケージを個別にインストール"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully\n")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}\n")
        return False

# インストール順序を考慮したパッケージリスト
packages = [
    # 基本パッケージ
    "numpy==1.24.3",
    "pandas==2.0.3",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    "python-dotenv==1.0.0",
    
    # ネットワーク関連
    "requests==2.31.0",
    "aiohttp==3.9.1",
    "websockets==12.0",
    
    # 機械学習（PyTorch）
    "torch==2.1.0",
    "scikit-learn==1.3.0",
    "xgboost==1.7.6",
    
    # データ処理
    "scipy==1.11.1",
    "ta==0.10.2",
    "PyWavelets==1.4.1",
    "statsmodels==0.14.0",
    
    # ユーティリティ
    "pyyaml==6.0.1",
    "colorlog==6.7.0",
    "tqdm==4.66.1",
    
    # 仮想通貨
    "eth-account==0.10.0",
    "ccxt==4.1.22"
]

failed_packages = []

print("Starting package installation...\n")

for package in packages:
    if not install_package(package):
        failed_packages.append(package)

if failed_packages:
    print("\nFailed to install:")
    for pkg in failed_packages:
        print(f"  - {pkg}")
else:
    print("\n✅ All packages installed successfully!")