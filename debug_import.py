import os
import sys
from pathlib import Path

print("=== Import Debug ===")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Path:")
for p in sys.path:
    print(f"  - {p}")

# src/api ディレクトリの確認
api_dir = Path("src/api")
print(f"\nChecking src/api directory:")
print(f"  Exists: {api_dir.exists()}")

if api_dir.exists():
    print(f"  Contents:")
    for file in api_dir.iterdir():
        print(f"    - {file.name}")
        
    # __init__.py の内容確認
    init_file = api_dir / "__init__.py"
    if init_file.exists():
        print(f"\n  __init__.py contents:")
        content = init_file.read_text()
        print(f"    Length: {len(content)} chars")
        print("    First 200 chars:")
        print(content[:200])
    else:
        print(f"\n  ERROR: __init__.py not found!")
        
    # hyperliquid_client.py の確認
    client_file = api_dir / "hyperliquid_client.py"
    if client_file.exists():
        print(f"\n  hyperliquid_client.py:")
        print(f"    Size: {client_file.stat().st_size} bytes")
        # クラス定義を探す
        content = client_file.read_text()
        if "class HyperliquidClient" in content:
            print("    ✓ HyperliquidClient class found")
        else:
            print("    ✗ HyperliquidClient class NOT found")
    else:
        print(f"\n  ERROR: hyperliquid_client.py not found!")

# インポートテスト
print("\n=== Import Test ===")
try:
    import src
    print("✓ src module imported")
    
    import src.api
    print("✓ src.api module imported")
    
    # 属性を確認
    print(f"  src.api attributes: {dir(src.api)}")
    
    from src.api import HyperliquidClient
    print("✓ HyperliquidClient imported successfully!")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    
    # 直接インポートを試す
    try:
        sys.path.insert(0, str(Path("src/api")))
        import hyperliquid_client
        print("✓ Direct import of hyperliquid_client worked")
        if hasattr(hyperliquid_client, 'HyperliquidClient'):
            print("✓ HyperliquidClient class exists in module")
        else:
            print("✗ HyperliquidClient class not found in module")
    except Exception as e2:
        print(f"✗ Direct import also failed: {e2}")