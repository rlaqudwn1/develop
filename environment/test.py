import sys

print("=" * 60)
print("🐍 upstage312_gpu 환경 설치 검사를 시작합니다...")
print(f"현재 파이썬 실행 파일 경로:\n{sys.executable}\n")
print("=" * 60)
print("1. 필수 패키지 임포트 검사:\n")

# yml 파일 기준 핵심 패키지 목록
packages_to_check = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "transformers": "transformers",
    "datasets": "datasets",
    "wandb": "wandb",
    "torch": "torch",
}


failed_packages = []

for pkg_name, import_name in packages_to_check.items():
    try:
        __import__(import_name)
        print(f"✅ [SUCCESS] '{pkg_name}' 패키지를 성공적으로 불러왔습니다.")
    except ImportError:
        print(f"❌ [FAILURE] '{pkg_name}' 패키지를 찾을 수 없습니다. (Not installed)")
        failed_packages.append(pkg_name)
    except Exception as e:
        print(f"❌ [FAILURE] '{pkg_name}' 패키지 임포트 중 오류: {e}")
        failed_packages.append(pkg_name)

print("\n" + "=" * 60)
print("2. PyTorch 및 GPU (CUDA) 호환성 검사:\n")

# PyTorch + CUDA 검사
try:
    import torch

    print(f"✅ [INFO] PyTorch 버전: {torch.__version__}")

    # 가장 중요한 검사
    is_cuda_available = torch.cuda.is_available()
    print(f"✅ [INFO] CUDA 사용 가능 여부: {is_cuda_available}")

    if is_cuda_available:
        print(f"✅ [INFO] 인식된 GPU 개수: {torch.cuda.device_count()}개")
        print(f"✅ [INFO] 현재 GPU 이름: {torch.cuda.get_device_name(0)}")
        print("\n🎉 [최종 성공] PyTorch가 GPU(CUDA)를 성공적으로 인식했습니다!")
    else:
        print("\n❌ [최종 실패] PyTorch가 GPU(CUDA)를 인식하지 못합니다.")
        print("   torch.cuda.is_available()가 'False'입니다.")
        print("   NVIDIA 드라이버 또는 PyTorch CUDA 버전을 확인하세요.")
        failed_packages.append("torch (CUDA)")

except ImportError:
    print("❌ [CRITICAL FAILURE] 'torch' (PyTorch)가 설치되지 않았습니다.")
    failed_packages.append("torch (Not Found)")
except Exception as e:
    print(f"❌ [CRITICAL FAILURE] PyTorch 검사 중 알 수 없는 오류 발생: {e}")
    failed_packages.append(f"torch (Error: {e})")

print("=" * 60)
if not failed_packages:
    print(
        "\n>> 종합 결과: 🟢 모든 검사를 통과했습니다. 환경이 올바르게 설정되었습니다."
    )
else:
    print(
        f"\n>> 종합 결과: 🔴 일부 검사에 실패했습니다. (실패 항목: {', '.join(failed_packages)})"
    )
    print("   아래 '해결책'을 따라 환경을 업데이트하세요.")
print("=" * 60)
