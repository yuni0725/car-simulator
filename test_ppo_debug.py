import torch
import numpy as np
from PPO_models import PPOModel


def test_ppo_forward():
    """PPO 모델의 forward 메서드 테스트"""
    print("=== PPO Forward 메서드 디버깅 ===")

    # 1. 모델 생성
    model = PPOModel(input_size=92, nb_action=15)

    # 2. 사용자가 보여준 텐서와 유사한 입력 생성
    test_input = torch.tensor(
        [
            [
                0.3240,
                0.3260,
                0.3290,
                0.3310,
                0.3330,
                0.3360,
                0.3390,
                0.3430,
                0.3460,
                0.3490,
                0.3520,
                0.3540,
                0.3570,
                0.3610,
                0.3640,
                0.3690,
                0.3730,
                0.3770,
                0.3800,
                0.3840,
                0.3890,
                0.3930,
                0.3970,
                0.4020,
                0.4080,
                0.4130,
                0.4190,
                0.4260,
                0.4330,
                0.4410,
                0.4480,
                0.4540,
                0.4620,
                0.4700,
                0.4790,
                0.4860,
                0.4950,
                0.5080,
                0.5200,
                0.5300,
                0.5460,
                0.5570,
                0.5650,
                0.5740,
                0.6110,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.3200,
                0.3160,
                0.2960,
                0.2950,
                0.2930,
                0.2940,
                0.2950,
                0.2910,
                0.2900,
                0.2950,
                0.2960,
                0.2960,
                0.2950,
                0.2940,
                0.2940,
                0.2940,
                0.2960,
                0.2970,
                0.3000,
                0.3000,
                0.3000,
                0.3060,
                0.3090,
                0.3070,
                0.3050,
                0.3060,
                0.3110,
                0.3150,
                0.3170,
                0.3150,
                0.3170,
                0.3190,
                0.3230,
                0.3230,
                0.3240,
                0.3270,
                0.3220,
                0.3190,
                0.3210,
                0.3200,
                0.3220,
                0.0000,
                0.0000,
            ]
        ]
    )

    print(f"입력 텐서 shape: {test_input.shape}")
    print(f"입력 텐서 타입: {type(test_input)}")

    try:
        # 3. 단계별로 forward 메서드 실행
        print("\n--- 단계별 실행 ---")

        # 슬라이싱 테스트
        print(f"state[:, :-2] shape: {test_input[:, :-2].shape}")
        print(f"state[:, -2:] shape: {test_input[:, -2:].shape}")

        lidar_data = test_input[:, :-2]
        other_data = test_input[:, -2:]

        print(f"lidar_data: {lidar_data.shape}")
        print(f"other_data: {other_data.shape}")
        print(f"other_data 값: {other_data}")

        # unsqueeze 테스트
        x1 = lidar_data.unsqueeze(1)
        print(f"x1 after unsqueeze: {x1.shape}")

        # forward 메서드 전체 실행
        print("\n--- 전체 forward 실행 ---")
        logits, value = model.forward(test_input)
        print(f"✅ Forward 성공!")
        print(f"logits shape: {logits.shape}")
        print(f"value shape: {value.shape}")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


def test_different_inputs():
    """다양한 입력 형태 테스트"""
    print("\n=== 다양한 입력 형태 테스트 ===")

    model = PPOModel(input_size=92, nb_action=15)

    # 테스트 케이스들
    test_cases = [
        ("정상 입력", torch.randn(1, 92)),
        ("배치 크기 2", torch.randn(2, 92)),
        ("1D 입력", torch.randn(92)),
        ("3D 입력", torch.randn(1, 1, 92)),
    ]

    for name, test_input in test_cases:
        try:
            print(f"\n{name}: shape={test_input.shape}")
            if len(test_input.shape) == 1:
                test_input = test_input.unsqueeze(0)
                print(f"  -> 차원 추가 후: {test_input.shape}")

            logits, value = model.forward(test_input)
            print(f"  ✅ 성공: logits={logits.shape}, value={value.shape}")
        except Exception as e:
            print(f"  ❌ 실패: {e}")


if __name__ == "__main__":
    test_ppo_forward()
    test_different_inputs()
