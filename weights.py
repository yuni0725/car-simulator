import torch
import json
import numpy as np
import os


def convert_pth_to_json(pth_path, json_path):
    """
    .pth 파일을 JSON으로 변환합니다.

    Parameters:
        pth_path (str): .pth 파일 경로
        json_path (str): 저장할 JSON 파일 경로
    """
    try:
        state_dict = torch.load(pth_path, map_location="cpu")
        json_dict = {}

        # 만약 model_state_dict가 있다면 그걸 사용
        if "model_state_dict" in state_dict:
            print(f"model_state_dict를 발견했습니다. 모델 가중치만 변환합니다.")
            weights_to_convert = state_dict["model_state_dict"]
        else:
            # 그렇지 않다면 전체 state_dict 사용
            weights_to_convert = state_dict

        for key, value in weights_to_convert.items():
            if hasattr(value, "numpy"):
                arr = value.numpy()
                json_dict[key] = {
                    "shape": list(arr.shape),
                    "values": arr.flatten().tolist(),
                    "dtype": str(arr.dtype),
                }
            else:
                print(f"⚠️  {key}는 텐서가 아닙니다. 스킵합니다.")

        # 디렉토리가 있는 경우에만 생성
        dir_path = os.path.dirname(json_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ .pth → .json 변환 완료: {pth_path} → {json_path}")
        print(f"📊 변환된 가중치 레이어 수: {len(json_dict)}")
        return True
    except Exception as e:
        print(f"❌ .pth → .json 변환 실패: {e}")
        return False


def save_text_as_json(text: str, file_path: str):
    """
    JSON 형식의 문자열을 JSON 파일로 저장합니다.

    Parameters:
        text (str): JSON 형식의 문자열
        file_path (str): 저장할 파일 경로
    """
    try:
        data = json.loads(text)

        # 디렉토리가 있는 경우에만 생성
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON 파일 저장 완료: {file_path}")
        return True
    except json.JSONDecodeError:
        print("❌ 오류: 유효하지 않은 JSON 형식입니다.")
        return False
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False


def json_to_torch_weights(json_path):
    """
    JSON 파일을 PyTorch 가중치 형태로 변환합니다.

    Parameters:
        json_path (str): JSON 파일 경로

    Returns:
        dict: PyTorch state_dict 형태의 가중치
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state_dict = {}
        for key, item in data.items():
            shape = tuple(item["shape"])
            values = np.array(item["values"], dtype=np.float32).reshape(shape)
            state_dict[key] = torch.tensor(values)

        print(f"✅ JSON → PyTorch 가중치 변환 완료: {json_path}")
        return state_dict
    except Exception as e:
        print(f"❌ JSON → PyTorch 가중치 변환 실패: {e}")
        return None


def save_torch_weights(state_dict, pth_path):
    """
    PyTorch 가중치를 .pth 파일로 저장합니다.

    Parameters:
        state_dict (dict): PyTorch state_dict
        pth_path (str): 저장할 .pth 파일 경로
    """
    try:
        # 디렉토리가 있는 경우에만 생성
        dir_path = os.path.dirname(pth_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        torch.save(state_dict, pth_path)
        print(f"✅ PyTorch 가중치 저장 완료: {pth_path}")
        return True
    except Exception as e:
        print(f"❌ PyTorch 가중치 저장 실패: {e}")
        return False


def text_to_numpy_dict(text: str):
    """
    JSON 형식의 문자열(text)을 받아서 NumPy 딕셔너리로 변환

    Parameters:
        text (str): JSON 문자열

    Returns:
        dict[str, np.ndarray]: key → numpy 배열 dict
    """
    try:
        data = json.loads(text)

        np_weights = {}
        for key, item in data.items():
            shape = tuple(item["shape"])
            values = np.array(item["values"], dtype=np.float32).reshape(shape)
            np_weights[key] = values

        print("✅ JSON 텍스트 → NumPy 딕셔너리 변환 완료")
        return np_weights

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return None


def text_to_torch_weights(text: str):
    """
    JSON 형식의 문자열을 PyTorch 가중치로 변환합니다.

    Parameters:
        text (str): JSON 형식의 문자열

    Returns:
        dict: PyTorch state_dict 형태의 가중치
    """
    try:
        data = json.loads(text)

        state_dict = {}
        for key, item in data.items():
            shape = tuple(item["shape"])
            values = np.array(item["values"], dtype=np.float32).reshape(shape)
            state_dict[key] = torch.tensor(values)

        print("✅ JSON 텍스트 → PyTorch 가중치 변환 완료")
        return state_dict
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return None


# 완전한 워크플로 예시 함수들
def complete_workflow_example():
    """
    완전한 워크플로 예시를 보여줍니다.
    """
    print("=== 완전한 워크플로 예시 ===")

    # 1. .pth 파일을 JSON으로 변환
    pth_file = "ppo/model_epoch_100.pth"
    json_file = "ppo/model_epoch_100_weights.json"

    if os.path.exists(pth_file):
        print(f"\n1단계: .pth → .json 변환")
        convert_pth_to_json(pth_file, json_file)

        # 2. JSON 파일을 다시 PyTorch 가중치로 변환
        print(f"\n2단계: .json → PyTorch 가중치 변환")
        restored_weights = json_to_torch_weights(json_file)

        if restored_weights:
            # 3. 복원된 가중치를 새로운 .pth 파일로 저장
            print(f"\n3단계: 복원된 가중치 저장")
            save_torch_weights(restored_weights, "restored_weights.pth")

            print(f"\n✅ 워크플로 완료!")
            print(f"원본: {pth_file}")
            print(f"JSON: {json_file}")
            print(f"복원: restored_weights.pth")
    else:
        print(f"❌ 예시 파일 {pth_file}이 존재하지 않습니다.")


# 기존 코드들 (호환성 유지)
def load_weights(json_path):
    """기존 호환성을 위한 함수"""
    with open(json_path, "r") as f:
        weights = json.load(f)
    return {k: np.array(v) for k, v in weights.items()}


# 실행 예시
if __name__ == "__main__":
    # 완전한 워크플로 예시 실행

    # 개별 함수 테스트를 위한 예시
    print("\n=== 개별 함수 사용 예시 ===")

    # 예시 JSON 텍스트
    example_text = """{
      "example.layer.weight": {
        "shape": [2, 3],
        "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "dtype": "float32"
      },
      "example.layer.bias": {
        "shape": [2],
        "values": [0.01, -0.01],
        "dtype": "float32"
      }
    }"""

    print("\n1. JSON 텍스트를 파일로 저장:")
    save_text_as_json(example_text, "example_weights.json")

    print("\n2. JSON 텍스트를 PyTorch 가중치로 변환:")
    torch_weights = text_to_torch_weights(example_text)
    if torch_weights:
        for k, v in torch_weights.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

            # To use the state_dict with a model, you need to load the state_dict into the model's parameters.
            # Assuming you have a model instance, you can load the state_dict as follows:

            # Example: Load the state_dict into a model
        # model = PPOModel(input_size=92, nb_action=15)  # Create an instance of the model
        # model.load_state_dict(torch_weights)  # Load the state_dict into the model

        # Now the model's parameters are updated with the weights from the state_dict
