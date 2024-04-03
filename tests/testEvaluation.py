import evaluation.evaluation as eval

from pathlib import Path

OUT_DIR = Path('tests','evaluation_out')
MODEL_PATH = Path('tests','smol_solar-shadow_1711138656.pth')
VALIDATION_SET_PATH = Path('tests','AfroCuban_Validation_PreProcessed_On_03_04_2024_at_01_04_hrs')
SYNTHESIZE_UP_TO = 5

def test_evaluate_model():
    eval_time = eval.evaluateModel(OUT_DIR, MODEL_PATH, VALIDATION_SET_PATH, SYNTHESIZE_UP_TO)
    print(f"Output evaluation saved to {OUT_DIR}. Evaltime: {eval_time}")

if __name__ == "__main__":
    test_evaluate_model()