from ..evaluation import evaluation as eval
from .constants import MODEL_PATH, TEST_DATA_DIR, VALIDATION_SET_PATH

from pathlib import Path

# relevant paths
OUT_DIR = TEST_DATA_DIR / 'evaluation_out'

SYNTHESIZE_UP_TO = 5

def test_evaluate_model():
    eval_time = eval.evaluateModel(OUT_DIR, MODEL_PATH, VALIDATION_SET_PATH, SYNTHESIZE_UP_TO)
    print(f"Output evaluation saved to {OUT_DIR}. Evaltime: {eval_time}")

if __name__ == "__main__":
    test_evaluate_model()