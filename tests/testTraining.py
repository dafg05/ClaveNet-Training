from training import training
from tests.constants import TEST_OUT_DIR, PROCESSED_DATASET_PATH

MODEL_OUT = TEST_OUT_DIR / 'model'

HYPERPARAMS_SETTING = 'solar-shadow'
LOG_WANDB = False
IS_SMOL = True

def testTrain():
    # clear the model directory first
    for file in MODEL_OUT.iterdir():
        if file.is_file():
            file.unlink()
        else:
            for subfile in file.iterdir():
                subfile.unlink()
            file.rmdir()
    
    model_path = training.train(HYPERPARAMS_SETTING, PROCESSED_DATASET_PATH, MODEL_OUT, LOG_WANDB, IS_SMOL)
    print(f"Training complete. Check {model_path} for results.")

if __name__ == "__main__":
    testTrain()
