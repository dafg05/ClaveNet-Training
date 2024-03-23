import training.training as training
from pathlib import Path

HYPERPARAMS_SETTING = 'solar-shadow'
PROCESSED_DATASET_PATH = Path(__file__).parent / 'processed_at_1711223657'
# NOTE: The git repo does not contain the processed data, so this test will fail unless you run the processing script and place the output in the appropriate directory
MODEL_DIR = Path(__file__).parent / 'model_out'
LOG_WANDB = True
IS_SMOL = True

def testTrain():
    # clear the model directory first
    for file in MODEL_DIR.iterdir():
        if file.is_file():
            file.unlink()
        else:
            for subfile in file.iterdir():
                subfile.unlink()
            file.rmdir()
    
    model_path = training.train(HYPERPARAMS_SETTING, PROCESSED_DATASET_PATH, MODEL_DIR, LOG_WANDB, IS_SMOL)
    print(f"Training complete. Check {model_path} for results.")

if __name__ == "__main__":
    testTrain()