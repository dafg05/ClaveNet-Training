import training.process as process
from pathlib import Path

PREPROCESSED_DIR = Path(__file__).parent / 'PreProcessed_On_13_02_2024_at_11_08_hrs'
# NOTE: The git repo does not contain the preprocessed data, so this test will fail unless you run the preprocessing script and place the output in the appropriate directory
PROCESSED_DIR = Path(__file__).parent / 'processed_out'

def testProcessing():
    # Clear the processed directory first
    for file in PROCESSED_DIR.iterdir():
        if file.is_file():
            file.unlink()
        else:
            for subfile in file.iterdir():
                subfile.unlink()
            file.rmdir()

    process.processing(PREPROCESSED_DIR, PROCESSED_DIR)
    print(f"Check {PROCESSED_DIR} for test results")

if __name__ == "__main__":
    testProcessing()