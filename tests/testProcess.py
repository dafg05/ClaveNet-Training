from training import process

from .constants import TEST_OUT_DIR, PREPROCESSED_DATASET_PATH

PROCESSED_DIR = TEST_OUT_DIR / 'processed'

def testProcessing():
    # Clear the processed directory first
    for file in PROCESSED_DIR.iterdir():
        if file.is_file():
            file.unlink()
        else:
            for subfile in file.iterdir():
                subfile.unlink()
            file.rmdir()

    processed_dataset_path = process.processing(PREPROCESSED_DATASET_PATH, PROCESSED_DIR)
    print(f"Processing complete. Check {processed_dataset_path} for results.")

if __name__ == "__main__":
    testProcessing()
