from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.parent / 'testData'
VALIDATION_SET_PATH = Path(__file__).parent.parent / 'AfroCuban_Validation_PreProcessed_On_03_04_2024_at_21_31_hrs'

# Files for testing
PREPROCESSED_DATASET_PATH = TEST_DATA_DIR / 'PreProcessed_On_21_03_2024_at_17_38_hrs'
# NOTE: The git repo does not contain the processed data. The easiest thing to do is to run the processing test and place the output in the appropriate directory
PROCESSED_DATASET_PATH = TEST_DATA_DIR / 'processed_at_1711223657'
# NOTE: The git repo does not contain the preprocessed data, so this test will fail unless you run the preprocessing script from the preprocessing package and place the output in the appropriate directory
MODEL_PATH = TEST_DATA_DIR / 'smol_solar-shadow_1711138656.pth'