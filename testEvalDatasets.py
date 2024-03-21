import torch

from evalDatasets import ValidationHvoDataset, EvaluationHvoDataset, MonotonicHvoDataset, GeneratedHvoDataset
from constants import MODELS_DIR, SF_PATH
from grooveTransformer import GrooveTransformer

from hvo_sequence.hvo_seq import HVO_Sequence

SOURCE_DIR = "preprocessedDatasets/PreProcessed_On_15_02_2024_at_16_49_hrs"
EVAL_SET_SIZE = 4
METADATA_CSV = "metadata.csv"
HVO_PICKLE = "hvo_sequence_data.obj"
AUDIO_OUT_DIR = "out/audio/test"

MODEL_PATH = f'{MODELS_DIR}/full_100e_1708034529t.pth'
MODEL = GrooveTransformer(d_model = 512, nhead = 4, dim_feedforward=16, num_layers=6, pitches=9, time_steps=32, hit_sigmoid_in_forward=False)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))


def testEvalDatasets():
    validation_set = ValidationHvoDataset(SOURCE_DIR, METADATA_CSV, HVO_PICKLE)
    
    for validation_seq in validation_set:
        assert validation_seq.hvo.shape == (32, 27), f"Validation hvo shape is invalid: {validation_seq.hvo.shape}"
    print("validation set looking good")

    eval_set = EvaluationHvoDataset(validation_set, EVAL_SET_SIZE)
    # check if eval set is valid
    assert len(eval_set) == EVAL_SET_SIZE
    assert len(eval_set.selectedIndices) == EVAL_SET_SIZE
    for eval_seq in eval_set:
        assert eval_seq.hvo.shape == (32, 27), f"Eval hvo shape is invalid: {eval_seq.hvo.shape}"
    # synthesize
    eval_set[0].save_audio(filename=f"{AUDIO_OUT_DIR}/eval_audio.wav", sf_path=SF_PATH)

    print("eval set looking good")

    monotonic_set = MonotonicHvoDataset(eval_set)
    # check if monotonic set is valid
    assert len(monotonic_set) == EVAL_SET_SIZE
    assert monotonic_set[0].master_id == eval_set[0].master_id
    assert monotonic_set[0].style_primary == eval_set[0].style_primary
    for monotonic_seq in monotonic_set:
        assert monotonic_seq.hvo.shape == (32, 27), f"Monotonic hvo shape is invalid: {monotonic_seq.hvo.shape}"
    # synthesize
    monotonic_set[0].save_audio(filename=f"{AUDIO_OUT_DIR}/monotonic_audio.wav", sf_path=SF_PATH)
    print("monotonic set looking good")

    generated_set = GeneratedHvoDataset(monotonic_set, MODEL)
    # check if generated set is valid
    assert len(generated_set) == EVAL_SET_SIZE
    assert generated_set[0].master_id == eval_set[0].master_id
    assert generated_set[0].style_primary == eval_set[0].style_primary
    for generated_seq in generated_set:
        assert generated_seq.hvo.shape == (32, 27), f"Generated hvo shape is invalid: {generated_seq.hvo.shape}"
    # synthesize
    generated_set[0].save_audio(filename=f"{AUDIO_OUT_DIR}/generated_audio.wav", sf_path=SF_PATH)
    print("generated set looking good")

    print("All sets looking good")

if __name__ == "__main__":
    testEvalDatasets()