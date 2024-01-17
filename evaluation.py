import torch
import numpy as np
import os
import mido

from hvo_processing import tools
from midiUtils import utils as mu
from grooveTransformer import GrooveTransformerModel as GT
from synthesize import synthesize_all

OUT_DIR = "out"
AUDIO_DIR = OUT_DIR + "/audio"

MIDI_DIR = "midi"

MONOTONIC_DIR = MIDI_DIR + "/monotonic"
INFERRED_DIR = MIDI_DIR + "/inferred"
SOURCE_DIR = MIDI_DIR + "/source"
MODELS_DIR = "models"

def inference(model: GT, monotonic_hvo_tensor: torch.Tensor):
    """
    Generates a full hvo tensor from a partial hvo tensor
    """
    # with model.eval():
    return model(monotonic_hvo_tensor)
    

def runInferenceOnMonotonicMidi(model: GT, monotonicMidiPath: str, outPath: str):
    """
    TODO: test
    """
    # convert midi to hvo tensor
    hvoSeq = tools.midi_to_hvo_seq(monotonicMidiPath)
    hvoArray = tools.hvo_seq_to_array(hvoSeq)
    hvoTensor = torch.from_numpy(hvoArray).float()

    # run inference
    outTensor = inference(model, hvoTensor)

    assert len(outTensor) == 1, f"Batch size of outTensor should be 1! outTensor.shape: {outTensor.shape}"
    
    # convert hvo tensor to back to midi
    outArray = outTensor[0].detach().numpy()

    assert len(outArray) == 32 and len(outArray[0]) == 27, f"Shape of out array is wrong! outArray.shape: {outArray.shape}"

    outSeq = tools.array_to_hvo_seq(outArray)
    tools.hvo_seq_to_midi(outSeq, outPath)

    print(f"Wrote midi at {outPath}")

def loadModelForInference(modelPath: str) -> GT:
    """
    Loads a model and sets it to eval mode
    """
    model = GT(d_model=8)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    return model

def transformFilesInDir(transformFunc, sourceDir: str, outDir: str, extension: str=None):
    """
    Applies a transformation function to all files in a directory
    """
    for filename in os.listdir(sourceDir):
        if extension and not filename.endswith(extension):
            continue
        sourcePath = os.path.join(sourceDir, filename)
        outPath = os.path.join(outDir, filename)
        transformFunc(sourcePath, outPath)

def audioEval():
    midExtension = ".mid"

    # from the source directory, convert all midi files to a monotonic form
    transformFilesInDir(tools.midi_to_monotonic_midi, SOURCE_DIR, MONOTONIC_DIR, midExtension)
    # trim to two bars: we do this because when converting to an hvo sequence, we could end up with a note that slightly surprasses the two-bar mark
    trimFunc = lambda sourcePath, outPath: mu.trimMidiFile(sourcePath=sourcePath, outPath=outPath, startBar=0, endBar=2, beatsPerBar=4)
    transformFilesInDir(trimFunc, MONOTONIC_DIR, MONOTONIC_DIR, midExtension)
    # synthesize monotonic dir
    synthesize_all(MONOTONIC_DIR, AUDIO_DIR, prefix="monotonic")
    
    # load model for inference
    model = loadModelForInference(MODELS_DIR + "/full_model.pth")
    # from the monotonic directory, run inference on all midi files
    inferFunc = lambda sourcePath, outPath: runInferenceOnMonotonicMidi(model, sourcePath, outPath)
    transformFilesInDir(inferFunc, MONOTONIC_DIR, INFERRED_DIR, midExtension)
    # trim again
    transformFilesInDir(trimFunc, INFERRED_DIR, INFERRED_DIR, midExtension)
    # synthesie inferred dir
    synthesize_all(INFERRED_DIR, AUDIO_DIR, prefix="inferred")

            
if __name__ == "__main__":
    audioEval()