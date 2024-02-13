import torch
import numpy as np
import os
import mido
import sys
import json

from hvo_processing import converters
from midiUtils import utils as mu
from midiUtils.synthesize import synthesize_all
from grooveTransformer import GrooveTransformer as GT
from constants import *

FULL = True

def runInferenceOnMonotonicMidi(model: GT, monotonicMidiPath: str, outPath: str):
    # convert midi to hvo tensor
    hvoSeq = converters.midi_to_hvo_seq(monotonicMidiPath)
    hvoArray = converters.hvo_seq_to_array(hvoSeq)
    hvoArray = converters.pad_hvo_timesteps(hvoArray, 32)
    hvoTensor = torch.from_numpy(hvoArray).float()

    # run inference
    h, v, o = model.inference(hvoTensor)
    outTensor = torch.cat((h, v, o), dim=2)

    assert len(outTensor) == 1, f"Batch size of outTensor should be 1! outTensor.shape: {outTensor.shape}"
    
    # convert hvo tensor to back to midi
    outArray = outTensor[0].detach().numpy()
    assert len(outArray) == 32 and len(outArray[0]) == 27, f"Shape of out array is wrong! outArray.shape: {outArray.shape}"
    outSeq = converters.array_to_hvo_seq(outArray)
    converters.hvo_seq_to_midi(outSeq, outPath)

def loadModel(modelPath: str, hypersDict: dict) -> GT:
    """
    Loads model from path and hypersDict
    """
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model at {modelPath} does not exist!")

    d_model = hypersDict["d_model"]
    dim_forward = hypersDict["dim_forward"]
    n_heads = hypersDict["n_heads"]
    n_layers = hypersDict["n_layers"]
    
    model = GT(d_model=d_model, nhead = n_heads, dim_feedforward=dim_forward, num_layers=n_layers, pitches=9, time_steps=32, hit_sigmoid_in_forward=False)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    return model

def getModelPath(start_time,full=True, epochs=100):
    """
    Returns the path to the model
    """
    size_str = "full" if full else "smol"
    return f'{MODELS_DIR}/{size_str}_{epochs}e_{start_time}t.pth'

def getModelPathFromStartTimeAndHypers(start_time: int, hypersDict: dict) -> str:
    """
    Returns the path to the model
    """
    dataAug = hypersDict["data_augmentation"]
    epochs = hypersDict["epochs"]
    # TODO: don't hardcode the seed
    return getModelPath(start_time, FULL, epochs)

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

def clearDir(dir):
    """
    Clears a directory
    """
    for filename in os.listdir(dir):
        filePath = os.path.join(dir, filename)
        try:
            os.remove(filePath)
        except Exception as e:
            print(f"Something went wrong when removing {filePath}.")
            continue

def clearDirsForAudioEval():
    # clear monotonic, inferred and audio dirs
    clearDir(MONOTONIC_DIR)
    clearDir(INFERRED_DIR)
    clearDir(AUDIO_DIR)
    

def audioEval(model: GT):

    clearDirsForAudioEval()

    midExtension = ".mid"

    synthesize_all(SOURCE_DIR, AUDIO_DIR, prefix="source")
    # from the source directory, convert all midi files to a monotonic form
    transformFilesInDir(converters.midi_to_monotonic_midi, SOURCE_DIR, MONOTONIC_DIR, midExtension)
    # trim to two bars. otherwise, we could get incompatible shapes when running inference
    trimFunc = lambda sourcePath, outPath: mu.trimMidiFile(sourcePath=sourcePath, outPath=outPath, startBar=0, endBar=2, beatsPerBar=4)
    transformFilesInDir(trimFunc, MONOTONIC_DIR, MONOTONIC_DIR, midExtension)
    synthesize_all(MONOTONIC_DIR, AUDIO_DIR, prefix="monotonic")

    # run inference on all monotonic midis
    inferFunc = lambda sourcePath, outPath: runInferenceOnMonotonicMidi(model, sourcePath, outPath)
    transformFilesInDir(inferFunc, MONOTONIC_DIR, INFERRED_DIR, midExtension)
    # need to trim again
    transformFilesInDir(trimFunc, INFERRED_DIR, INFERRED_DIR, midExtension)
    synthesize_all(INFERRED_DIR, AUDIO_DIR, prefix="inferred")        

def usage_and_exit():
    print("Usage: python evaluation.py <eval_type> <modelStartTime> <hypersFile>")
    print("eval_type: 'audio'")
    print("modelStartTime: 'used to id the model'")
    print("hypersFile: 'json file containing hyperparameters'")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage_and_exit()
    
    eval_type = sys.argv[1]
    modelStartTime = int(sys.argv[2])
    hypersFile = sys.argv[3]
    hypersPath = f'{HYPERS_DIR}/{hypersFile}'
    if eval_type == "audio":
        with open(hypersPath) as hp:
            hypersDict = json.load(hp)
            modelPath = getModelPathFromStartTimeAndHypers(modelStartTime, hypersDict)
            model = loadModel(modelPath, hypersDict)
            audioEval(model)
    else:
        usage_and_exit()