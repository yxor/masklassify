import numpy as np
import os
import json
import librosa
import logging

def wav_to_mfccs(filepath : str, n_mfcc : int = 13) -> np.array:
    signal, sample_rate = librosa.load(filepath)

    hop_length = 512    # numbers of samples to hop
    n_fft = 2048        # number of samples in each window

    # extract mfccs
    mfccs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    return mfccs


def prepare_data(data_path : str, labels_path :str) -> dict:
    data = {
        "labels": [],
        "inputs": []
    }

    with open(labels_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            filename, label = line.split(",")
            if i % 100 == 0:
                logging.info(f"Proccessing {i}/{len(lines)}")
            full_path = os.path.join(data_path, filename)

            mfccs = wav_to_mfccs(full_path)

            data["labels"].append(int(label))
            data["inputs"].append(mfccs.T.tolist()) # transpose the mfccs so we can flatten them later
    
    return data


if __name__=="__main__":
    # setting up logging
    logging.getLogger().setLevel(logging.INFO)

    TRAINING_DATA_PATH = "./data/train/train"
    VALIDATION_DATA_PATH = "./data/validation/validation"
    TRAINING_FILE = "./data/train.txt"
    VALIDATION_FILE = "./data/validation.txt"

    # prepare the data for training and testing
    logging.info("Loading Training dataset")
    training_data = prepare_data(TRAINING_DATA_PATH, TRAINING_FILE)
    logging.info("Loading Validation dataset")
    validation_data = prepare_data(VALIDATION_DATA_PATH, VALIDATION_FILE)

    # save the files as JSON
    training_data_json = json.dumps(training_data)
    validation_data_json = json.dumps(validation_data)

    logging.info("Saving dataset")
    with open("training_0.json", "w") as f:
        f.write(training_data_json)

    with open("validation_0.json", "w") as f:
        f.write(validation_data_json)

