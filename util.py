import matplotlib.pyplot as plt
import numpy as np
import librosa


def plot_mfccs(mfccs : np.array) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_history(history):

    fig, axs = plt.subplots(2)

    # accuracy
    axs[0].plot(history.history["acc"], label="Training accuracy")
    axs[0].plot(history.history["val_acc"], label="Testing accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evolution")

    # error
    axs[1].plot(history.history["loss"], label="Training loss")
    axs[1].plot(history.history["val_loss"], label="Testing loss")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evolution")

    plt.show()
