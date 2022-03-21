import librosa
import tensorflow as tf
import numpy as np

AI_MODEL_PATH = "model.h5"
CYCLES_PER_SECOND = 22050

class LungCondition:
    """
    Singleton class for Lung Condition inference with trained AI Model.
    """

    aiModel = None
    lungConditions = [
        "Asthma",
        "COPD",
        "Healthy"
    ]
    lungConditionInstance = None


    def startPredicting(self, sound_file_path):
        """
        :param sound_file_path (str): Path to lung sound file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # Extract mfcc features
        mfccFeatures = self.startPreProcessing(sound_file_path)

        # We need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        mfccFeatures = mfccFeatures[np.newaxis, ..., np.newaxis]

        # Get the predicted label
        predictions = self.aiModel.predict(mfccFeatures)
        predicted_index = np.argmax(predictions)
        predicted_condition = self.lungConditions[predicted_index]
        return predicted_condition


    def startPreProcessing(self, sound_file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load sound file
        signal, sampleRate = librosa.load(sound_file_path)

        if len(signal) >= CYCLES_PER_SECOND:
            # Ensure consistency of the length of the signal
            signal = signal[:CYCLES_PER_SECOND]

            # Extract mfcc features
            mfccFeatures = librosa.feature.mfcc(signal, sampleRate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return mfccFeatures.T


def lungConditionPredicting():
    """
    Factory function for LungCondition class.
    :return LungConditon.lungConditionInstance (LungConditon):
    """

    # Ensure an instance is created only the first time the factory function is called
    if LungCondition.lungConditionInstance is None:
        LungCondition.lungConditionInstance = LungCondition()
        LungCondition.aiModel = tf.keras.models.load_model(AI_MODEL_PATH)
    return LungCondition.lungConditionInstance