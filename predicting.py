import tensorflow as tf
import numpy as np

AI_MODEL_PATH = "model.h5"

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


    def startPredicting(self, mfcc_features):
        """
        :param mfcc_features (n dimensional array): MFCC features extracted from lung sound file
        :return predicted_condition (str): condition predicted by the model
        """

        mfccFeatures = mfcc_features

        # We need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        mfccFeatures = mfccFeatures[np.newaxis, ..., np.newaxis]

        # Get the predicted label
        predictions = self.aiModel.predict(mfccFeatures)
        predicted_index = np.argmax(predictions)
        predicted_condition = self.lungConditions[predicted_index]
        return predicted_condition


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