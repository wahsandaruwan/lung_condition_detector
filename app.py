import recording
import mfcc
import predicting
import requests
import json

# Start recording
if(recording.record_sound()):
    # Create 2 instances of the LungCondition
    lungConditionInstance1 = predicting.lungConditionPredicting()
    lungConditionInstance2 = predicting.lungConditionPredicting()

    # # Check that different instances of the LungCondition point back to the same object (singleton)
    assert lungConditionInstance1 is lungConditionInstance2

    # Extract mfcc features
    mfccFeatures = mfcc.startPreProcessing("sound.wav")

    # Make a prediction
    condition = lungConditionInstance1.startPredicting(mfccFeatures)
    print(condition)
