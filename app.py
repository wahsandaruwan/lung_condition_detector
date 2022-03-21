import recording
import predicting

# Start recording
if(recording.record_sound()):
    # Create 2 instances of the LungCondition
    lungConditionInstance1 = predicting.lungConditionPredicting()
    lungConditionInstance2 = predicting.lungConditionPredicting()

    # Check that different instances of the LungCondition point back to the same object (singleton)
    assert lungConditionInstance1 is lungConditionInstance2

    # Make a prediction
    condition = lungConditionInstance1.startPredicting("sound.wav")
    print(condition)