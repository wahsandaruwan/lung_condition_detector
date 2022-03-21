import sounddevice
from scipy.io.wavfile import write

# Sample rate
cps = 22050
# Clip duration
length = 10

def record_sound():
    """
    For recording a lung sound clip using intergrated microphone
    """
    print("Recording started!")

    # Record audio data from your sound device into a NumPy array
    recording = sounddevice.rec(int(length*cps), samplerate=cps, channels=2)
    # Check if the recording is finished
    sounddevice.wait()

    print("Recording stopped!")

    # Write a NumPy array as a WAV file.
    write("sound.wav", cps, recording)

    return True