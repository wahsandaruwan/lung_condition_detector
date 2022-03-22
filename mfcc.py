import librosa

CYCLES_PER_SECOND = 22050

def startPreProcessing(sound_file_path, num_mfcc=13, n_fft=2048, hop_length=512):
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