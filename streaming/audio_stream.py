import sounddevice as sd
import numpy as np
import librosa

class AudioStream:
    def __init__(self, sample_rate=16000, duration=2):
        self.sample_rate = sample_rate
        self.duration = duration
        self.buffer = None

    def record(self):
        print("Recording audio...")
        audio = sd.rec(int(self.sample_rate * self.duration),
                       samplerate=self.sample_rate, channels=1)
        sd.wait()
        audio = np.squeeze(audio)
        self.buffer = audio
        return audio

    def extract_features(self):
        if self.buffer is None:
            return None
        mfcc = librosa.feature.mfcc(y=self.buffer,
                                    sr=self.sample_rate,
                                    n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
