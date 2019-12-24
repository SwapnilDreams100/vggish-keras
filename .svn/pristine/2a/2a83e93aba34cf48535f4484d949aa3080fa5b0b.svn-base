import pumpp
from . import params as p

def get_pump(
        sr=p.SAMPLE_RATE,
        n_fft_secs=p.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=p.STFT_HOP_LENGTH_SECONDS,
        n_mels=p.NUM_MEL_BINS,
        fmax=p.MEL_MAX_HZ):

    mel = pumpp.feature.Mel(
        name='mel', sr=sr,
        n_mels=n_mels,
        n_fft=int(n_fft_secs * sr),
        hop_length=int(hop_length_secs * sr),
        fmax=fmax, log=True, conv='tf')

    return pumpp.Pump(mel)
