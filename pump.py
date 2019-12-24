import pumpp
import params as p

def get_pump(
        sr=p.SAMPLE_RATE,
        n_mels=p.NUM_MEL_BINS):

    mel = pumpp.feature.Mel(
        name='mel', sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        log=True, conv='tf')

    return pumpp.Pump(mel)