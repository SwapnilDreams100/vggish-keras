import pumpp
import numpy as np

from .vggish import VGGish
from .pump import get_pump
from . import params
p = params



def get_embeddings(filename=None, y=None, sr=None, **kw):
    model, pump, sampler = get_embedding_model(**kw)

    # compute input data
    X = pump.transform(filename, y=y, sr=sr)
    X = np.stack([x[params.PUMP_INPUT][0] for x in sampler(X)], axis=0)

    # compute model outputs
    z = model.predict(X)
    ts = np.arange(len(z)) / pump.ops[0].sr * pump.ops[0].hop_length

    return ts, z

def get_embedding_model(model=None, pump=None, sampler=None,
                        duration=None, hop_duration=None,
                        include_top=None, compress=None, weights=None,):
    # make sure we have model, pump, and sampler
    pump = pump or get_pump()
    model = model or VGGish(pump, include_top=include_top, compress=compress, weights=weights)

    # get the sampler with the proper frame sizes
    if not sampler:
        # it's defined by the model top
        _, n_frames, _, _ = model.input_shape

        # or use func parameters
        op = pump['mel']
        n_frames = n_frames or op.n_frames(duration or params.EXAMPLE_WINDOW_SECONDS)
        hop_frames = op.n_frames(hop_duration or params.EXAMPLE_HOP_SECONDS)
        sampler = pumpp.SequentialSampler(
            n_frames, *pump.ops, stride=hop_frames)

    return model, pump, sampler

def get_embedding_function(*a, **kw):
    import functools
    model, pump, sampler = get_embedding_model(*a, **kw)
    return functools.partial(get_embeddings, model=model, pump=pump, sampler=sampler)
