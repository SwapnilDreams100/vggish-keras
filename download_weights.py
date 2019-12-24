import os
import tqdm
import logging
import requests

import pkg_resources
resfile = lambda f: pkg_resources.resource_filename(__name__.split('.')[0], f)

log = logging.getLogger(__name__)

DRIVE_URL = 'https://drive.google.com/uc?id={id}&export=download'
DRIVE_CONFIRM_URL = 'https://drive.google.com/uc?id={id}&export=download&confirm={confirm}'

FILES = {
    'audioset': [
        # Weights from DTao
        # ('vggish_keras/model/audioset_top.h5',
        #  DRIVE_URL.format(id='1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6')),
        # ('vggish_keras/model/audioset_no_top.h5',
        #  DRIVE_URL.format(id='16JrWEedwaZFVZYvn1woPKCuWx85Ghzkp')),
        # ('vggish_keras/model/audioset_pca_params.npz',
        #  'https://storage.googleapis.com/audioset/vggish_pca_params.npz'),
        # ('vggish_keras/model/vggish_model.ckpt',
        #  'https://storage.googleapis.com/audioset/vggish_model.ckpt'),

        # merged weights
        (resfile('model/audioset_weights.h5'),
         '1QbMNrhu4RBUO6hIcpLqgeuVye51XyMKM')
    ],
}


def download_gdrive_file(path, gdrive_id=None):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log.info('Downloading file {} to {} ...'.format(gdrive_id, path))

        sess = requests.Session()
        r = sess.get(DRIVE_URL.format(id=gdrive_id), stream=True)

        # check for google virus message
        confirm = next(
            (v for k, v in r.cookies.get_dict().items()
             if 'download_warning_' in k), None)

        if confirm:
            log.info('Using confirmation code {}...'.format(confirm))
            r = sess.get(DRIVE_CONFIRM_URL.format(id=gdrive_id, confirm=confirm), stream=True)

        # download w/ progress bar

        chunk_size = 1024
        unit = 1024 ** 2
        with open(path, 'wb') as f:
            pbar = tqdm.tqdm(
                unit='mb', leave=False,
                total=int(r.headers.get('Content-Length', 0)) / unit or None)
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk: # filter out keep-alive new chunks
                    pbar.update(len(chunk) / unit)
                    f.write(chunk)

        log.info('Done. {} exists? {}'.format(path, os.path.isfile(path)))
    return path

def download(name='audioset'):
    return [download_gdrive_file(os.path.abspath(f), url) for f, url in FILES[name]]

if __name__ == '__main__':
    for name in FILES:
        log.info('Downloading weights for {}...'.format(name))
        download(name)
