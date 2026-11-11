"""Tải mô hình chính thức và tính vector phim một lần: python setup_ai.py."""
import os
import numpy as np
from huggingface_hub import hf_hub_download
from recommender import MovieRecommender
from semantic import (MODEL_ID, REVISION, MODEL_FILE, MODEL_DIR, INDEX_FILE,
                      Encoder, documents, fingerprint)


def main():
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in (MODEL_FILE, 'tokenizer.json', 'config.json',
                     'sentence_bert_config.json', '1_Pooling/config.json', 'README.md'):
        print('Downloading: ' + filename, flush=True)
        hf_hub_download(MODEL_ID, filename, revision=REVISION, local_dir=MODEL_DIR)
    model = MovieRecommender()
    vectors = Encoder().encode(documents(model.movies), progress=True)
    temporary = INDEX_FILE.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, vectors=vectors, fingerprint=fingerprint(model.movies))
    os.replace(temporary, INDEX_FILE)
    print(f'AI ready: {vectors.shape[0]} movies, {vectors.shape[1]} dimensions.', flush=True)


if __name__ == '__main__':
    main()
