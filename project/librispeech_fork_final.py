"""
librispeech.py

this is a fork of the default librispeech module, with some modifications done to it's class loader.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor, tensor, float32
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
import torchaudio
import random
import librosa
from sklearn.preprocessing import scale
from enum import Enum
from typing import Literal, Optional

# make noises
# ---------------------------------------------------------------------------------------
# dirty_waveform = tensor(make_second_noise(make_first_noise(dirty_waveform)))


def make_first_noise(wave):
    return wave * (np.random.uniform(0.6, 0.9, size=wave.shape) + np.random.choice([0, 0.5], size=wave.shape))


def make_second_noise(wave):
    # here we add some White Noise using normal distribution,
    # the mean is 0, the variance is chosen random- it's the amplitude of the signal,
    # multiplied randomly by a value from the interval (0.1,0.3)
    # and then it is normalized to be with the norm of the original wave.
    samples = 0.6 * np.random.normal(0, np.max(np.abs(wave)) * np.random.uniform(0.06, 0.1), size=wave.shape)
    return (wave + samples) * np.linalg.norm(wave) / np.linalg.norm(wave + samples)


def add_2_sounds(wave1, wave2, scale_original=1):  # adding wave1 to wave 2
    wave1_copy = scale_original * np.copy(wave1)
    shape1 = wave1.shape[0]
    shape2 = wave2.shape[0]

    amplitude = np.random.uniform(0.15, 0.3)
    if shape1 < shape2:
        return wave1_copy + amplitude * wave2[:wave1.shape[0], :]
    if shape1 == shape2:
        return wave1_copy + amplitude * wave2
    residual = shape1
    for i in range(shape1 // shape2):
        residual -= shape2
        wave1_copy[i * shape2:(i + 1) * shape2, :] += amplitude * wave2
    wave1_copy[shape1 - residual:shape1, :] += amplitude * wave2[:residual, :]
    return wave1_copy


def add_3_sounds(wave1, wave2, wave3):
    first_add_2 = add_2_sounds(wave1, wave2, np.random.uniform(1.05, 1.15))
    return add_2_sounds(first_add_2, wave3)


def make_third_noise(wave1, wave2, wave3):
    return add_3_sounds(wave1, wave2, wave3)
# ---------------------------------------------------------------------------------------

# make spectrogram
# ---------------------------------------------------------------------------------------


def create_spectrogram(save_path, audio, sr=16000, hop_length=512, n_fft=2048, n_mels=256):
    # STFT with 75% overlap
    window = np.blackman(n_fft)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, center=True)

    # Convert to mel and back
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=True)
    mel_spec = np.dot(mel_basis, np.abs(D))
    phase = np.angle(D)

    return mel_spec, {"phase": phase[:, :400], "mel_basis": mel_basis, "spec_shape": (mel_spec.shape[0], 400)}


def create_spectrogram_stft(save_path, audio, sr=16000, hop_length=512, n_fft=2048, n_mels=256):
    mfcc = librosa.feature.mfcc(y=audio.squeeze(), sr=sr)
    # mfcc = scale(mfcc, axis=1)

    return mfcc, {"phase": [], "mel_basis": [], "spec_shape": (0, 400)}

# ---------------------------------------------------------------------------------------


URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"
SAMPLE_RATE = 16000
_DATA_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",  # noqa: E501
    "http://www.openslr.org/resources/12/dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",  # noqa: E501
    "http://www.openslr.org/resources/12/test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",  # noqa: E501
    "http://www.openslr.org/resources/12/test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",  # noqa: E501
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",  # noqa: E501
}


def _download_librispeech(root, url):
    base_url = "http://www.openslr.org/resources/12/"
    ext_archive = ".tar.gz"

    filename = url + ext_archive
    archive = os.path.join(root, filename)
    download_url = os.path.join(base_url, filename)
    if not os.path.isfile(archive):
        checksum = _CHECKSUMS.get(download_url, None)
        download_url_to_file(download_url, archive, hash_prefix=checksum)
    _extract_tar(archive)


def _get_librispeech_metadata(
    fileid: str, root: str, folder: str, ext_audio: str, ext_txt: str
) -> Tuple[str, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    # Get audio path and sample rate
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    filepath = os.path.join(folder, speaker_id, chapter_id, f"{fileid_audio}{ext_audio}")

    # Load text
    file_text = f"{speaker_id}-{chapter_id}{ext_txt}"
    file_text = os.path.join(root, folder, speaker_id, chapter_id, file_text)
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")

    return (
        filepath,
        SAMPLE_RATE,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


class Mode(Enum):
    X = 'x'
    Y = 'y'


class LIBRISPEECH(Dataset):
    """*LibriSpeech* :cite:`7178964` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        mode (Literal[x, y], optional):
            x will load only dirty sounds, y will load only clean data. (default: ``None``).
        limit (int, optional):
            Whether to limit the amount of sounds in the dataset. (default: ``4000``). pass ``None`` to load the full dataset
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        mode: Optional[Literal['x', 'y', None]] = None,
        limit: Optional[int] = 4000,
    ) -> None:
        self.limit = limit
        self.mode = mode
        self._url = url
        if url not in _DATA_SUBSETS:
            raise ValueError(f"Invalid url '{url}' given; please provide one of {_DATA_SUBSETS}.")

        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)
        self._path = os.path.join(root, folder_in_archive, url)

        if not os.path.isdir(self._path):
            if download:
                _download_librispeech(root, url)
            else:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. Please set `download=True` to download the dataset."
                )

        self.walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio) if not p.stem.endswith('.dirty'))
        if self.limit:
            self.walker = self.walker[:self.limit]

    def get_metadata(self, n: int) -> Tuple[str, int, str, int, int, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        fileid = self.walker[n]
        return _get_librispeech_metadata(fileid, self._archive, self._url, self._ext_audio, self._ext_txt)

    def find_two_other_flac_files(self, file_path, base_noise):
        directory = os.path.dirname(file_path)
        current_filename = os.path.basename(file_path)

        """Finds two other FLAC files in the given directory, excluding the current file."""
        flac_files = list(f for f in Path(directory).iterdir() if f.suffix == ".flac" and f.name != current_filename)
        if len(flac_files) < 2:
            return base_noise

        return make_third_noise(base_noise, *[torchaudio.load(path)[0].t().numpy() for path in random.sample(flac_files, 2)])

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        path = f"{self._archive}/{metadata[0].replace('.flac', '.dirty.flac')}"

        dirty_waveform = None
        if not os.path.isfile(path):
            dirty_waveform = self.find_two_other_flac_files(f"{self._archive}/{metadata[0]}", waveform.t().numpy()).T
            dirty_waveform = tensor(make_second_noise(dirty_waveform))
            torchaudio.save(path, dirty_waveform, metadata[1])
        else:
            dirty_waveform = torchaudio.load(path)[0]  # torchaudio loads the waveform as well as the sample rate

        if self.mode == "x":
            return dirty_waveform[0].unsqueeze(0)
        if self.mode == "y":
            return waveform

        return (dirty_waveform[0].unsqueeze(0), waveform) + metadata[1:]  # no special mode selected

    def __len__(self) -> int:
        return len(self.walker)


class LIBRISPEECH_MEL(LIBRISPEECH):
    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download)

    def __getitem__(self, n):
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        clean_path = f"{self._archive}/{metadata[0]}"
        dirty_path = f"{self._archive}/{metadata[0].replace('.flac', '.dirty.flac')}"

        dirty_waveform = self.find_two_other_flac_files(f"{self._archive}/{metadata[0]}", waveform.t().numpy()).T
        dirty_waveform = tensor(make_second_noise(dirty_waveform))
        torchaudio.save(dirty_path, dirty_waveform, metadata[1])
        if not os.path.isfile(dirty_path):
            dirty_waveform = self.find_two_other_flac_files(f"{self._archive}/{metadata[0]}", waveform.t().numpy()).T
            dirty_waveform = tensor(make_second_noise(dirty_waveform))
            torchaudio.save(dirty_path, dirty_waveform, metadata[1])
        else:
            dirty_waveform = torchaudio.load(dirty_path)[0]  # torchaudio loads the waveform as well as the sample rate

        clean_spectrogram = create_spectrogram(clean_path, waveform.numpy().squeeze())
        dirty_spectrogram = create_spectrogram(dirty_path, dirty_waveform.numpy().squeeze())

        return (tensor(dirty_spectrogram[0][:, :400]).unsqueeze(0).to(float32), tensor(clean_spectrogram[0][:, :400]).unsqueeze(0).to(float32), {"metadata": metadata, "clean": clean_spectrogram[1], "dirty": dirty_spectrogram[1], "original": (waveform, dirty_waveform)})
        # return (tensor(clean_spectrogram[0][:, :400]).unsqueeze(0).to(float32), tensor(dirty_spectrogram[0][:, :400]).unsqueeze(0).to(float32), {"metadata": metadata, "clean": clean_spectrogram[1], "dirty": dirty_spectrogram[1], "original": (waveform, dirty_waveform)})
