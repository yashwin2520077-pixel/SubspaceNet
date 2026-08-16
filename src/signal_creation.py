"""Subspace-Net
Details
----------
Name: signal_creation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the Samples class, which inherits from SystemModel class.
This class is used for defining the samples model.
"""

# Imports
import numpy as np
from src.system_model import SystemModel, SystemModelParams
from src.utils import D2R
import scipy.io.wavfile as wavfile
from scipy.signal import resample
from pathlib import Path

class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.
    """

    def __init__(self, system_model_params: SystemModelParams, speech_dir=None):
        super().__init__(system_model_params)
        self.speech_signals = []
        if speech_dir is not None:
            speech_path = Path(speech_dir)
            wav_files = sorted(speech_path.glob("*.wav"))
            for wav_file in wav_files[:system_model_params.M]:
                sr, data = wavfile.read(str(wav_file))
                if len(data.shape) > 1:
                    data = data[:, 0]
                data = data.astype(np.float64) / np.max(np.abs(data))
                if self.params.signal_type == "Broadband":
                    target_sr = self.f_sampling["Broadband"]
                    num_samples = int(len(data) * target_sr / sr)
                    data = resample(data, num_samples)
                self.speech_signals.append(data)
            print(f"Loaded {len(self.speech_signals)} speech files")

    def set_doa(self, doa):
        def create_doa_with_gap(gap: float):
            M = self.params.M
            while True:
                DOA = np.round(np.random.rand(M) * 180, decimals=2) - 90
                DOA.sort()
                diff_angles = np.array(
                    [np.abs(DOA[i + 1] - DOA[i]) for i in range(M - 1)]
                )
                if (np.sum(diff_angles > gap) == M - 1) and (
                    np.sum(diff_angles < (180 - gap)) == M - 1
                ):
                    break
            return DOA

        if doa == None:
            self.doa = np.array(create_doa_with_gap(gap=15)) * D2R
        else:
            self.doa = np.array(doa) * D2R

    def samples_creation(
        self,
        noise_mean: float = 0,
        noise_variance: float = 1,
        signal_mean: float = 0,
        signal_variance: float = 1,
    ):
        signal = self.signal_creation(signal_mean, signal_variance)
        noise = self.noise_creation(noise_mean, noise_variance)
        if self.params.signal_type.startswith("NarrowBand"):
            A = np.array([self.steering_vec(theta) for theta in self.doa]).T
            samples = (A @ signal) + noise
            return samples, signal, A, noise
        elif self.params.signal_type.startswith("Broadband"):
            samples = []
            SV = []
            for idx in range(self.f_sampling["Broadband"]):
                if idx > int(self.f_sampling["Broadband"]) // 2:
                    f = -int(self.f_sampling["Broadband"]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.doa]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, : self.params.T]
            return samples_time_domain, signal, SV, noise
        else:
            raise Exception(
                f"Samples.samples_creation: signal type {self.params.signal_type} is not defined"
            )

    def noise_creation(self, noise_mean, noise_variance):
        if self.params.signal_type.startswith("NarrowBand"):
            return (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, self.params.T)
                    + 1j * np.random.randn(self.params.N, self.params.T)
                )
                + noise_mean
            )
        elif self.params.signal_type.startswith("Broadband"):
            noise = (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, len(self.time_axis["Broadband"]))
                    + 1j
                    * np.random.randn(self.params.N, len(self.time_axis["Broadband"]))
                )
                + noise_mean
            )
            return np.fft.fft(noise)
        else:
            raise Exception(
                f"Samples.noise_creation: signal type {self.params.signal_type} is not defined"
            )

    def signal_creation(self, signal_mean: float = 0, signal_variance: float = 1):
        amplitude = 10 ** (self.params.snr / 10)
        # NarrowBand signal creation
        if self.params.signal_type == "NarrowBand":
            # If real speech files are loaded, use them
            if hasattr(self, 'speech_signals') and len(self.speech_signals) > 0:
                if not hasattr(self, '_speech_printed'):
                    print(f"Using {len(self.speech_signals)} real speech files (NarrowBand)")
                    self._speech_printed = True
                signal = np.zeros((self.params.M, self.params.T), dtype=complex)
                for i in range(self.params.M):
                    speech = self.speech_signals[i % len(self.speech_signals)]
                    max_start = len(speech) - self.params.T
                    if max_start > 0:
                        start_idx = np.random.randint(0, max_start)
                    else:
                        start_idx = 0
                    segment = speech[start_idx:start_idx + self.params.T]
                    if len(segment) < self.params.T:
                        segment = np.pad(segment, (0, self.params.T - len(segment)))
                    from scipy.signal import hilbert
                    analytic = hilbert(segment)
                    signal[i] = amplitude * analytic / np.std(np.abs(analytic))

                return signal

            # Otherwise use synthetic signals (original code)
            elif self.params.signal_nature == "non-coherent":
                if not hasattr(self, '_synth_printed'):
                    print("No speech files, using synthetic signals")
                    self._synth_printed = True
                return (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(self.params.M, self.params.T)
                        + 1j * np.random.randn(self.params.M, self.params.T)
                    )
                    + signal_mean
                )

            elif self.params.signal_nature == "coherent":
                sig = (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(1, self.params.T)
                        + 1j * np.random.randn(1, self.params.T)
                    )
                    + signal_mean
                )
                return np.repeat(sig, self.params.M, axis=0)

        # Broadband signal creation
        elif self.params.signal_type.startswith("Broadband"):
            num_sub_carriers = self.max_freq["Broadband"]
            time_len = len(self.time_axis["Broadband"])

            # If real speech files are loaded, use them
            if hasattr(self, 'speech_signals') and len(self.speech_signals) > 0:
                signal = np.zeros((self.params.M, time_len)) + 1j * np.zeros((self.params.M, time_len))
                for i in range(self.params.M):
                    speech = self.speech_signals[i % len(self.speech_signals)]
                    max_start = len(speech) - time_len
                    if max_start > 0:
                        start_idx = np.random.randint(0, max_start)
                    else:
                        start_idx = 0
                    segment = speech[start_idx:start_idx + time_len]
                    if len(segment) < time_len:
                        segment = np.pad(segment, (0, time_len - len(segment)))
                    signal[i] = amplitude * segment / np.std(segment)
                return np.fft.fft(signal)

            # Otherwise use synthetic OFDM
            else:
                if self.params.signal_nature == "non-coherent":
                    signal = np.zeros(
                        (self.params.M, time_len)
                    ) + 1j * np.zeros((self.params.M, time_len))
                    for i in range(self.params.M):
                        for j in range(num_sub_carriers):
                            sig_amp = (
                                amplitude
                                * (np.sqrt(2) / 2)
                                * (np.random.randn(1) + 1j * np.random.randn(1))
                            )
                            signal[i] += sig_amp * np.exp(
                                1j
                                * 2
                                * np.pi
                                * j
                                * len(self.f_rng["Broadband"])
                                * self.time_axis["Broadband"]
                                / num_sub_carriers
                            )
                        signal[i] *= 1 / num_sub_carriers
                    return np.fft.fft(signal)
                elif self.params.signal_nature == "coherent":
                    signal = np.zeros(
                        (1, time_len)
                    ) + 1j * np.zeros((1, time_len))
                    for j in range(num_sub_carriers):
                        sig_amp = (
                            amplitude
                            * (np.sqrt(2) / 2)
                            * (np.random.randn(1) + 1j * np.random.randn(1))
                        )
                        signal += sig_amp * np.exp(
                            1j
                            * 2
                            * np.pi
                            * j
                            * len(self.f_rng["Broadband"])
                            * self.time_axis["Broadband"]
                            / num_sub_carriers
                        )
                    signal *= 1 / num_sub_carriers
                    return np.tile(np.fft.fft(signal), (self.params.M, 1))
                else:
                    raise Exception(
                        f"signal nature {self.params.signal_nature} is not defined"
                    )

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
