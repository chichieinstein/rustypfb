import numpy as np 
import scipy.signal as sig 
from scipy.signal import resample_poly
import os 
import matplotlib.pyplot as plt 

if __name__ == "__main__":

    chann_iq = np.fromfile("../iq/tone_channelized.32cf", dtype="complex64")
    synth_chann_iq = np.fromfile("../iq/synthetic_channelized.32cf", dtype="complex64")
    # Plot Channogram and ChannPsd
    # fig, ax = plt.subplots(3, 1) 
    fig, ax = plt.subplots(2,1)
    # reshaped_iq = np.abs(chann_iq).reshape((1024, -1))[100:510,150:].transpose()[:, ::-1]
    # reshaped_iq_synth = 
    slice_dim = chann_iq.reshape(1024, -1).shape[1]
    reduced_iq = np.sum(np.square(np.abs(chann_iq.reshape(1024, -1))[:, 150:]), axis=1) / slice_dim
    reduced_iq_synth = np.sum(np.square(np.abs(synth_chann_iq.reshape(1024, -1))[:, 150:]), axis=1) / slice_dim
    reduced_iq = 10*np.log10(reduced_iq / np.max(reduced_iq))
    synth_reduced_iq = 10*np.log10(reduced_iq_synth / np.max(reduced_iq_synth))
    # synth_reduced_iq = resample_poly(synth_reduced_iq, up=20, down=1)
    # c = ax[0].pcolor(2.5 - 5.0 * np.array([ind for ind in range(400, 650)][::-1]).astype("float32") / 1024, np.arange(reshaped_iq.shape[0]), reshaped_iq, vmax=80)
    os.makedirs("../images", exist_ok=True)
    #fig.colorbar(c, ax=ax[0])
    cf = 0.0
    fs = 100.0
    synth_fs = 5.0
    start_index = 490
    stop_index = 512
    
    ax[0].plot(fs / 2 - fs * np.array([ind for ind in range(start_index, stop_index)][::-1]).astype("float32") / 1024, reduced_iq[start_index:stop_index][::-1])
    ax[0].plot(fs / 2 - fs * np.array([ind for ind in range(start_index, stop_index)][::-1]).astype("float32") / 1024, synth_reduced_iq[start_index:stop_index][::-1], linestyle="--")
    #ax[1].plot(2.5 - 5.0 * np.array([ind for ind in range(0, 1024)][::-1]).astype("float32") / 1024, reduced_iq[::-1])
    # ax[0].axes.set_aspect('auto')
    # ax[0].axes.set_ylabel("sample count")
    ax[0].axes.set_aspect('auto')
    ax[0].axes.set_xlabel("Frequency (MHz)")
    ax[0].axes.set_ylabel("Power in dB")
    ax[0].axes.set_ylim(-60.0, 0)
    ax[0].axes.minorticks_on()
    ax[0].axes.grid(axis="y", which="both")
    #ax[1].axes.set_aspect('auto')
    #ax[1].axes.set_xlabel("Frequency (MHz)")
    #ax[1].axes.set_ylabel("Power in dB")
    #ax[1].axes.set_ylim(-60.0, 0)
    #ax[1].axes.grid(axis="y", which="major")
    #fig.tight_layout()
    #fig.savefig("../images/channogram.png")
    # Plot Spectrogram and Normal PSD 
    Nfft = 1024 
    Nover = 512 
    Nwind = 1024 
    
    start_fft_index = 505
    stop_fft_index = 534
    iq = np.fromfile("../iq/tones.32cf", dtype="complex64")
    f, psd = sig.welch(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    fsp, t, spec = sig.spectrogram(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
   # new_fig, new_ax = plt.subplots(2,1,sharex=True)
    reshaped_spec = np.fft.fftshift(np.abs(spec), axes=0).transpose()
    # c = new_ax[0].pcolor(np.fft.fftshift(fsp)[400:650], t, reshaped_spec, vmax=10)
    reduced_psd = 10*np.log10(np.fft.fftshift(psd) / np.max(psd))
    ax[1].plot(np.fft.fftshift(f)[start_fft_index:stop_fft_index], reduced_psd[::-1][start_fft_index:stop_fft_index])

    # new_ax[0].axes.set_aspect('auto')
    # new_ax[0].axes.set_ylabel("sample count")
    # ax[1].axes.set_aspect('auto')
    # ax[1].axes.set_xlabel("Frequency (MHz)")
    # ax[1].axes.set_ylabel("Power in dB")
    # ax[1].axes.set_ylim(-60.0, 0)
    # ax[1].axes.grid(axis="y", which="both")
    
    iq = np.fromfile("../iq/synthetic.32cf", dtype="complex64")
    f_, psd_ = sig.welch(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    fsp, t, spec = sig.spectrogram(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
   # new_fig, new_ax = plt.subplots(2,1,sharex=True)
    # reshaped_spec = np.fft.fftshift(np.abs(spec), axes=0).transpose()
    # c = new_ax[0].pcolor(np.fft.fftshift(fsp)[400:650], t, reshaped_spec, vmax=10)
    reduced_psd_ = 10*np.log10(np.fft.fftshift(psd_) / np.max(psd_))
    ax[1].plot(np.fft.fftshift(f_)[start_fft_index:stop_fft_index], reduced_psd_[::-1][start_fft_index:stop_fft_index], linestyle='--')

    # new_ax[0].axes.set_aspect('auto')
    # new_ax[0].axes.set_ylabel("sample count")
    ax[1].axes.set_aspect('auto')
    ax[1].axes.set_xlabel("Frequency (MHz)")
    ax[1].axes.set_ylabel("Power in dB")
    ax[1].axes.minorticks_on()
    ax[1].axes.set_ylim(-60.0, 0)
    ax[1].axes.grid(axis="y", which="both")


    fig.tight_layout()
    fig.savefig("../images/chann_spectrogram.png")



