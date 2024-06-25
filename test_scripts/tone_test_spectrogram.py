import numpy as np 
import scipy.signal as sig 
import os 
import matplotlib.pyplot as plt 

if __name__ == "__main__":

    chann_iq = np.fromfile("../iq/tone_channelized.32cf", dtype="complex64")

    # Plot Channogram and ChannPsd
    fig, ax = plt.subplots(2, 1) 
    reshaped_iq = np.abs(chann_iq).reshape((1024, -1))[400:650,150:].transpose()[:, ::-1]
    slice_dim = chann_iq.reshape(1024, -1).shape[1]
    reduced_iq = np.sum(np.square(np.abs(chann_iq.reshape(1024, -1))[:, 150:]), axis=1) / slice_dim
    reduced_iq = 10*np.log10(reduced_iq / np.max(reduced_iq))

    c = ax[0].pcolor(50.0 - 100.0 * np.array([ind for ind in range(400, 650)][::-1]).astype("float32") / 1024, np.arange(reshaped_iq.shape[0]), reshaped_iq, vmax=80)
    os.makedirs("../images", exist_ok=True)
    #fig.colorbar(c, ax=ax[0])
    ax[1].plot(50.0 - 100.0 * np.array([ind for ind in range(400, 650)][::-1]).astype("float32") / 1024, reduced_iq[400:650][::-1])
    ax[0].axes.set_aspect('auto')
    ax[0].axes.set_ylabel("sample count")
    ax[1].axes.set_aspect('auto')
    ax[1].axes.set_xlabel("Frequency (MHz)")
    ax[1].axes.set_ylabel("Power in dB")
    ax[1].axes.set_ylim(-60.0, 0)
    ax[1].axes.grid(axis="y", which="major")
    fig.tight_layout()
    fig.savefig("../images/channogram.png")
    # Plot Spectrogram and Normal PSD 
    Nfft = 1024 
    Nover = 512 
    Nwind = 1024 
    
    iq = np.fromfile("../iq/tones.32cf", dtype="complex64")
    f, psd = sig.welch(iq, fs=100.0, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    fsp, t, spec = sig.spectrogram(iq, fs=100.0, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    new_fig, new_ax = plt.subplots(2,1,sharex=True)
    reshaped_spec = np.fft.fftshift(np.abs(spec), axes=0)[::-1][400:650, :].transpose()
    c = new_ax[0].pcolor(np.fft.fftshift(fsp)[400:650], t, reshaped_spec, vmax=10)
    reduced_psd = 10*np.log10(np.fft.fftshift(psd) / np.max(psd))
    new_ax[1].plot(np.fft.fftshift(f)[400:650], reduced_psd[::-1][400:650])
    new_ax[0].axes.set_aspect('auto')
    new_ax[0].axes.set_ylabel("sample count")
    new_ax[1].axes.set_aspect('auto')
    new_ax[1].axes.set_xlabel("Frequency (MHz)")
    new_ax[1].axes.set_ylabel("Power in dB")
    new_ax[1].axes.set_ylim(-60.0, 0)
    new_ax[1].axes.grid(axis="y", which="major")


    new_fig.tight_layout()
    new_fig.savefig("../images/spectrogram.png")



