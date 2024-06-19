import numpy as np 
import scipy.signal as sig 
import os 
import matplotlib.pyplot as plt 

if __name__ == "__main__":

    chann_iq = np.fromfile("../iq/tone_channelized.32cf", dtype="complex64")

    # Plot Channogram and ChannPsd
    fig, ax = plt.subplots(2, 1) 
    reshaped_iq = np.abs(chann_iq).reshape((1024, -1))[400:650,150:].transpose()
    reduced_iq = np.sum(np.abs(chann_iq.reshape(1024, -1))[:, 150:], axis=1)
    reduced_iq = reduced_iq / np.max(reduced_iq)

    c = ax[0].pcolor(reshaped_iq, vmax=80)
    os.makedirs("../images", exist_ok=True)
    #fig.colorbar(c, ax=ax[0])
    ax[1].plot(np.log10(reduced_iq)[400:650])
    ax[0].axes.set_aspect('auto')
    ax[1].axes.set_aspect('auto')
    fig.savefig("../images/channogram.png")
    # Plot Spectrogram and Normal PSD 
    Nfft = 1024 
    Nover = 512 
    Nwind = 1024 
    
    iq = np.fromfile("../iq/tones.32cf", dtype="complex64")
    f, psd = sig.welch(iq, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    _, _, spec = sig.spectrogram(iq, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    new_fig, new_ax = plt.subplots(2,1)
    reshaped_spec = np.fft.fftshift(np.abs(spec), axes=0)[400:650, :].transpose()
    c = new_ax[0].pcolor(reshaped_spec, vmax=300)
    reduced_psd = np.log10(np.fft.fftshift(psd) / np.max(psd))
    new_ax[1].plot(reduced_psd[400:650])
    new_fig.tight_layout()
    new_fig.savefig("../images/spectrogram.png")



