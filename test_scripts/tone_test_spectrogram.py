import numpy as np 
import scipy.signal as sig 
from scipy.signal import resample_poly
import os 
import matplotlib.pyplot as plt 

SYNTHETIC_THRESHOLD = -40.0
OTA_THRESHOLD = -33.0

def detect(norm_iq, is_synth=False):
    """
    Detect the energy in the normalized IQ data. This will return
    a logical array where the energy is above a certain threshold.
    """
    threshold_db = SYNTHETIC_THRESHOLD if is_synth else OTA_THRESHOLD
    energy = np.array([x > threshold_db for x in norm_iq])    
    return energy
    

def find_continuous_segments(logical_array):
    """
    Find the continuous segments in the logical array. This will
    return a list of tuples where each tuple is the start and end
    index of the segment.
    """
    segments = []
    segment_start = None
    segment_has_started = False

    for i, value in enumerate(logical_array):
        if value and segment_start is None:
            segment_start = i
        elif not value and segment_has_started:
            segments.append((segment_start, i - 1))
            segment_start = None
        segment_has_started = value
        
    if logical_array[0] and logical_array[-1]:
        segments.append((segment_start, len(logical_array) - 1))

    return segments

def combine_continuous_segments(segments):
    """
    Combine the continuous segments into a single value.
    """
    combined = np.zeros(len(segments), dtype="int")
    for i, segement in enumerate(segments):
        combined[i] = int(np.mean(segement))
    return combined

def find_fcs(detected_energy, fs):
    segments = find_continuous_segments(detected_energy)
    average_bin = combine_continuous_segments(segments)
    fcs = [fs/2 - fs * ind / 1024 for ind in average_bin]
    return fcs

def save_fcs(fcs_ota_channogram, fcs_synth_channogram, fcs_ota_spectrogram, fcs_synth_spectrogram, filename):
    # create a file and write the FCS in OTA and Synthetic
    # AttributeError: 'numpy.float64' object has no attribute 'write'
    with open(filename, "w") as f:
        f.write("Center frequency of detected energy in OTA Channogram:")
        for fcs in fcs_ota_channogram:
            f.write("\n"+ str(fcs) + " MHz")
        f.write("\n")
        f.write("\nCenter frequency of detected energy in Synthetic Channogram:")
        for fcs in fcs_synth_channogram:
            f.write("\n"+ str(fcs) + " MHz")
        f.write("\n")
        f.write("\nCenter frequency of detected energy in OTA Spectrogram:")
        for fcs in fcs_ota_spectrogram:
            f.write("\n"+ str(fcs) + " MHz")
        f.write("\n")
        f.write("\nCenter frequency of detected energy in Synthetic Spectrogram:")
        for fcs in fcs_synth_spectrogram:
            f.write("\n"+ str(fcs) + " MHz")
        f.write("\n")
        
if __name__ == "__main__":

    chann_iq = np.fromfile("../iq/ota_tones_channelized.32cf", dtype="complex64")
    synth_chann_iq = np.fromfile("../iq/synthetic_tones_channelized.32cf", dtype="complex64")
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
    start_index = 450
    stop_index = 530
    
    ax[0].plot(fs / 2 - fs * np.array([ind for ind in range(start_index, stop_index)][::-1]).astype("float32") / 1024, reduced_iq[start_index:stop_index][::-1], label="ota")
    ax[0].plot(fs / 2 - fs * np.array([ind for ind in range(start_index, stop_index)][::-1]).astype("float32") / 1024, synth_reduced_iq[start_index:stop_index][::-1], linestyle="--", label="synthetic")
    ax[0].plot([0.5, 0.5], [-60, 0], linestyle="--", color="black", label="true tone")
    ax[0].plot([0.85, 0.85], [-60, 0], linestyle="--", color="black")
    ax[0].plot([1.26, 1.26], [-60, 0], linestyle="--", color="black")
    #ax[1].plot(2.5 - 5.0 * np.array([ind for ind in range(0, 1024)][::-1]).astype("float32") / 1024, reduced_iq[::-1])
    # ax[0].axes.set_aspect('auto')
    # ax[0].axes.set_ylabel("sample count")
    ax[0].axes.set_aspect('auto')
    # ax[0].axes.set_xlabel("Frequency (MHz)")
    ax[0].axes.set_ylabel("Power in dB")
    ax[0].axes.set_ylim(-60.0, 0)
    # ax[0].axes.minorticks_on()
    ax[0].axes.grid(axis="y", which="major")
    ax[0].axes.set_xlim(-1.0, 2.0)
    ax[0].axes.legend(loc="best")
    ax[0].axes.set_title("Channogram")
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
    
    start_fft_index = 500
    stop_fft_index = 534
    iq = np.fromfile("../iq/ota_tones.32cf", dtype="complex64")
    f, psd = sig.welch(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    fsp, t, spec = sig.spectrogram(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
   # new_fig, new_ax = plt.subplots(2,1,sharex=True)
    reshaped_spec = np.fft.fftshift(np.abs(spec), axes=0).transpose()
    # c = new_ax[0].pcolor(np.fft.fftshift(fsp)[400:650], t, reshaped_spec, vmax=10)
    reduced_psd = 10*np.log10(np.fft.fftshift(psd) / np.max(psd))
    ax[1].plot(np.fft.fftshift(f)[start_fft_index:stop_fft_index], reduced_psd[::-1][start_fft_index:stop_fft_index], label="ota")

    # new_ax[0].axes.set_aspect('auto')
    # new_ax[0].axes.set_ylabel("sample count")
    # ax[1].axes.set_aspect('auto')
    # ax[1].axes.set_xlabel("Frequency (MHz)")
    # ax[1].axes.set_ylabel("Power in dB")
    # ax[1].axes.set_ylim(-60.0, 0)
    # ax[1].axes.grid(axis="y", which="both")
    
    iq = np.fromfile("../iq/synthetic_tones.32cf", dtype="complex64")
    f_, psd_ = sig.welch(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
    fsp, t, spec = sig.spectrogram(iq, fs=fs, window=("kaiser", 10.0), nperseg=Nwind, noverlap=Nover, nfft=Nfft, return_onesided=False)
   # new_fig, new_ax = plt.subplots(2,1,sharex=True)
    # reshaped_spec = np.fft.fftshift(np.abs(spec), axes=0).transpose()
    # c = new_ax[0].pcolor(np.fft.fftshift(fsp)[400:650], t, reshaped_spec, vmax=10)
    reduced_psd_ = 10*np.log10(np.fft.fftshift(psd_) / np.max(psd_))
    ax[1].plot(np.fft.fftshift(f_)[start_fft_index:stop_fft_index], reduced_psd_[::-1][start_fft_index:stop_fft_index], linestyle='--', label="synthetic")
    ax[1].plot([0.5, 0.5], [-60, 0], linestyle="--", color="black", label="true tone")
    ax[1].axes.legend(loc="best")
    ax[1].plot([0.85, 0.85], [-60, 0], linestyle="--", color="black")
    ax[1].plot([1.26, 1.26], [-60, 0], linestyle="--", color="black")
    # new_ax[0].axes.set_aspect('auto')
    # new_ax[0].axes.set_ylabel("sample count")
    ax[1].axes.set_aspect('auto')
    ax[1].axes.set_xlabel("Frequency (MHz)")
    ax[1].axes.set_ylabel("Power in dB")
    ax[1].axes.set_title("Spectrogram")
    # ax[1].axes.minorticks_on()
    ax[1].axes.set_ylim(-60.0, 0)
    ax[1].axes.set_xlim(-1.0, 2.0)
    ax[1].axes.grid(axis="y", which="major")


    fig.tight_layout()
    fig.savefig("../images/chann_spectrogram.png")


    # Detect energy in reduced_iq and synth_reduced_iq    
    detected_energy_ota_channogram = detect(reduced_iq)
    detected_energy_synthetic_channogram = detect(synth_reduced_iq, True)
    detected_energy_ota_spectro = detect(reduced_psd)
    detected_energy_synthetic_spectro = detect(reduced_psd_, True)
     
    fcs_ota_channogram = find_fcs(detected_energy_ota_channogram, fs)[::-1]
    fcs_synthetic_channogram = find_fcs(detected_energy_synthetic_channogram, fs)[::-1]
    fcs_ota_spectro = find_fcs(detected_energy_ota_spectro, fs)[::-1]
    fcs_synthetic_spectro = find_fcs(detected_energy_synthetic_spectro, fs)[::-1]
    
    save_fcs(fcs_ota_channogram, fcs_synthetic_channogram, fcs_ota_spectro, fcs_synthetic_spectro,"../images/fcs.txt")
    
    print("Detected energy in OTA Channogram: ", fcs_ota_channogram)
    print("Detected energy in Synthetic Channogram: ", fcs_synthetic_channogram)
    print("Detected energy in OTA Spectrogram: ", fcs_ota_spectro)
    print("Detected energy in Synthetic Spectrogram: ", fcs_synthetic_spectro)
    
    detect_fig, detect_ax = plt.subplots(2,1)
    
    detect_ax[0].axes.plot(fs / 2 - fs * np.array([ind for ind in range(start_index, stop_index)][::-1]).astype("float32") / 1024, detected_energy_ota_channogram[start_index:stop_index][::-1], label="OTA")
    detect_ax[0].axes.plot(fs / 2 - fs * np.array([ind for ind in range(start_index, stop_index)][::-1]).astype("float32") / 1024, detected_energy_synthetic_channogram[start_index:stop_index][::-1], linestyle="--", label="Synthetic")
    detect_ax[0].axes.set_title("Channogram Detection")
    detect_ax[0].axes.set_xlabel("Frequency (MHz)")
    detect_ax[0].axes.set_ylabel("Detected Energy")
    detect_ax[0].axes.grid(axis="y", which="major")
    detect_ax[0].axes.set_ylim(-0.1, 1.1)
    detect_ax[0].axes.set_xlim(-1.0, 2.0)
    detect_ax[0].axes.legend(loc="best")
    
    detect_ax[1].axes.plot(np.fft.fftshift(f_)[start_fft_index:stop_fft_index], detected_energy_ota_spectro[::-1][start_fft_index:stop_fft_index], label="OTA")
    detect_ax[1].axes.plot(np.fft.fftshift(f_)[start_fft_index:stop_fft_index], detected_energy_synthetic_spectro[::-1][start_fft_index:stop_fft_index], linestyle="--", label="Synthetic")
    detect_ax[1].axes.set_title("Spectrogram Detection")
    detect_ax[1].axes.set_xlabel("Frequency (MHz)")
    detect_ax[1].axes.set_ylabel("Detected Energy")
    detect_ax[1].axes.grid(axis="y", which="major")
    detect_ax[1].axes.set_ylim(-0.1, 1.1)
    detect_ax[1].axes.set_xlim(-1.0, 2.0)
    detect_ax[1].axes.legend(loc="best")
    
    detect_fig.tight_layout()
    detect_fig.savefig("../images/detection.png")