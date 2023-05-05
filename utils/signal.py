import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa, librosa.display
import os
from utils.annotation import get_simplified_annotation
import scipy


def get_segment(signal, annotation, Fs):
    signal_segment = []
    for i in range(annotation.shape[0]):
        signal_segment.append(signal[int(Fs * annotation['start'][i]): int(Fs * annotation['end'][i])])
    return signal_segment


def plot_signal(signal, Fs, title):
    time_axis = np.arange(0, len(signal) / Fs, 1 / Fs)
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, signal, alpha=0.5)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def envelope(signal, rate, threshold=0.0005):
    mask = []
    signal = pd.Series(signal).apply(np.abs)
    signal_mean = signal.rolling(window=int(rate / 100), min_periods=1, center=True).mean()
    for mean in signal_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def filter_signal(signal, Fs, cutoff_freq):
    b = scipy.signal.firwin(51, cutoff=cutoff_freq, fs=Fs, pass_zero='highpass')
    return scipy.signal.lfilter(b, [1.0], signal)

def plot_spectra(signal, Fs, title="", xlabel="", ylabel="", max_freq=None):
    if (len(signal) % 2 != 0):
        signal = signal[:-1]

    n_half = int(len(signal) / 2)

    transform = np.abs(np.fft.fft(signal)[:n_half])  # perform transformation and get first half points
    transform = transform / int(len(signal) / 2)  # normalise

    # x-axis calculation
    freqs = Fs * np.arange(len(signal)) / len(signal)
    freqs = freqs[:n_half]  # first half of the frequencies

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, transform, alpha=0.5,color='grey')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (Hz)')
    if max_freq is None:
        max_x = Fs / 2
    else:
        max_x = max_freq
    plt.xlim([0, max_x])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# https://stackoverflow.com/questions/61600977/can-someone-help-me-understand-the-np-abs-conversion-for-stft-in-librosa
def stft_audio(signal, Fs, n_fft, title, hop_percent=0.25, plot=False):
    signal_stft = librosa.stft(signal, n_fft=n_fft)
    signal_stft = np.abs(signal_stft) ** 2

    hop_length = int(hop_percent * n_fft)
    if plot:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(librosa.amplitude_to_db(signal_stft),
                                 sr=Fs,
                                 hop_length=hop_length,
                                 y_axis='log', x_axis='time')
        plt.title(title)
        colormap = plt.cm.get_cmap('plasma')
        plt.colorbar(format="%+2.0f dB")
        plt.show()

    return signal_stft


# Constants for calc_mfcc
# n_mfcc = 13
# MORE parameters could be adjusted according to: https://blog.csdn.net/qq_37653144/article/details/89045363
def calc_mfcc(x, Fs, d, frame_length=256, hop_length=128, plot=True):
    M = librosa.feature.mfcc(y=x, sr=Fs,
                             n_mfcc=13,
                             n_fft=frame_length,
                             hop_length=hop_length)
    if d > 0:
        M = librosa.feature.delta(M, order=d)

    if (plot == True):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(M,
                                 x_axis="time",
                                 sr=Fs)
        plt.colorbar(format="%+2.f")
        plt.show()
    return M


def get_features(signal, Fs, frame_length, hop_length, mfcc_order):
    features = pd.DataFrame(np.transpose(calc_mfcc(x=signal, Fs=Fs, d=mfcc_order, frame_length=frame_length, hop_length=hop_length, plot=False)))
    features['zcr'] = np.transpose(librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length))
    features['sc'] = np.transpose(librosa.feature.spectral_centroid(y=signal, sr=Fs, n_fft=frame_length, hop_length=hop_length))
    features['bw'] = np.transpose(librosa.feature.spectral_bandwidth(y=signal, sr=Fs, n_fft=frame_length, hop_length=hop_length))
    features['srf'] = np.transpose(librosa.feature.spectral_rolloff(y=signal, sr=Fs, n_fft=frame_length, hop_length=hop_length, roll_percent=0.95))

    return features


# Add missing get_segment function definition


def get_segment_stats(signal, Fs, frame_length, hop_length, mfcc_order, annotation, mean=True, filter=False, cutoff_freq=500):
    if filter:
        signal = filter_signal(signal=signal, Fs=Fs, cutoff_freq=cutoff_freq)

    seg_stats = get_features(signal=signal, Fs=Fs, frame_length=frame_length, hop_length=hop_length, mfcc_order=mfcc_order)

    if mean:
        seg_stats = seg_stats.rename(columns=dict(enumerate([f'mfcc_{i}' for i in range(mfcc_order + 1)])))
        segment_feature = annotation[['low_freq', 'high_freq', 'species']]
        segment_feature = segment_feature.reindex(columns=[f'm_{col}' for col in seg_stats.columns] + segment_feature.columns.tolist())
        segment_feature.loc[:, [f'm_{col}' for col in seg_stats.columns]] = seg_stats.mean().values
    else:
        segment_feature = pd.concat([seg_stats, annotation[['selection', 'low_freq', 'high_freq', 'date', 'species']].repeat(seg_stats.shape[0])]).reset_index(drop=True)
        segment_feature.columns = seg_stats.columns.tolist() + ['selection', 'low_freq', 'high_freq', 'date', 'species']

    return segment_feature



def get_total_audio_CNN(audio_path, target_audio_path, total_feature, frame_length=512, hop_length=256, mfcc_order=0, signal_filter=True):
    file_folder = target_audio_path['recording'].apply(lambda x: x.split('_Segment')[0])
    mid = target_audio_path.shift(1)
    feature_CNN = pd.DataFrame(columns=total_feature.columns)

    for i in tqdm(range(1, len(target_audio_path))):
        audio_filepath = os.path.join(audio_path, file_folder, target_audio_path['recording'][i]) + '.wav'
        annotation_filepath = audio_filepath.replace('wav_Files', 'annotation_Files').replace('.wav', '.Table.1.selections.txt')

        if mid['recording'][i] != target_audio_path['recording'][i]:
            audio, Fs = librosa.load(audio_filepath)
            annotation = get_simplified_annotation(annotation_filepath, filter=False)
            audio = get_segment(signal=audio, annotation=annotation, Fs=Fs)

        target_seg = audio[target_audio_path['selection'][i] - 1]

        pre_padding_sample = round((5 * Fs - audio.shape[0]) / 2, 1)
        after_padding_sample = 5 * Fs - pre_padding_sample - audio.shape[0]
        target_seg = np.pad(target_seg, (int(pre_padding_sample), int(after_padding_sample)), mode='constant')

        target_annotation = annotation[target_audio_path['selection'][i] - 1]
        feature = get_segment_stats(signal=target_seg, Fs=Fs, frame_length=frame_length, hop_length=hop_length,
                                mfcc_order=0, annotation=target_annotation, mean=False, filter=signal_filter)
        feature_CNN = pd.concat([feature_CNN, feature])

    return feature_CNN



def get_total_segment_stats(signal, Fs, mfcc_order, annotation,
                            frame_length, hop_length, pcp,
                            env=False, thres=0.0005, mean=True, filter=False, cutoff_freq=500):
    
    #:param signal: 5min的wav文件
    #:param Fs: Fs
    #:param mfcc_order: calc_mfcc()需要的参数
    #:param annotation: 5min的wav文件对应的txt
    #:param frame_length: 
    #:param hop_length: 
    #:param pcp: 仅含有表头的pcp
    #:param env: 是否需要平滑
    #:param thres: 平滑阈值
    #:param mean: 如果mean=True,则对一份鸟鸣片段中的所有特征取平均（未加权），如果mean=False，则保留所有片段中的frame
    #:return: 
    
    for seg in range(0, len(signal)):
        if env:
            mask = envelope(signal[seg], rate=Fs, threshold=thres)
            signal[seg] = signal[seg][mask]
        if filter:
            signal[seg] = filter_signal(signal[seg],Fs, cutoff_freq=cutoff_freq)
        seg_stats = get_segment_stats(signal[seg], Fs,
                                      frame_length, hop_length, mfcc_order, annotation[seg:(seg + 1)], mean, filter,cutoff_freq)
        feature = pd.concat([feature, seg_stats], ignore_index=True)

    return feature

def get_total_audio_stats(audio_path, mfcc_order, frame_length, hop_length, feature_seg,
                          env=False, thres=0.0005,
                          signal_filter=False, cutoff_freq=500,
                          annotation_filter=False, overlap_percent=0.1,
                          mean=True):
    total_feature = pd.DataFrame(columns=feature_seg.columns)

    for dirpath, _, filenames in os.walk(audio_path):
        if not filenames:
            continue

        for f in filenames:
            if f.startswith("."):
                continue

            # Read audio files and annotation files
            audio_filepath = os.path.join(dirpath, f)
            audio, Fs = librosa.load(audio_filepath)
            annotation_filepath = audio_filepath.replace('wav_Files', 'annotation_Files').replace('.wav', '.Table.1.selections.txt')
            annotation = get_simplified_annotation(annotation_filepath, filter=annotation_filter, overlap_percent=overlap_percent)

            if len(annotation) == 0:
                continue

            audio = get_segment(signal=audio, annotation=annotation, Fs=Fs)
            feature = get_total_segment_stats(signal=audio, Fs=Fs, mfcc_order=mfcc_order,
                                              annotation=annotation, frame_length=frame_length,
                                              hop_length=hop_length, feature_seg=feature_seg,
                                              env=env, thres=thres, mean=mean,
                                              filter=signal_filter, cutoff_freq=cutoff_freq)

            last_seg_num = total_feature.iloc[-1, 17] if not total_feature.empty else 0
            feature['segment'] = feature['segment'].apply(lambda x: x + last_seg_num)
            total_feature = pd.concat([total_feature, feature], ignore_index=True)

    return total_feature


"""
def __get_species_ix(elem, annotation):
    diffs = annotation['start'] - elem
    if len(diffs[diffs >= 0]) == 0:
        return 'm'
    else:
        return diffs[diffs >= 0].index[0]


def get_annotated_species_sequence(pcp, annotation, test_version=False):
    # pcp['low_freq'] = pcp.apply(lambda row: np.NaN if __get_species_ix(row['start'], annotation) == False
    # else annotation.iloc[__get_species_ix(row['start'], annotation)]['low_freq'], axis=1)
    # pcp['high_freq'] = pcp.apply(lambda row: np.NaN if __get_species_ix(row['start'], annotation) == False
    # else annotation.iloc[__get_species_ix(row['start'], annotation)]['high_freq'], axis=1)
    pcp['species'] = pcp.apply(lambda row: np.NaN if __get_species_ix(row['end'], annotation) == False
    else annotation.iloc[__get_species_ix(row['end'], annotation)]['species'], axis=1)
    if (test_version == False):
        pcp['species'][0] = '<START>'
        pcp['species'][-1] = '<END>'
    return pcp

def get_frame_stats(mfcc, signal, Fs):
    frames_per_sec = mfcc.shape[1] / (len(signal) / Fs)  # Nbr of frames / length in seconds = frames per second
    frame_duration_sec = 1 / frames_per_sec  # frame duration = 1 / frames per second
    return [frames_per_sec, frame_duration_sec]
    
def get_total_audio_CNN(audio_path, target_audio_path, total_pcp, frame_length=512, hop_length=256, mfcc_order=0,
                        signal_filter=True):
    file_folder = target_audio_path['recording'].apply(lambda x: x.split('_Segment')[0])
    mid = target_audio_path.shift(1)
    pcp_CNN = pd.DataFrame(total_pcp.columns)

    for i in range(1, len(target_audio_path)):
        audio_filepath = os.path.join(audio_path, file_folder, target_audio_path['recording'][i]) + '.wav'
        annotation_filepath = audio_filepath.replace('wav_Files', 'annotation_Files')
        annotation_filepath = annotation_filepath.replace('.wav', '.Table.1.selections.txt')

        if mid['recording'][i] != target_audio_path['recording'][i]:
            audio, Fs = librosa.load(audio_filepath)
            annotation = get_simplified_annotation(annotation_filepath, filter=False)
            audio = get_segment(signal=audio, annotation=annotation, Fs=Fs)

        target_seg = audio[target_audio_path['selection'][i] - 1]

        pre_padding_sample = round((5 * Fs - audio.shape[0]) / 2, 1)
        after_padding_sample = 5 * Fs - pre_padding_sample - audio.shape[0]
        target_seg = np.pad(target_seg, (int(pre_padding_sample), int(after_padding_sample)), mode='constant')

        target_annotation = annotation[target_audio_path['selection'][i] - 1]
        pcp = get_segment_stats(signal=target_seg, Fs=Fs, frame_length=frame_length, hop_length=hop_length,
                                mfcc_order=0,
                                annotation=target_annotation, mean=False, filter=signal_filter)
        pcp_CNN = pd.concat(pcp_CNN, pcp)

    return pcp_CNN
    

def get_total_audio_stats(audio_path, mfcc_order, frame_length, hop_length, feature_seg,
                          env=False, thres=0.0005,
                          signal_filter=False, cutoff_freq=500,
                          annotation_filter=False, overlap_percent=0.1,
                          mean=True):
    total_feature = pd.DataFrame(columns=feature_seg.columns)
    for dirpath, dirnames, filenames in os.walk(audio_path):
        if len(dirnames) == 0:
            for f in filenames:
                if not f.startswith("."):
                    # read audio files and annotation files
                    audio_filepath = os.path.join(dirpath, f)
                    audio, Fs = librosa.load(audio_filepath)
                    annotation_filepath = audio_filepath.replace('wav_Files', 'annotation_Files')
                    annotation_filepath = annotation_filepath.replace('.wav', '.Table.1.selections.txt')
                    annotation = get_simplified_annotation(annotation_filepath, filter=annotation_filter, overlap_percent=overlap_percent)
                    if len(annotation) == 0:
                        continue
                    audio = get_segment(signal=audio, annotation=annotation, Fs=Fs)
                    feature = pd.DataFrame(columns=feature_seg.columns)
                    feature = get_total_segment_stats(signal=audio, Fs=Fs, mfcc_order=mfcc_order,
                                                  annotation=annotation, frame_length=frame_length,
                                                  hop_length=hop_length, feature_seg =feature_seg,
                                                  env=env, thres=thres, mean=mean,
                                                  filter=signal_filter, cutoff_freq=cutoff_freq)
                    last_seg_num = total_feature.iloc[-1,17] if total_feature.size !=0 else 0
                    feature['segment'] = feature['segment'].apply(lambda x: x + last_seg_num)
                    total_feature = pd.concat([total_feature, feature],ignore_index=True)
    return total_feature

"""
