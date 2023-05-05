import os
import pandas as pd
import numpy as np
from operator import truediv
from tqdm import tqdm


def get_meta_annotation(header_path, annotation_path):
    bird_annotation = pd.read_csv(header_path, sep='\t')

    for dirpath, _, filenames in os.walk(annotation_path):
        if not filenames:
            continue

        for f in filenames:
            if f.startswith("."):
                continue

            annotation_filepath = os.path.join(dirpath, f)
            df_mid = pd.read_csv(annotation_filepath, sep='\t')
            df_mid['duration'] = df_mid['End Time (s)'] - df_mid['Begin Time (s)']
            df_mid['is_isolate'] = if_isolate(df_mid)
            df_mid['overlap_percent'] = overlap_2_duration_percent(df_mid)
            df_mid['recording'] = f.split('.')[0]
            bird_annotation = bird_annotation.append(df_mid)

    bird_annotation = bird_annotation.rename(columns={'Begin Time (s)': 'start',
                                                      'End Time (s)': 'end',
                                                      'Low Freq (Hz)': 'low_freq',
                                                      'High Freq (Hz)': 'high_freq',
                                                      'Species': 'species',
                                                      'Selection': 'selection'})
    return bird_annotation.sort_values(by=['recording', 'selection'])

def if_isolate(annotation):
    isolate = []
    for i in range(annotation.shape[0]):
        iso_elem_s = [annotation['Begin Time (s)'][i] > annotation['End Time (s)'][i - max(1,j)] for j in range(min(5, i))]
        iso_elem_e = [annotation['End Time (s)'][i] < annotation['Begin Time (s)'][i + k] for k in range(1, min(5, annotation.shape[0] - i))]
        isolate.append(all(iso_elem_e) and all(iso_elem_s))
    return isolate


def get_simplified_meta_annotation(meta_annotation):
    meta_annotation = meta_annotation.groupby('recording')
    meta_annotation['avg_annotation_per_species'] = round(
        meta_annotation['number_of_annotation'] / meta_annotation['species_count'], 1)
    return meta_annotation.sort_values(by='avg_annotation_per_species', ascending=False)

def overlap_2_duration_percent(annotation):
    overlap_size = []

    for i in range(annotation.shape[0]):
        start_overlap = np.array([annotation['End Time (s)'][i - max(1, j)] - annotation['Begin Time (s)'][i] for j in range(min(5, i))])
        end_overlap = np.array([annotation['End Time (s)'][i] - annotation['Begin Time (s)'][i + k] for k in range(1, min(5, annotation.shape[0] - i))])

        start_overlap_max = max(start_overlap) if start_overlap.size else 0
        end_overlap_max = max(end_overlap) if end_overlap.size else 0

        overlap_size.append(start_overlap_max + end_overlap_max)

    overlap_2_duration = np.array(list(map(truediv, np.array(overlap_size), np.array(annotation['duration']))))

    return overlap_2_duration

def get_simplified_annotation(annotation_path, filter=False, overlap_percent=0.1):
    annotation = pd.read_csv(annotation_path, sep='\t')
    annotation['duration'] = annotation['End Time (s)'] - annotation['Begin Time (s)']
    annotation['is_isolate'] = if_isolate(annotation)
    annotation['overlap_percent'] = overlap_2_duration_percent(annotation)
    annotation['date'] = annotation_path.split('Recording_')[1].split('/')[0]
    annotation = annotation.rename(columns={'Begin Time (s)': 'start',
                                            'End Time (s)': 'end',
                                            'Low Freq (Hz)': 'low_freq',
                                            'High Freq (Hz)': 'high_freq',
                                            'Species': 'species',
                                            'Selection': 'selection'})
    if filter:
        annotation = annotation[annotation['overlap_percent']<overlap_percent]
        annotation = annotation.reset_index(drop=True)
        annotation['selection'] = 1 + np.arange(annotation.shape[0])

    return annotation

"""
F  = '/Users/ruyu/Desktop/thesis/dryad/annotation_Files/Recording_3/Recording_3_Segment_01.Table.1.selections.txt'
a = get_simplified_annotation(F)
a = a.iloc[0:10,3:5]
print(a)
print(overlap_2_duration_percent(a))

o_s = []
d_s = []

for i in range(a.shape[0]):
    elem_overlap_s_d = np.array([a['end'][i - max(1,j)] - a['start'][i] for j in range(0, min(5, i))])
    elem_overlap_e_d = np.array([a['end'][i] - a['start'][i + k] for k in range(1, min(5, a.shape[0] - i))])
    elem_overlap_s_d_m = max(max(elem_overlap_s_d) if (elem_overlap_s_d.size != 0) else 0, 0)
    elem_overlap_e_d_m = max(max(elem_overlap_e_d) if (elem_overlap_e_d.size != 0) else 0, 0)
    o_s.append(elem_overlap_s_d_m + elem_overlap_e_d_m)
    d_s.append(a['end'][i] - a['start'][i])

print(list(map(truediv,np.array(o_s),np.array(d_s))))
"""




