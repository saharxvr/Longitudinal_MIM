import matplotlib.pyplot as plt
import numpy as np
import pydicom
from datetime import datetime
import shutil
import os
from pathlib import Path
from os.path import join, dirname
from glob import glob
# import SimpleITK as sitk

import torch
from tqdm import tqdm
import pandas as pd
from time import time
from collections import defaultdict
# import pytesseract
from PIL import Image
# from skimage.transform import resize
import nibabel as nib
from random import shuffle
from constants import *
import imageio.v2 as imageio

home_folder = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis"
files_path = "/cs/labs/josko/public/itamar_sab/physionet.org/files"
files_removed_path = "/cs/labs/josko/public/itamar_sab/physionet.org/files_removed"

files_path_local = "physionet.org/files"

keywords_dict = {"Atelectasis": (['atelectasis', 'atelectasic'], []), "Consolidation": (['consolidation', 'consolidated'], ['no consolidation', 'not consolidated']),
                 "Pneumothorax": (['pneumothorax'], []), "Pleural Effusion": (["pleural effusion"], ["no pleural effusion"]), "Pericardial Effusion": (['pericardial effusion'], ['no pericardial effusion']),
                 "Cardiac Hypertrophy": (['cardiac hypertrophy'], ['no cardiac hypertrophy']), 'Cardiac Hyperinflation': (['cardiac hyperinflation'], ['no cardiac hyperinflation']),
                 'Tuberculosis': (['tuberculosis'], ['no tuberculosis']), 'Pneumonectomy': (['pneumonectomy'], []), 'Lobectomy': (['lobectomy'], [])}

cxr14_path = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14"
cl_path = cxr14_path + '/contrastive_learning'
pd_path = cxr14_path + '/pathology_detection'


def remove_empty_series():
    series_dircs = glob(files_path + '/p*/p*/s????????')
    print(series_dircs[0])
    print(len(series_dircs))
    for dirc in tqdm(series_dircs):
        if not os.listdir(dirc):
            shutil.rmtree(dirc)


def filter_AP_files():
    dicoms = glob(files_path_local + '/p*/p*/pair*/*/*.dcm')
    print(dicoms)
    for dicom in tqdm(dicoms):
        info = pydicom.read_file(dicom)
        print(info.ProcedureCodeSequence[0].CodeMeaning)
        print(info.ViewPosition)
        continue
        if info.ViewPosition != 'AP':
            report_path = dirname(dicom) + '.txt'
            inner_dicom_folder = dicom.split('files/')[1]
            new_dicom_path = join(files_removed_path, inner_dicom_folder)
            inner_report_path = report_path.split('files/')[1]
            new_report_path = join(files_removed_path, inner_report_path)
            os.makedirs(new_dicom_path, exist_ok=True)
            shutil.move(dicom, new_dicom_path)
            if not os.path.exists(new_report_path):
                new_report_path = '/'.join(new_report_path.split('/')[:-1])
                os.makedirs(new_report_path, exist_ok=True)
                shutil.move(report_path, new_report_path)
            continue


def filter_AP_files_CXR14():
    csv_name = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/sheets/Data_Entry_2017_v2020.csv"
    df = pd.read_csv(csv_name)
    pa_df = df[df["View Position"] == "PA"]
    pa_folder = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/PA_images/"
    for i, row in enumerate(pa_df.iterrows()):
        row = row[1]
        shutil.move(cxr14_path + f'/AP_images/{row["Image Index"]}', pa_folder + row["Image Index"])
        if i % 1000 == 0:
            print(f'Moved {i}')

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return (d2 - d1).days


def is_within_3_days(d1, d2):
    return abs(days_between(d1, d2)) <= 3


def is_within_2_weeks(d1, d2):
    return abs(days_between(d1, d2)) <= 14


def update_high_proximity_study_pairs_subdirectories():
    paths = glob(os.path.join(files_path, 'p*/p*/pair_s????????_s????????'))
    print(f"Current number of pair folders: {len(paths)}")
    failures = []
    not_AP = []
    for path in tqdm(paths):
        try:
            bl_p = glob(os.path.join(path, 'BL_s????????/*.dcm'))[0]
            fu_p = glob(os.path.join(path, 'FU_s????????/*.dcm'))[0]
        except Exception:
            print(f"######## Failed on\n{path} ###########")
            failures.append(path)
            continue
        bl_dcm = pydicom.read_file(bl_p)
        fu_dcm = pydicom.read_file(fu_p)
        if bl_dcm.ViewPosition != 'AP':
            print(f"##################### Case {bl_p} is no AP\n#####################")
            not_AP.append(bl_p)
            continue
        if fu_dcm.ViewPosition != 'AP':
            print(f"##################### Case {fu_p} is no AP\n#####################")
            not_AP.append(fu_p)
            continue
        bl_date = bl_dcm.StudyDate
        fu_date = fu_dcm.StudyDate
        if not is_within_3_days(bl_date, fu_date):
            print(f"Pair:\n{bl_p}\n{fu_p}\n are not within 3 days.\nDeleting path folder.")
            shutil.rmtree(path)
    print(F"Failures: {failures}")
    print(f'Not APs: {not_AP}')
    print(f"New number of pair folders: {len(glob(os.path.join(files_path, 'p*/p*/pair_s????????_s????????')))}")


def create_pair_dirc(p_dir, series1, series2):
    pair_dir = join(p_dir, f"pair_{os.path.basename(series1)}_{os.path.basename(series2)}")
    for s, st in [(series1, 'BL'), (series2, 'FU')]:
        new_s_dir = join(pair_dir, st + '_' + os.path.basename(s))
        os.makedirs(new_s_dir, exist_ok=True)
        for f in os.listdir(s):
            src = join(s, f)
            dst = join(new_s_dir, f)
            if not os.path.lexists(dst):
                os.symlink(src, dst)


def create_high_proximity_study_pairs_subdirectories():
    total = 0
    patient_dircs = glob(files_path_local + '/p*/p????????')
    for p_dir in tqdm(patient_dircs):
        print(f"Working on {p_dir}")
        series_date_dircs = [[join(p_dir, series), pydicom.read_file(join(p_dir, series, os.listdir(join(p_dir, series))[0])).StudyDate, pydicom.read_file(join(p_dir, series, os.listdir(join(p_dir, series))[0])).StudyTime] for series in os.listdir(p_dir) if not (series.endswith('.txt') or series.startswith('pair_'))]
        series_date_dircs.sort(key=lambda item: (item[1], item[2]))
        for i in range(len(series_date_dircs) - 1):
            elem1 = series_date_dircs[i]
            elem2 = series_date_dircs[i + 1]
            d1 = elem1[1]
            d2 = elem2[1]
            if is_within_2_weeks(d1, d2):
                series1 = elem1[0]
                series2 = elem2[0]
                create_pair_dirc(p_dir, series1, series2)
                total += 1
                print(f'Created pair directory for:\n{series1.split("/")[-1]}\n{series2.split("/")[-1]}\n'
                      f'{total} directories have been creaated so far')
    print(f'Total number of pairs found is {total}')


def delete_pair_directories():
    print("Tried deleting pair directories. Blocked")
    exit()  # For safety
    pair_dircs = glob(files_path + '/p*/p*/pair_s????????_s????????')
    for pair_dirc in tqdm(pair_dircs):
        for dirc in os.listdir(pair_dirc):
            for file in os.listdir(join(pair_dirc, dirc)):
                f_path = join(pair_dirc, dirc, file)
                if os.path.islink(f_path):
                    os.unlink(f_path)
        shutil.rmtree(pair_dirc)


def get_cases_with_keywords_in_report(dic=(), keywords=(), negative_keywords=()):
    if not dict:
        dic = {'Keyword': (keywords, negative_keywords)}
    report_files = glob(files_path + '/p*/p*/s????????.txt')
    result_dict = {key: {'files': [], 'len': 0} for key in dic}
    for file in tqdm(report_files):
        with open(file) as rf:
            contents = rf.read().lower()
            for dis, (kws, neg_kws) in dic.items():
                found = [(word in contents) for word in kws]
                neg_found = [(word in contents) for word in neg_kws]
                if any(found) and not any(neg_found):
                    result_dict[dis]['len'] += 1
                    if result_dict[dis]["len"] <= 20:
                        result_dict[dis]['files'].append(file.split('.txt')[0])
                        print(f'{result_dict[dis]["len"]} added to {dis}')

    return result_dict


def count_and_mark_supine_cases_and_pairs():
    supine_path = '../../physionet.org/supine_pairs'
    paths = glob(os.path.join(files_path, 'p*/p*/pair_s????????_s????????'))
    approx_sup_t = 0
    sup_pairs_t = 0
    for path in tqdm(paths):
        bl_ps = glob(os.path.join(path, 'BL_s????????/*.dcm'))
        fu_ps = glob(os.path.join(path, 'FU_s????????/*.dcm'))

        bl_sup = False
        for bl_p in bl_ps:
            bl = pydicom.dcmread(bl_p).pixel_array
            bl = ((bl == np.max(bl)) * 255).astype(np.uint8)
            bl_text = pytesseract.image_to_string(bl, config='--oem 3 --psm 6')
            if "SUPINE" in bl_text.upper():
                bl_sup = True
                break

        fu_sup = False
        for fu_p in fu_ps:
            fu = pydicom.dcmread(fu_p).pixel_array
            fu = ((fu == np.max(fu)) * 255).astype(np.uint8)
            fu_text = pytesseract.image_to_string(fu, config='--oem 3 --psm 6')
            if "SUPINE" in fu_text.upper():
                fu_sup = True
                break

        if bl_sup and fu_sup:
            approx_sup_t += 2
            sup_pairs_t += 1
            print(f"Pairs found so far: {sup_pairs_t}")

            new_path = os.path.join(supine_path, "/".join(path.split('/')[-3:]))
            for dirc in os.listdir(path):
                dirc_path = os.path.join(path, dirc)
                new_dirc_path = os.path.join(new_path, dirc)
                os.makedirs(new_dirc_path, exist_ok=True)
                for dcm in os.listdir(dirc_path):
                    dcm_path = os.path.join(dirc_path, dcm)
                    new_dcm_path = os.path.join(new_dirc_path, dcm)
                    print(new_dcm_path)
                    if not os.path.lexists(new_dcm_path):
                        os.symlink(dcm_path, new_dcm_path)

        elif bl_sup or fu_sup:
            approx_sup_t += 1
    print(f'Approximate total number of dicoms with "SUPINE" is {approx_sup_t}')
    print(f'Total number of BL FU pairs, both with "SUPINE" is {sup_pairs_t}')


def copy_supine_cases():
    supine_path = '../../physionet.org/supine_pairs'
    paths = glob(os.path.join(supine_path, 'p*/p*/pair_s????????_s????????/*/*.dcm'))
    for path in tqdm(paths):
        if os.path.islink(path):
            dcm_path = Path(path).resolve()
            os.unlink(path)
            shutil.copy(dcm_path, path)


AFFINE_DCM = np.array([
    [-0.139, 0, 0, 0],
    [0, -0.139, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


def convert_cxr14_to_nii_gz():
    from PIL import Image
    files_dir = cxr14_path + '/AP_images'
    for i, f in enumerate(os.listdir(files_dir)):
        p = f'{files_dir}/{f}'
        if not p.endswith('.png'):
            print(f"skipping {p}")
            continue
        img = Image.open(p).convert("L")
        arr = np.array(img.getdata()).astype(np.uint8).reshape((1024, 1024)).T
        arr_nif = nib.Nifti1Image(arr, AFFINE_DCM)
        nib.save(arr_nif, files_dir + f'/{f.split(".")[0]}.nii.gz')
        os.remove(p)
        if i % 1000 == 0:
            print(f'converted {i+1}')


def split_cxr14_to_dirs():
    ap_path = cxr14_path + '/AP_images'
    data_path = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/sheets/Data_Entry_2017_v2020.csv"
    data_df = pd.read_csv(data_path)
    AP_df = data_df[data_df["View Position"] == "AP"]
    AP_df = AP_df.rename(columns={"Image Index": "id"})
    p1 = cxr14_path + '/sheets/miccai2023_nih-cxr-lt_labels_train.csv'
    p2 = cxr14_path + '/sheets/miccai2023_nih-cxr-lt_labels_val.csv'
    p3 = cxr14_path + '/sheets/miccai2023_nih-cxr-lt_labels_test.csv'
    df = pd.concat([pd.read_csv(p1), pd.read_csv(p2), pd.read_csv(p3)])
    df = pd.merge(df, AP_df, how='inner', on=["id"])[['id', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
            'Pleural Thickening', 'Pneumonia', 'Pneumothorax']].reset_index()
    cl_train_df = df.iloc[:42000].copy().reset_index()
    cl_val_df = df.iloc[42000:43200].copy().reset_index()
    cl_test_df = df.iloc[43200:].copy().reset_index()
    pd_train_df = df.iloc[30000:41000].copy().reset_index()
    pd_val_df = df.iloc[41000:42300].copy().reset_index()
    pd_test_df = df.iloc[42300:].copy().reset_index()
    ds_dfs = [cl_train_df, cl_val_df, cl_test_df, pd_train_df, pd_val_df, pd_test_df]
    ds_dirs = [cl_path + '/train', cl_path + '/val', cl_path + '/test', pd_path + '/train', pd_path + '/val', pd_path + '/test']

    for z, (ds_df, ds_dir) in enumerate(zip(ds_dfs, ds_dirs)):
        if z == 3:
            exit()
        print(f'working on {ds_dir}')
        images_dir = ds_dir + '/images'
        # df_arr = ds_df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
        #     'Pleural Thickening', 'Pneumonia', 'Pneumothorax']].to_numpy().astype(np.bool_)
        ds_df["labels"] = ds_df.apply(lambda r: r[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
            'Pleural Thickening', 'Pneumonia', 'Pneumothorax']].to_numpy(dtype=np.bool_), axis=1)
        ds_df = ds_df.drop(columns=['level_0', 'index'])
        for i in tqdm(range(len(ds_df))):
            cur_im = ds_df.iloc[i]['id'].split(".")[0] + '.nii.gz'
            ds_df.at[i, 'id'] = cur_im
            # src_im_path = ap_path + f'/{cur_im}'
            # dst_im_path = images_dir + f'/{cur_im}'
            # os.symlink(src_im_path, dst_im_path)
        print(ds_df)
        print(f"saving to {ds_dir + '/labels.csv'}")
        ds_df.to_csv(ds_dir + '/labels.csv')


def delete_npy():
    paths = glob(cxr14_path + '/AP_images/*.npy')
    for i, path in enumerate(paths):
        os.remove(path)
        if i % 1000 == 0:
            print(f'removed {i}')


def transpose_images():
    ap_path = cxr14_path + '/AP_images'
    paths = glob(ap_path + '/*.nii.gz')
    for i, path in enumerate(paths):
        im_data = nib.load(path).get_fdata().T
        im_nif = nib.Nifti1Image(im_data, AFFINE_DCM)
        os.remove(path)
        nib.save(im_nif, path)
        if i % 1000 == 0:
            print(f'transposed {i}')


def create_MIMIC_no_finding_csv():
    negbio_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-negbio.csv.gz'
    chexpert_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz'
    metadata_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-metadata.csv.gz'

    negbio_csv = pd.read_csv(negbio_path, delimiter=',')
    negbio_csv = negbio_csv.loc[negbio_csv['No Finding'] == 1.0]

    chexpert_csv = pd.read_csv(chexpert_path, delimiter=',')
    chexpert_csv = chexpert_csv.loc[chexpert_csv['No Finding'] == 1.0]

    no_finding_csv = pd.merge(chexpert_csv, negbio_csv, how='inner', on=['subject_id', 'study_id', 'No Finding']).reset_index()[['subject_id', 'study_id']]

    metadata_csv = pd.read_csv(metadata_path, delimiter=',')
    metadata_csv = metadata_csv.loc[metadata_csv['ViewPosition'] == 'AP']

    AP_no_finding_csv = pd.merge(no_finding_csv, metadata_csv, how='inner', on=['subject_id', 'study_id']).reset_index()[['subject_id', 'study_id']]

    AP_no_finding_csv.to_csv('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/No_Finding.csv')


def create_cxr14_no_finding_csv():
    data_path = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/sheets/Data_Entry_2017_v2020.csv"
    data_df = pd.read_csv(data_path)
    p1 = cxr14_path + '/sheets/miccai2023_nih-cxr-lt_labels_train.csv'
    p2 = cxr14_path + '/sheets/miccai2023_nih-cxr-lt_labels_val.csv'
    p3 = cxr14_path + '/sheets/miccai2023_nih-cxr-lt_labels_test.csv'
    df = pd.concat([pd.read_csv(p1), pd.read_csv(p2), pd.read_csv(p3)])
    no_f_df = df.loc[df['No Finding'] == 1]

    for vp in ['AP', 'PA']:
        vp_df = data_df[data_df["View Position"] == vp]
        vp_df = vp_df.rename(columns={"Image Index": "id"})
        cur_no_f_df = pd.merge(no_f_df, vp_df, how='inner', on=["id"]).reset_index()[['id']]
        # cur_all_df = pd.merge(df, vp_df, how='inner', on=["id"]).reset_index()[['id','Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural Thickening','Pneumonia','Pneumothorax','No Finding']]

        for i in range(len(cur_no_f_df)):
            cur_im = cur_no_f_df.iloc[i]['id'].split(".")[0] + '.nii.gz'
            cur_no_f_df.at[i, 'id'] = cur_im

        cur_no_f_df.to_csv(CXR14_FOLDER + f'/No_Finding_{vp}.csv')


def create_mimic_cxr_no_finding_no_duplicates_csv():
    meta_path = 'mimic-cxr-2.0.0-metadata.csv.gz'
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.loc[(meta_df['ViewPosition'] == 'AP') | (meta_df['PerformedProcedureStepDescription'] == 'CHEST (PORTABLE AP)') | (meta_df['ProcedureCodeSequence_CodeMeaning'] == 'CHEST (PORTABLE AP)') | (meta_df['ViewCodeSequence_CodeMeaning'] == 'antero-posterior')]
    meta_df = meta_df[['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime']]
    print(f'meta_df len: {len(meta_df)}')

    negbio_path = 'mimic-cxr-2.0.0-negbio.csv.gz'
    chexpert_path = 'mimic-cxr-2.0.0-chexpert.csv.gz'

    negbio_csv = pd.read_csv(negbio_path, delimiter=',')
    negbio_csv = negbio_csv.loc[negbio_csv['No Finding'] == 1.0]

    chexpert_csv = pd.read_csv(chexpert_path, delimiter=',')
    chexpert_csv = chexpert_csv.loc[chexpert_csv['No Finding'] == 1.0]

    chexneg_csv = pd.merge(chexpert_csv, negbio_csv, how='inner', on=['subject_id', 'study_id']).reset_index()
    chexneg_csv = pd.merge(chexneg_csv, meta_df, how='inner', on=['subject_id', 'study_id'])[['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime']]
    # chexneg_csv = chexneg_csv.drop_duplicates(['subject_id', 'study_id'], keep=False)
    print(f'chexneg len: {len(chexneg_csv)}')

    path_train = 'mimic-cxr-lt_single-label_train.csv'
    path_val = 'mimic-cxr-lt_single-label_balanced-val.csv'
    path_test = 'mimic-cxr-lt_single-label_test.csv'
    df_t = pd.read_csv(path_train)
    df_t = df_t.loc[df_t['No Finding'] == 1]
    df_v = pd.read_csv(path_val)
    df_v = df_v.loc[df_v['No Finding'] == 1]
    df_test = pd.read_csv(path_test)
    df_test = df_test.loc[df_test['No Finding'] == 1]
    df = pd.concat([df_t, df_v, df_test])
    df = pd.merge(df, meta_df, how='inner', on=['subject_id', 'study_id', 'dicom_id'])[['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime']]
    # df = df.drop_duplicates(['subject_id', 'study_id'], keep=False)
    print(f'tr_val_te_df len: {len(df)}')

    df = pd.merge(df, chexneg_csv, how='outer')
    print(f"outer len: {len(df)}")

    df = df.sort_values(by=['StudyDate', 'StudyTime'])
    df = df.drop_duplicates(['subject_id', 'study_id'], keep='first')
    df = df.reset_index()[['subject_id', 'study_id', 'dicom_id']]
    print(f'final len {len(df)}')

    for i in range(len(df)):
        cur_im = df.iloc[i]['dicom_id'].split(".")[0] + '_cropped.nii.gz'
        df.at[i, 'dicom_id'] = cur_im

    df.to_csv('No_Finding.csv')


def check_num_uncropped():
    all_paths = glob(f'{MIMIC_OTHER_AP}/**/*.nii.gz', recursive=True)
    cropped_paths = glob(f'{MIMIC_OTHER_AP}/**/*_cropped.nii.gz', recursive=True)
    print(len(all_paths))
    print(len(cropped_paths))


def remove_from_cxr14_no_finding():
    path = 'No_Finding.csv'
    df = pd.read_csv(path)
    names = ['00000057_001.nii.gz', '00000061_019.nii.gz', '00000091_007.nii.gz', '00000112_001.nii.gz', '00000127_006.nii.gz']


def prep_pneumonia_normal_ds():
    from PIL import Image
    from torchvision.transforms import Resize
    import torch
    img_resize = Resize((512, 512))
    path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal/normal'
    dst_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal/normal_nibs'
    for f in tqdm(os.listdir(path)):
        cur_p = os.path.join(path, f)

        if not cur_p.endswith('.jpg'):
            print(f'skipping {cur_p}')
            continue
        img = Image.open(cur_p).convert("L")
        arr = torch.tensor(np.array(img.getdata()).astype(np.uint8).reshape((img.height, img.width)).T)[None, ...]
        arr = np.array(img_resize(arr)).squeeze(0)
        arr_nif = nib.Nifti1Image(arr, AFFINE_DCM)
        dst_name = '.'.join(f.split('.')[:-1]) + '.nii.gz'
        cur_dst_p = os.path.join(dst_path, dst_name)
        nib.save(arr_nif, cur_dst_p)


def prep_pneumonia_abnormal_ds():
    from PIL import Image
    from torchvision.transforms import Resize
    import torch
    img_resize = Resize((512, 512))
    path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal/pneumonia'
    dst_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal/pneumonia_nibs'
    for f in tqdm(os.listdir(path)):
        cur_p = os.path.join(path, f)

        if not cur_p.endswith('.jpg'):
            print(f'skipping {cur_p}')
            continue
        img = Image.open(cur_p).convert("L")
        arr = torch.tensor(np.array(img.getdata()).astype(np.uint8).reshape((img.height, img.width)).T)[None, ...]
        arr = np.array(img_resize(arr)).squeeze(0)
        arr_nif = nib.Nifti1Image(arr, AFFINE_DCM)
        dst_name = '.'.join(f.split('.')[:-1]) + '.nii.gz'
        cur_dst_p = os.path.join(dst_path, dst_name)
        nib.save(arr_nif, cur_dst_p)


def prep_VinDrCXR(d_set='train'):
    import numpy as np
    csv_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_{d_set}.csv'
    df = pd.read_csv(csv_path)

    if d_set == 'train':
        # df = df.drop(columns=['rad_id'])

        print('e2b2f50550d1dc76448410e7ff060251' in list(df['image_id']))

        all_df = df.groupby('image_id').mean().reset_index().round()
        all_df[all_df.columns.difference(['image_id'])] = all_df[all_df.columns.difference(['image_id'])].astype(int)
        all_dst_csv = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_{d_set}.csv'
        # all_df.to_csv(all_dst_csv)

        print('e2b2f50550d1dc76448410e7ff060251' in list(all_df['image_id']))

        # no_finding_df = np.floor(df.groupby('image_id').mean()).reset_index()
        no_finding_df = all_df[all_df['No finding'] == 1]['image_id']
        print(no_finding_df)
        no_finding_df_dst_csv = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/no_finding_{d_set}_2.csv'
        no_finding_df.to_csv(no_finding_df_dst_csv)

        # print('e2b2f50550d1dc76448410e7ff060251' in list(no_finding_df['image_id']))

    elif d_set == 'test':
        no_finding_df = df[df['No finding'] == 1]['image_id']
        no_finding_df_dst_csv = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/no_finding_{d_set}.csv'
        no_finding_df.to_csv(no_finding_df_dst_csv)


def create_PadChest_relevant_csv():
    csv_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz'
    df = pd.read_csv(csv_path)
    df = df[df["MethodLabel"] == 'Physician']
    df = df[df["ViewPosition_DICOM"].isin({'AP', 'ANTEROPOSTERIOR', 'POSTEROANTERIOR', 'PA'}) | df["Projection"].isin({'AP', 'PA', 'AP_horizontal'})]
    # print(set(df["ViewPosition_DICOM"]))
    # print(set(df["Projection"]))
    # print(set(df["MethodProjection"]))
    # print(set(df["MethodLabel"]))
    # print(df["Labels"])
    df.to_csv('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv')


def create_PadChest_no_finding_csv():
    from ast import literal_eval
    csv_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv'
    df = pd.read_csv(csv_path)

    relevant = {'loc basal', 'loc diffuse bilateral', 'loc paracardiac', 'loc upper lung field', 'loc lobar', 'loc bilateral costophrenic angle',
                'loc supradiaphragm', 'loc apical', 'loc suprahilar', 'loc right costophrenic angle', 'loc middle lobe', 'loc aortopulmonary window',
                'loc central', 'loc minor fissure', 'loc left', 'loc upper lobe', 'loc left costophrenic angle', 'loc lower lobe', 'loc major fissure',
                'loc pleural', 'loc right', 'loc costophrenic angle', 'loc hemithorax', 'loc bilateral', 'loc perihilar', 'loc cardiophrenic angle',
                'loc subsegmental', 'loc infrahilar', 'loc right upper lobe', 'loc fissure', 'loc basal bilateral', 'loc hilar', 'loc lower lung field',
                'loc hilar bilateral', 'loc subpleural', 'loc middle lung field', 'loc left lower lobe', 'loc right lower lobe', 'loc lung field',
                'loc lingula', 'loc left upper lobe'}

    df_normal = df[df["Labels"] == "[\'normal\']"]["ImageID"]

    # df['Localizations'] = df.Localizations.apply(literal_eval)
    # df['Localizations'] = df.Localizations.apply(lambda x: len(set(x).intersection(relevant)) == 0)
    # df = df[df['Localizations'] == True]
    # print(df)
    # df = df.merge(df_normal, 'outer', on='ImageID')['ImageID']
    df_normal.to_csv('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/no_finding_only_normal.csv')


def create_PadChest_specific_abnormalities_csv():
    from ast import literal_eval
    csv_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv'
    df = pd.read_csv(csv_path)

    relevant = {'pulmonary mass', 'pneumothorax', 'pneumonia', 'pulmonary edema', 'consolidation', 'nodule', 'pleural effusion', 'infiltrates', 'atelectasis', 'cardiomegaly'}
    # relevant = {'consolidation'}

    df['Labels'] = df.Labels.apply(literal_eval)
    df['Labels'] = df.Labels.apply(lambda x: len(set(x).intersection(relevant)) > 0)
    df = df[df['Labels'] == True]['ImageID']
    print(df)

    df.to_csv(f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/specific_abnormalities.csv')


def filter_PadChest_manual_labels():
    csv_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv'
    df = pd.read_csv(csv_path)
    all_names = set(df['ImageID'])
    print(len(all_names))

    images_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images'
    images_set = set(os.listdir(images_path))
    print(len(images_set))

    to_remove = images_set.difference(all_names)
    print(len(to_remove))
    for name in tqdm(to_remove):
        c_path = images_path + f'/{name}'
        os.remove(c_path)


def convert_PadChest_to_uint8():
    src_dir_path = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/PadChest/images'
    dst_dir_path = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/PadChest/images_new'

    files = os.listdir(src_dir_path)

    for f in tqdm(files, total=len(files)):
        c_path = src_dir_path + f'/{f}'
        c_data = nib.load(c_path).get_fdata()[None, ...]
        c_max = np.max(c_data)
        c_min = np.min(c_data)
        if c_min == c_max:
            print(f)
        c_data = (c_data - c_min) / (c_max - c_min)
        c_data = c_data * 255
        c_data = c_data.astype(np.uint8)
        c_data = c_data.squeeze()
        nif = nib.Nifti1Image(c_data, AFFINE_DCM)
        nib.save(nif, dst_dir_path + f'/{f}')


def crop_PadChest_edges():
    from utils import crop_out_edges
    import torchvision.transforms as tf

    src_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images'
    dst_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images_new'

    resize = tf.Resize((512, 512))

    for name in tqdm(os.listdir(src_dir)):
        path = f'{src_dir}/{name}'
        data = nib.load(path).get_fdata()
        d_std = np.std(data)
        if d_std < 5:
            print(f'Found std {d_std} for scan\n{name}')
            continue
        new_data = crop_out_edges(data)
        if new_data.shape != (512, 512):
            # print(f'{name} shape after cropping is: {new_data.shape}')
            new_data = np.array(resize(torch.tensor(new_data[None, ...])).squeeze(0))
        new_data = new_data.astype(np.uint8)
        nif = nib.Nifti1Image(new_data, AFFINE_DCM)
        dst_path = f'{dst_dir}/{name}'
        nib.save(nif, dst_path)


def create_new_PadChest_relevant_images_csv():
    csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv'
    df = pd.read_csv(csv_path)
    df = df[df['ImageID'] != '303333177099020648058143195258938210104_9nx0n6.png']
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images_new.csv')


def create_PadChest_specific_abnormalities_labels_csv():
    from ast import literal_eval
    csv_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv'
    df = pd.read_csv(csv_path)

    relevant = {'pulmonary mass', 'pneumothorax', 'pneumonia', 'pulmonary edema', 'consolidation', 'nodule',
                'pleural effusion', 'infiltrates', 'atelectasis', 'cardiomegaly', 'pulmonary fibrosis',
                'interstitial pattern', 'reticulonodular interstitial pattern', 'reticular interstitial pattern'}

    sup_devs = {'NSG tube', 'double J stent', 'gastrostomy tube', 'reservoir central venous catheter',
                'central venous catheter via umbilical vein', 'metal', 'central venous catheter',
                'artificial aortic heart valve', 'central venous catheter via jugular vein',
                'artificial mitral heart valve', 'single chamber device', 'artificial heart valve',
                'dual chamber device', 'catheter', 'endotracheal tube', 'chest drain tube', 'pacemaker',
                'tracheostomy tube', 'ventriculoperitoneal drain tube', 'electrical device', 'nephrostomy tube'}

    df['Labels'] = df.Labels.apply(literal_eval)
    # df['filt'] = df.Labels.apply(lambda x: len(set(x).intersection(relevant)) > 0)
    # df = df[df['filt'] == True]

    for label in relevant:
        df[label] = df.Labels.apply(lambda x: label in set(x))
        print(f'{label}, {len(df[df[label] == True])}')
    df['support devices'] = df.Labels.apply(lambda x: len(set(x).intersection(sup_devs)) > 0)
    df['filt'] = df.apply(lambda c_row: any(c_row[l] == True for l in list(relevant) + ['support devices']), axis=1)
    df = df[df['filt'] == True]
    df = df[['ImageID', 'support devices'] + list(relevant)]
    df[['support devices'] + list(relevant)] = df[['support devices'] + list(relevant)].astype(int)
    df.to_csv('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/specific_abnormalities_labels.csv')


def create_CXR14_and_MIMIC_specific_abnormalities_labels_csv():
    cxr_ap_csv = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/All_AP.csv'
    cxr_pa_csv = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/All_PA.csv'
    mimic_ap_csv1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz'
    mimic_ap_csv2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-negbio.csv.gz'

    cxr_ap_df = pd.read_csv(cxr_ap_csv)
    cxr_pa_df = pd.read_csv(cxr_pa_csv)
    mimic_ap_df = pd.read_csv(mimic_ap_csv1)
    mimic_ap_df = pd.read_csv(mimic_ap_csv2)

    print(cxr_pa_df[cxr_pa_df['Mass'] == 1].head(10))


def invert_photometric_VinDr():
    dir1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train_dic'
    dir2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test_dic'
    data_dir1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train_new'
    data_dir2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test_new'
    dst1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train_inverted'
    dst2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test_inverted'
    files1 = [f'{dir1}/{n}' for n in os.listdir(dir1)]
    files2 = [f'{dir2}/{n}' for n in os.listdir(dir2)]
    files = files1 + files2

    for p in files:
        c_dic = pydicom.dcmread(p)

        try:
            if c_dic.PhotometricInterpretation == 'MONOCHROME1':
                n = f"{p.split('/')[-1].split('.')[0]}.nii.gz"
                if '/train' in p:
                    c_dir = data_dir1
                    c_dst = dst1
                elif '/test' in p:
                    c_dir = data_dir2
                    c_dst = dst2
                else:
                    raise 'wa??'
                c_p = f'{c_dir}/{n}'
                c_data = nib.load(c_p).get_fdata()
                c_data = 255 - c_data
                c_dst_p = f'{c_dst}/{n}'
                nif = nib.Nifti1Image(c_data, AFFINE_DCM)
                nib.save(nif, c_dst_p)
            elif c_dic.PhotometricInterpretation == 'MONOCHROME2':
                pass
            else:
                print(f'Found image with PhotometricInterpretation {c_dic.PhotometricInterpretation}')
        except Exception as e:
            print(f'Got this error:\n{e}')


def remove_bad_CXR14_case():
    p_good_1 = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/AP_images'
    p_good_2 = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/PA_images'
    good_cases = {n.split('.')[0] for n in os.listdir(p_good_1) + os.listdir(p_good_2)}
    print(len(good_cases))

    p_new = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images'
    new_cases = {n.split('.')[0] for n in os.listdir(p_new)}
    print(len(new_cases))

    print(new_cases.difference(good_cases))


def convert_cases_to_nib():
    folder = 'cases_nitzan\D'
    for n in os.listdir(folder):
        if not (n.endswith('.png') or n.endswith('.PNG') or n.endswith('.jpg')):
            continue
        p = folder + f'/{n}'
        new_p = folder + f'/{n.split(".")[0]}.nii.gz'
        img = Image.open(p).convert("L")
        arr = np.array(img.getdata()).reshape((img.height, img.width)).T
        arr = arr.astype(np.uint8)
        arr_nif = nib.Nifti1Image(arr, AFFINE_DCM)
        nib.save(arr_nif, new_p)
        os.remove(p)


def resize_cases():
    import torchvision.transforms.v2 as v2

    resize = v2.Resize((768, 768), antialias=True)
    resize_seg = v2.Resize((768, 768), antialias=False)
    folder = 'cases_nitzan/images3'
    dst_folder = 'cases_nitzan/images3_new'
    folder_segs = 'cases_nitzan/images3_segs'
    dst_folder_segs = 'cases_nitzan/images3_segs_new'
    for n in tqdm(os.listdir(folder)):
        if n.endswith('_resized.nii.gz'):
            continue
        p = folder + fr'/{n}'
        # new_p = folder + f'/{n.split(".")[0]}_resized.nii.gz'
        new_p = dst_folder + fr'/{n}'
        img = torch.tensor(nib.load(p).get_fdata().T[None, ...])
        img = np.array(resize(img).squeeze()).astype(np.uint8).T
        arr_nif = nib.Nifti1Image(img, AFFINE_DCM)
        nib.save(arr_nif, new_p)
        # os.remove(p)
    for n in tqdm(os.listdir(folder_segs)):
        if n.endswith('_resized.nii.gz'):
            continue
        p = folder_segs + fr'/{n}'
        # new_p = folder_segs + f'/{n.split(".")[0]}_resized.nii.gz'
        new_p = dst_folder_segs + fr'/{n}'
        img = torch.tensor(nib.load(p).get_fdata().T[None, ...])
        img = np.array(resize_seg(img).squeeze()).astype(np.uint8).T
        arr_nif = nib.Nifti1Image(img, AFFINE_DCM)
        nib.save(arr_nif, new_p)
        # os.remove(p)


def generate_spectrum_graphs():
    def get_val_and_std(df, col):
        y_to_split = list(df[col])
        y_to_split = np.array([sp.split('\n± ') for sp in y_to_split], dtype=float).T
        y_val = y_to_split[0]
        y_std = y_to_split[1]
        return y_val, y_std

    idx_to_measure = {0: 'Precision', 1: 'Recall', 2: 'Dice coefficient'}
    name_to_param = {'masses_ab_power_spectrum_measures': 'Abnormality Power', 'masses_frequency_scale_spectrum_measures': 'Inverse Frequency',
                     'masses_inv_size_spectrum_measures': 'Size Cutoff', 'masses_blur_fac_spectrum_measures': 'Blur Factor',
                     'lesions_ab_power_spectrum_measures': 'Abnormality Power', 'lesions_frequency_scale_spectrum_measures': 'Inverse Frequency',
                     'lesions_sparsity_mult_spectrum_measures': 'Inverse Sparsity', 'lesions_blur_fac_spectrum_measures': 'Blur Factor', 'lesions_inv_size_spectrum_measures': 'Size Cutoff',
                     'general_opacity_cent_mult_spectrum_measures': 'Abnormality Power', 'general_opacity_inv_size_spectrum_measures': 'Size Cutoff',
                     'general_opacity_opacity_coef_spectrum_measures': 'Transparency', 'general_opacity_blur_fac_spectrum_measures': 'Blur Factor',
                     'general_opacity_add_texture_spectrum_measures': 'Textured opacity', 'general_opacity_frequency_scale_spectrum_measures': 'Inverse Frequency', 'general_opacity_texture_frequencies_spectrum_measures': 'Texture Frequency',
                     'disordered_opacity_ab_power_spectrum_measures': 'Abnormality Power', 'disordered_opacity_blur_fac_spectrum_measures': 'Blur Factor',
                     'disordered_opacity_both_frequencies_spectrum_measures': 'Inverse Frequency 1 and 2', 'disordered_opacity_central_focus_spectrum_measures': 'Central Focus',
                     'disordered_opacity_frequency_scale1_spectrum_measures': 'Inverse Frequency 1', 'disordered_opacity_inv_size_spectrum_measures': 'Size Cutoff'}

    name_to_trained_span = {
        'masses_ab_power_spectrum_measures': (8, 32),
        'masses_frequency_scale_spectrum_measures': (32, 64),
        'masses_inv_size_spectrum_measures': (0.45, 0.85),
        'masses_blur_fac_spectrum_measures': (5, 15),
        'lesions_ab_power_spectrum_measures': (35, 60),
        'lesions_frequency_scale_spectrum_measures': (6, 12),
        'lesions_sparsity_mult_spectrum_measures': (0.96, 1.),
        'lesions_blur_fac_spectrum_measures': (5, 13),
        'lesions_inv_size_spectrum_measures': (0.5, 1.),
        'general_opacity_cent_mult_spectrum_measures': (0.875, 1.025),
        'general_opacity_inv_size_spectrum_measures': (0.15, 0.4),
        'general_opacity_opacity_coef_spectrum_measures': (0.25, 0.8),
        'general_opacity_blur_fac_spectrum_measures': (31, 39),
        'general_opacity_add_texture_spectrum_measures': (0, 1),
        'general_opacity_frequency_scale_spectrum_measures': (60, 150),
        'general_opacity_texture_frequencies_spectrum_measures': (50, 70),
        'disordered_opacity_ab_power_spectrum_measures': (18.5, 28.5),
        'disordered_opacity_blur_fac_spectrum_measures': (5, 13),
        'disordered_opacity_both_frequencies_spectrum_measures': (8, 32),
        'disordered_opacity_central_focus_spectrum_measures': (3, 9),
        'disordered_opacity_frequency_scale1_spectrum_measures': (8, 32),
        'disordered_opacity_inv_size_spectrum_measures': (0.45, 0.75)
    }

    paths = [f'spectrum_sheets/{n}' for n in os.listdir('spectrum_sheets') if n.endswith('csv')]

    uncertainty = False

    if not uncertainty:
        cols = (('detection_precision_neg_th_1', 'detection_precision_neg_th_2', 'detection_precision_pos_th_1', 'detection_precision_pos_th_2'),
                ('detection_recall_neg_th_1', 'detection_recall_neg_th_2', 'detection_recall_pos_th_1', 'detection_recall_pos_th_2'),
                ('dice_neg_th_1', 'dice_neg_th_2', 'dice_pos_th_1', 'dice_pos_th_2'))
    else:
        cols = ((('detection_precision_neg_th_1', 'detection_precision_neg_th_2', 'detection_precision_pos_th_1', 'detection_precision_pos_th_2'),
                 ('detection_precision_neg_uncertainty_th_1', 'detection_precision_neg_uncertainty_th_2', 'detection_precision_pos_uncertainty_th_1', 'detection_precision_pos_uncertainty_th_2')),
                (('detection_recall_neg_th_1', 'detection_recall_neg_th_2', 'detection_recall_pos_th_1', 'detection_recall_pos_th_2'),
                 ('detection_recall_neg_uncertainty_th_1', 'detection_recall_neg_uncertainty_th_2', 'detection_recall_pos_uncertainty_th_1', 'detection_recall_pos_uncertainty_th_2')),
                (('dice_neg_th_1', 'dice_neg_th_2', 'dice_pos_th_1', 'dice_pos_th_2'),
                 ('dice_neg_uncertainty_th_1', 'dice_neg_uncertainty_th_2', 'dice_pos_uncertainty_th_1', 'dice_pos_uncertainty_th_2')))
    for path in paths:
        name = path.split('/')[1].split('.')[0]
        print(name)
        c_df = pd.read_csv(path)
        x = list(c_df[c_df.columns[0]])
        for it_val in enumerate(cols):
            # for i, (c_cols, c_cols_uncertainty) in enumerate(cols):
            # for i, (col1_neg, col2_neg, col1_pos, col2_pos) in enumerate(cols):
            if uncertainty:
                i, (c_cols, c_cols_uncertainty) = it_val
                col1_neg, col2_neg, col1_pos, col2_pos = c_cols
                col1_neg_unc, col2_neg_unc, col1_pos_unc, col2_pos_unc = c_cols_uncertainty
            else:
                i, (col1_neg, col2_neg, col1_pos, col2_pos) = it_val
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            y1_val_neg, y1_std_neg = get_val_and_std(c_df, col1_neg)
            y2_val_neg, y2_std_neg = get_val_and_std(c_df, col2_neg)
            y1_val_pos, y1_std_pos = get_val_and_std(c_df, col1_pos)
            y2_val_pos, y2_std_pos = get_val_and_std(c_df, col2_pos)
            if uncertainty:
                y1_val_neg_unc, y1_std_neg_unc = get_val_and_std(c_df, col1_neg_unc)
                y2_val_neg_unc, y2_std_neg_unc = get_val_and_std(c_df, col2_neg_unc)
                y1_val_pos_unc, y1_std_pos_unc = get_val_and_std(c_df, col1_pos_unc)
                y2_val_pos_unc, y2_std_pos_unc = get_val_and_std(c_df, col2_pos_unc)

            if not uncertainty:
                if 'central_focus' in name:
                    l = np.arange(len(x))
                    ax[0][0].errorbar(l, y1_val_neg, yerr=y1_std_neg, fmt='-o', capsize=3.)
                    ax[1][0].errorbar(l, y2_val_neg, yerr=y2_std_neg, fmt='-o', capsize=3., c='orange')
                    ax[1][0].set_xticks(l, x, rotation=25, fontsize=8)
                    ax[0][1].errorbar(l, y1_val_pos, yerr=y1_std_pos, fmt='-o', capsize=3., c='red')
                    ax[1][1].errorbar(l, y2_val_pos, yerr=y2_std_pos, fmt='-o', capsize=3., c='green')
                    ax[1][1].set_xticks(l, x, rotation=25, fontsize=8)
                else:
                    ax[0][0].errorbar(x, y1_val_neg, yerr=y1_std_neg, fmt='-o', capsize=3.)
                    ax[1][0].errorbar(x, y2_val_neg, yerr=y2_std_neg, fmt='-o', capsize=3., c='orange')
                    ax[1][0].set_xticks(x, x, rotation=25, fontsize=8)
                    ax[0][1].errorbar(x, y1_val_pos, yerr=y1_std_pos, fmt='-o', capsize=3., c='red')
                    ax[1][1].errorbar(x, y2_val_pos, yerr=y2_std_pos, fmt='-o', capsize=3., c='green')
                    ax[1][1].set_xticks(x, x, rotation=25, fontsize=8)
            else:
                if 'central_focus' in name:
                    l = np.arange(len(x))
                    ax[0][0].errorbar(l, y1_val_neg, yerr=0, fmt='-')
                    ax[0][0].errorbar(l, y1_val_neg_unc, yerr=0, fmt='-', c='black', label='High confidence')
                    ax[1][0].errorbar(l, y2_val_neg, yerr=0, fmt='-', c='orange')
                    ax[1][0].errorbar(l, y2_val_neg_unc, yerr=0, fmt='-', c='black')
                    ax[0][1].errorbar(l, y1_val_pos, yerr=0, fmt='-', c='red')
                    ax[0][1].errorbar(l, y1_val_pos_unc, yerr=0, fmt='-', c='black')
                    ax[1][1].errorbar(l, y2_val_pos, yerr=0, fmt='-', c='green')
                    ax[1][1].errorbar(l, y2_val_pos_unc, yerr=0, fmt='-', c='black')
                    ax[1][0].xaxis.set_ticks(l)
                    ax[1][0].xaxis.set_ticklabels(x, rotation=25, fontsize=8)
                    ax[1][1].xaxis.set_ticks(l)
                    ax[1][1].xaxis.set_ticklabels(x, rotation=25, fontsize=8)
                else:
                    ax[0][0].errorbar(x, y1_val_neg, yerr=0, fmt='-')
                    ax[0][0].errorbar(x, y1_val_neg_unc, yerr=0, fmt='-', c='black', label='High confidence')
                    ax[1][0].errorbar(x, y2_val_neg, yerr=0, fmt='-', c='orange')
                    ax[1][0].errorbar(x, y2_val_neg_unc, yerr=0, fmt='-', c='black')
                    ax[0][1].errorbar(x, y1_val_pos, yerr=0, fmt='-', c='red')
                    ax[0][1].errorbar(x, y1_val_pos_unc, yerr=0, fmt='-', c='black')
                    ax[1][1].errorbar(x, y2_val_pos, yerr=0, fmt='-', c='green')
                    ax[1][1].errorbar(x, y2_val_pos_unc, yerr=0, fmt='-', c='black')
                    ax[1][0].set_xticks(x, x, rotation=25, fontsize=8)
                    ax[1][1].set_xticks(x, x, rotation=25, fontsize=8)

            ax[0][0].xaxis.set_label_position('top')
            ax[0][0].set_xlabel('Subsiding abnormalities', labelpad=10, fontsize=12)
            ax[0][0].xaxis.label.set_backgroundcolor('magenta')
            ax[0][1].xaxis.set_label_position('top')
            ax[0][1].set_xlabel('Intensifying abnormalities', labelpad=10, fontsize=12)
            ax[0][1].xaxis.label.set_backgroundcolor('yellow')
            ax[0][1].yaxis.set_label_position('right')
            ax[0][1].set_ylabel('All\nabnormalities', rotation=270, labelpad=30, fontsize=12)
            ax[1][1].yaxis.set_label_position('right')
            ax[1][1].set_ylabel('Pronounced\nabnormalities', rotation=270, labelpad=30, fontsize=12)

            trained_span = name_to_trained_span[name]
            ax[0][0].axvspan(trained_span[0], trained_span[1], color='green', alpha=0.25)
            ax[0][1].axvspan(trained_span[0], trained_span[1], color='green', alpha=0.25)
            ax[1][0].axvspan(trained_span[0], trained_span[1], color='green', alpha=0.25)
            ax[1][1].axvspan(trained_span[0], trained_span[1], color='green', alpha=0.25)

            # plt.ylim(bottom=0.)
            fig.supxlabel(name_to_param[name], fontsize=15, y=0.0001)
            fig.supylabel(idx_to_measure[i], fontsize=15)
            # fig.suptitle(name)
            # fig.legend(loc='lower right')
            # plt.show()
            plt.savefig(f'spectrum_sheets/{name}_{idx_to_measure[i]}{"_uncertainty" if uncertainty else ""}.png')
            # exit()
            ax[0][0].clear()
            ax[0][1].clear()
            ax[1][0].clear()
            ax[1][1].clear()
            plt.clf()
            plt.close()


def convert_Chexpert_to_nib():
    # folders = ['Chexpert/CheXpert-v1.0 batch 2 (train 1)', 'Chexpert/CheXpert-v1.0 batch 3 (train 2)', 'Chexpert/CheXpert-v1.0 batch 4 (train 3)']
    folders = ['Chexpert/CheXpert-v1.0 batch 1 (validate & csv)/valid']
    dst = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images'
    for folder in folders:
        print(f'Working on folder {folder}')
        files = glob(f'{folder}/patient*/study*/*_frontal.*')
        for file_path in tqdm(files):
            n = file_path.split(folder)[-1]
            new_path = dst + f'{n.split(".")[0]}.nii.gz'
            img = Image.open(file_path).convert("L")
            arr = np.array(img.getdata()).reshape((img.height, img.width)).T
            arr = arr.astype(np.uint8)
            arr_nif = nib.Nifti1Image(arr, AFFINE_DCM)
            new_dir = new_path.split('/view')[0]
            os.makedirs(new_dir, exist_ok=True)
            nib.save(arr_nif, new_path)
            # os.remove(p)


def prepare_Benny_files():
    folders = [r'BennyCases\Pediatric', r'BennyCases\Neonatal']
    for folder in folders:
        print(f'Working on folder {folder}')
        paths = glob(fr'{folder}/Disk*/*')
        for path in paths:
            print(fr'Checking path {path}')
            f_dic = pydicom.dcmread(path)
            data = f_dic.pixel_array
            data = data.squeeze().astype(float)

            # data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
            # mono = f_dic.PhotometricInterpretation
            # if mono == 'MONOCHROME1':
            #     data = 255 - data
            # data = data.astype(np.uint8).T
            #
            study_date = f_dic.StudyDate
            disk_num = path.split('Disk')[1][0]
            disk_str = f'Disk{disk_num}'
            #
            dir_name = os.path.dirname(path)
            file_name = os.path.basename(path)
            file_name = f'{disk_num}_{file_name}'
            path = fr'{dir_name}/{file_name}'
            dst_path = fr'{path.replace(disk_str, "images")}_{study_date}.nii.gz'
            # nif = nib.Nifti1Image(data, AFFINE_DCM)
            # nib.save(nif, dst_path)

            seg = np.ones_like(data).T
            seg_dst_path = dst_path.replace("images", "images_segs").replace(".nii.gz", "_seg.nii.gz")
            seg_nif = nib.Nifti1Image(seg, AFFINE_DCM)
            nib.save(seg_nif, seg_dst_path)

            # print(mono)
            # try:
            #     view_pos = f_dic.ViewPosition
            #     if view_pos not in {'AP', 'PA', 'LL', 'CHEST PA', 'CHEST AP'}:
            #         try:
            #             protocol_name = f_dic.ProtocolName
            #             print(f'protocol_name: {protocol_name}')
            #         except:
            #             try:
            #                 code_meaning = f_dic.ViewCodeSequence[0].CodeMeaning
            #                 print(f'code_meaning: {code_meaning}')
            #             except:
            #                 try:
            #                     processing_type = f_dic.ImageProcessingType
            #                     print(f'processing_type: {processing_type}')
            #                 except:
            #                     print("NOTHING ########")
            #     else:
            #         print(f'view pos: {view_pos}')
            # except:
            #     try:
            #         protocol_name = f_dic.ProtocolName
            #         print(f'protocol_name: {protocol_name}')
            #     except:
            #         try:
            #             code_meaning = f_dic.ViewCodeSequence[0].CodeMeaning
            #             print(f'code_meaning: {code_meaning}')
            #         except:
            #             try:
            #                 processing_type = f_dic.ImageProcessingType
            #                 print(f'processing_type: {processing_type}')
            #             except:
            #                 print("NOTHING ########")
            continue
            if 'lateral' in cxr_type:
                print(path)
    # path = 'BennyDiscs/Pediatric/Disk1/65'
    # print(d)
    # print("################")
    # # path = 'BennyDiscs/Pediatric/Disk1/44'
    # # d = pydicom.dcmread(path)
    # path = 'BennyDiscs/Pediatric/Disk1/66'
    # d = pydicom.dcmread(path).ViewCodeSequence[0].CodeMeaning
    # print(d)


def remove_partial_diffusion_pairs():
    folders = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/' + f for f in os.listdir('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs')]
    # folders = ['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/inpainted_healthy']
    diff_im = torch.zeros((768, 768))
    nif_diff = nib.Nifti1Image(diff_im.numpy(), AFFINE_DCM)

    for f in folders:
        trig = True
        print(f'Working on folder {f}')
        for d in tqdm(os.listdir(f)):
            cur_folder = f + '/' + d

            # os.remove(f'{cur_folder}/difference_map.nii.gz')
            # nib.save(nif_diff, f'{cur_folder}/difference_map.nii.gz')

            if os.path.exists(f'{cur_folder}/difference_map.pt'):
                if trig:
                    print("Found!")
                    trig = False
                os.remove(f'{cur_folder}/difference_map.pt')
                nib.save(nif_diff, f'{cur_folder}/difference_map.nii.gz')

            lst_dir = os.listdir(cur_folder)

            if len(lst_dir) != 5:
                print(f'Removing incomplete folder {d} with the files: {lst_dir}')
                shutil.rmtree(cur_folder)
                continue

            for file in lst_dir:
                cur_path = f'{cur_folder}/{file}'
                if not os.path.exists(cur_path):
                    print(f'Removing broken symlink: {cur_path}')
                    shutil.rmtree(cur_folder)
                    break


def convert_pngs_to_CT():
    orig_dir = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\images\images'
    out_dir = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\CT_images'
    slice_names = os.listdir(orig_dir)
    slice_names_set = set([n.split('_')[0] for n in slice_names])
    case_dict = {n: sorted([r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\images\images\\' + slice_n for slice_n in slice_names if slice_n.startswith(n)], key=lambda x: int(x.split('_')[-1].split('.')[0])) for n in slice_names_set}

    i = 0

    for n, slices in case_dict.items():
        first_slice = imageio.imread(slices[0])
        ct_data = np.zeros((len(slices), first_slice.shape[0], first_slice.shape[1]))

        for j in range(len(slices)):
            c_slice = imageio.imread(slices[j])
            ct_data[j] = c_slice

        ct_data = np.transpose(ct_data, (2, 1, 0))
        ct_data = np.flip(ct_data, axis=2)

        c_nif = nib.Nifti1Image(ct_data, AFFINE_DCM)
        nib.save(c_nif, fr'{out_dir}\{n}.nii.gz')

        exit()

        # reader = sitk.ImageSeriesReader()
        # reader.SetFileNames(slices)
        # vol = reader.Execute()
        # sitk.WriteImage(vol, fr'{out_dir}\{n}.nii.gz')
        # i += 1
        # if i == 2:
        #     exit()


def copy_lymph_files():
    p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/test_scans'
    d_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/test_scans/new'
    for f in tqdm(os.listdir(p)):
        if not f.endswith('.nii'):
            continue
        c_p = f'{p}/{f}'
        nif = nib.load(c_p)

        d_name = f'{d_p}/{f}.gz'
        nib.save(nif, d_name)


def flip_DRRs():
    p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs'
    folders = [f'volume-{i}' for i in [13,14,15,16,17,18,26,27,4,48,49,50,51,52,68]]
    for f in folders:
        c_p = f'{p}/{f}'
        print(f'Working on {c_p}')
        for pair_dir in os.listdir(c_p):
            print(pair_dir)
            c_pair_dir = f'{c_p}/{pair_dir}'

            prior_p = f'{c_pair_dir}/prior.nii.gz'
            current_p = f'{c_pair_dir}/current.nii.gz'
            diff_p = f'{c_pair_dir}/difference_map.nii.gz'

            for n_p in [prior_p, current_p, diff_p]:
                c_nib = nib.load(n_p)
                c_data = c_nib.get_fdata()
                c_aff = c_nib.affine

                c_data = np.flip(c_data, axis=1)
                n_nib = nib.Nifti1Image(c_data, c_aff)

                nib.save(n_nib, n_p)


def convert_LUNA():
    import SimpleITK as sitk

    # LUNA_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\LUNA_scans'
    LUNA_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans_new'
    # files_p = rf'{LUNA_p}\subset0'
    files_p = rf'{LUNA_p}/mhds'

    files_lst = []
    for d in os.listdir(files_p):
        for f in os.listdir(f'{files_p}/{d}'):
            if f.endswith('.mhd'):
                files_lst.append(f'{d}/{f}')

    # files_lst = [f for f in os.listdir(files_p) if f.endswith('.mhd')]

    i = 0

    for f in tqdm(files_lst):
        # Load the MetaImage file (.mhd)
        # mhd_path = files_p + rf"\{f}"
        mhd_path = files_p + f"/{f}"
        image = sitk.ReadImage(mhd_path)

        if image.GetSpacing()[-1] > 1.:
            continue

        # Save as a NIfTI file (.nii.gz)
        # nifti_path = LUNA_p + rf"\LUNA_scan{i}.nii.gz"
        # nifti_path = LUNA_p + rf"\{'.'.join(f.split('.')[:-1])}.nii.gz"
        f = f.split('/')[-1]
        nifti_path = LUNA_p + f"/{'.'.join(f.split('.')[:-1])}.nii.gz"
        sitk.WriteImage(image, nifti_path)

        i += 1

    print(f"{i} CTs passed filtering")


def create_filtered_LUNA_subsets():
    pd.set_option('display.max_colwidth', None)
    df_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/TCIA_LIDC-IDRI_20200921-nbia-digest.xlsx'

    LUNA_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans'
    LUNA_cases = set('.'.join(f.split('.')[:-2]) for f in os.listdir(LUNA_p) if f.endswith('.nii.gz'))
    id_col = 'Series Instance UID'
    model_col = 'Manufacturer Model Name'
    # manufacturer_col = 'Manufacturer'

    luna_df = pd.read_excel(df_p)
    luna_df = luna_df[['Series Instance UID', 'Manufacturer', 'Manufacturer Model Name', 'Software Versions']]
    luna_df = luna_df[luna_df[id_col].isin(LUNA_cases)]

    # grouped = luna_df.groupby('Manufacturer Model Name')
    # print(grouped[id_col].first())

    man_model_names = luna_df[model_col].unique()
    # man_names = luna_df[manufacturer_col].unique()
    for n in man_model_names:
        model_df = luna_df[luna_df[model_col] == n]
        if len(model_df) < 13:
            continue
        # dir_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/{n}'
        dir_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_models/{n}'
        os.makedirs(dir_p, exist_ok=True)
        for i in range(len(model_df)):
            ct_n = model_df.iloc[i][id_col]
            old_p = f'{LUNA_p}/{ct_n}.nii.gz'
            new_p = f'{dir_p}/{ct_n}.nii.gz'
            os.symlink(old_p, new_p)
    exit()


def sample_unchosen_ICU_pairs():
    num_pairs = 15
    icu_pairs_dirs = [
        r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_nitzan\images',
        r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_nitzan\images2',
        r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_nitzan\images3',
        r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_sigal\images',
        r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_sigal\images2',
        r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_sigal\images3'
    ]
    chosen_pairs_dirs = [
        r'C:\Users\sharp\PycharmProjects\CXRDifferencesLabeling\BennyPairs',
        r'C:\Users\sharp\PycharmProjects\CXRDifferencesLabeling\BennyPairs2',
        r'C:\Users\sharp\PycharmProjects\CXRDifferencesLabeling\BennyPairs3',
        r'C:\Users\sharp\PycharmProjects\CXRDifferencesLabeling\BennyPairs4',
        r'C:\Users\sharp\PycharmProjects\CXRDifferencesLabeling\BennyPairs5',
        r'C:\Users\sharp\PycharmProjects\CXRDifferencesLabeling\BennyPairs6',
    ]

    chosen_pairs = []
    for d in chosen_pairs_dirs:
        for p_d in os.listdir(d):
            chosen_pairs.extend(os.listdir(f'{d}/{p_d}'))

    icu_pairs = []
    for d in icu_pairs_dirs:
        icu_pairs.extend(os.listdir(d))

    chosen_pairs = set(chosen_pairs)
    icu_pairs = set(icu_pairs)

    unchosen_pairs = list(icu_pairs.difference(chosen_pairs))

    shuffle(unchosen_pairs)
    # new_pairs = unchosen_pairs[:num_pairs]
    new_pairs = unchosen_pairs
    print(sorted(new_pairs))

    print("#######")
    print(sorted(list(chosen_pairs)))
    print(len(chosen_pairs))


def measure_LUNA_cases_resolution():
    import SimpleITK as sitk

    luna_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans'
    spacings = []

    for f in os.listdir(luna_p):
        c_p = f'{luna_p}/{f}'

        itk_f = sitk.ReadImage(c_p)
        spacings.append(itk_f.GetSpacing())

    spacings = np.array(spacings)
    print(f'Shape = {spacings.shape}')
    print(f'Means = {np.mean(spacings, axis=0)}')
    print(f'Max = {np.max(spacings, axis=0)}')
    print(f'Min = {np.min(spacings, axis=0)}')


def create_joint_lobes_seg():
    paths = [
        r'for_consolidation_presentation/1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797_lung_lower_lobe_left.nii.gz',
        r'for_consolidation_presentation/1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797_lung_lower_lobe_right.nii.gz',
        r'for_consolidation_presentation/1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797_lung_upper_lobe_left.nii.gz',
        r'for_consolidation_presentation/1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797_lung_upper_lobe_right.nii.gz',
        r'for_consolidation_presentation/1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797_lung_middle_lobe_right.nii.gz'
    ]

    segs = []
    for p in paths:
        seg = nib.load(p)
        aff = seg.affine
        seg = seg.get_fdata().astype(bool)
        segs.append(seg)

    joint_seg = np.zeros_like(segs[0]).astype(int)

    for i, seg in enumerate(segs):
        joint_seg[seg] = i + 1

    joint_nif = nib.Nifti1Image(joint_seg, aff)
    nib.save(joint_nif, r'for_consolidation_presentation/joint_lobes_seg.nii.gz')


def remove_nan_pairs():
    d = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pneumothorax_training'

    cases = sorted(os.listdir(d))
    cases_len = len(cases)
    for i, case in enumerate(cases):
        if i < 154:
            continue
        print(f"Working on case {case}\n#{i + 1}/{cases_len}")
        case_p = f'{d}/{case}'
        for pair in tqdm(sorted(os.listdir(case_p))):
            pair_p = f'{case_p}/{pair}'

            prior_p = f'{pair_p}/prior.nii.gz'
            current_p = f'{pair_p}/current.nii.gz'
            diff_p = f'{pair_p}/diff_map.nii.gz'

            if (not os.path.exists(prior_p)) or (not os.path.exists(current_p)) or (not os.path.exists(diff_p)):
                print(f"Incomplete pair directory. Removing {pair_p}")
                shutil.rmtree(pair_p)
                continue

            prior_d = nib.load(prior_p).get_fdata()
            current_d = nib.load(current_p).get_fdata()
            diff_d = nib.load(diff_p).get_fdata()

            flag = False
            if np.any(np.isnan(prior_d)):
                print(f'!! Found nan in path {prior_p} !!')
                flag = True
            if np.any(np.isnan(current_d)):
                print(f'!! Found nan in path {current_p} !!')
                flag = True
            if np.any(np.isnan(diff_d)):
                print(f'!! Found nan in path {diff_p} !!')
                flag = True
            if np.max(prior_d) <= 0:
                print(f'!! Non positive max in path {prior_p} !!')
                flag = True
            if np.max(current_d) <= 0:
                print(f'!! Non positive max in path {current_p} !!')
                flag = True
            if flag:
                print(f'### Removing folder {pair_p} ###')
                shutil.rmtree(pair_p)


def filter_high_resolution_cases():
    cases_dir = '/cs/labs/josko/sahar_aharon/xray_diff_proj/dataset/dataset/train_fixed'

    total_cases = 0
    removed_cases = 0

    for d1 in tqdm(sorted(os.listdir(cases_dir))):
        d1_p = f'{cases_dir}/{d1}'
        for d2 in os.listdir(d1_p):
            d2_p = f'{d1_p}/{d2}'
            for nif in os.listdir(d2_p):
                total_cases += 1

                nif_p = f'{d2_p}/{nif}'
                nifti = nib.load(nif_p)
                data = nifti.get_fdata()
                header = nifti.header
                spacings = header['pixdim'][1:4]

                spacings_condition = any(spacings[i] > 1. for i in range(3))
                intensity_condition = bool(np.min(data) == np.max(data) or np.any(np.isnan(data)))
                if spacings_condition or intensity_condition:
                    print(f"Removed case {nif_p}")
                    os.remove(nif_p)

                    removed_cases += 1

            if len(os.listdir(d2_p)) == 0:
                shutil.rmtree(d2_p)

        if len(os.listdir(d1_p)) == 0:
            shutil.rmtree(d1_p)

    print(f'Finished! Removed {removed_cases} / {total_cases}')


def create_symlinks_for_dataset():
    src_p = '/cs/labs/josko/sahar_aharon/xray_diff_proj/dataset/dataset/train_fixed'
    dst_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans'

    for d1 in tqdm(sorted(os.listdir(src_p))):
        d1_p = f'{src_p}/{d1}'
        for d2 in os.listdir(d1_p):
            d2_p = f'{d1_p}/{d2}'
            for nif in os.listdir(d2_p):
                nif_p = f'{d2_p}/{nif}'
                new_nif_p = f'{dst_p}/{nif}'

                os.symlink(nif_p, new_nif_p)


def crop_devices():
    segs_and_scans_ps = [
        (r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CTs_with_devices\pacemaker1_Case3CT.nii.gz',
         r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CTs_with_devices\Case3CT.nii.gz'),
        (r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CTs_with_devices\port1_volume-106.nii.gz',
         r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CTs_with_devices\volume-106.nii.gz'),
        (r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CTs_with_devices\electrode1_volume-109.nii.gz',
         r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CTs_with_devices\volume-109.nii.gz')
    ]

    for seg_p, scan_p in segs_and_scans_ps:
        seg_nif = nib.load(seg_p)
        scan_nif = nib.load(seg_p)
        aff = seg_nif.affine

        seg = seg_nif.get_fdata()
        scan = scan_nif.get_fdata()

        seg_coords = seg.nonzero().T
        min_coords, max_coords = np.amin(seg_coords, axis=1), np.amax(seg_coords, axis=1)
        print(min_coords)
        print(max_coords)





if __name__ == '__main__':
    crop_devices()
    exit()

    img1 = torch.tensor(nib.load(r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_sigal\images\p3_1.nii.gz').get_fdata().T)
    img2 = torch.tensor(nib.load(r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\cases_sigal\images\p3_2.nii.gz').get_fdata().T)
    diff = torch.abs(img2 - img1)
    plt.imshow(img1.cpu().squeeze(), cmap='gray')
    plt.show()
    plt.imshow(img2.cpu().squeeze(), cmap='gray')
    plt.show()
    plt.imshow(diff.cpu().squeeze(), cmap='gray')
    plt.show()
    exit()
    # cases_dict = get_cases_with_keywords_in_report(keywords_dict)
    # with open('diseases_stats.txt', 'w') as f:
    #     for disease, dis_dic in cases_dict.items():
    #         f.write(f'{disease}\n')
    #         f.write(f'Number of found X-rays: {dis_dic["len"]}\n')
    #         f.write(str(dis_dic["files"]) + '\n')
    #         f.write('\n')
    # print('---finished---')
    # print(len(glob(files_path + '/p*/p*/s*/*.dcm')))
    # a = pydicom.read_file("physionet.org/files/p10/p10002428/pair_s59659695_s59891001/BL_s59659695/51b5892c-e54ed6e6-59ff70db-fd0b8509-1792398e.dcm")
    # print(a)
    # update_high_proximity_study_pairs_subdirectories()
    # create_cxr14_no_finding_csv()
    # filter_AP_files_CXR14()
    # create_PadChest_no_finding_csv()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    generate_spectrum_graphs()

    # create_new_PadChest_relevant_images_csv()
    exit()
    create_PadChest_specific_abnormalities_csv()
    exit()

    from ast import literal_eval

    # csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_train.csv'
    # csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/labels_test.csv'
    csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/relevant_images.csv'
    # csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-negbio.csv.gz'
    df = pd.read_csv(csv_path)

    # locs = list(df['Localizations'])
    # all_locs = []
    # for l in locs:
    #     all_locs.extend(literal_eval(l))
    # all_locs = set(all_locs)
    # print('\n'.join(all_locs))
    # exit()
    print(df[df['ImageID']=='34206382049694892110316005354435392828_ppwgb4.png'])
    # print('e2b2f50550d1dc76448410e7ff060251' in list(df['image_id']))
    # create_PadChest_no_finding_csv()
    pass


# a = pydicom.read_file("C:/Users/sharp/PycharmProjects/LongitudinalCXRAnalysis/physionet.org/files/p10/p10000032/pair_s53911762_s56699142/BL_s53911762/68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714.dcm")
# print(a.StudyTime)
# print(a.ProcedureCodeSequence[0].CodeMeaning)
# print(a.ViewPosition)
# print(a.StudyDate)
# print(a.ImageLaterality)
# print(a)

# a1 = "21881023"
# a2 = "21891205"
# print(days_between(a1, a2))