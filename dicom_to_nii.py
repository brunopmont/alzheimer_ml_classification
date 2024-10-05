
import os
from helpers import *
import numpy as np
import nibabel as nib
import pydicom
import ants
import SimpleITK as sitk

# DIRETÓRIOS

DIR_BASE = os.path.abspath('/mnt/d/ADNI/ADNI1')
DIR_DICOM = os.path.abspath(os.path.join(DIR_BASE, 'ADNI1_Screening', 'ADNI'))
DIR_RAW = os.path.join(DIR_BASE, 'ADNI_nii_raw')
DIR_OUTPUT = os.path.join(DIR_BASE, 'ADNI_nii_processed')

#FUNÇÕES

def arq_nii(name):
    return name + ".nii.gz"

def get_f_dir(directory):
    sub_item = os.listdir(directory)
    directory = os.path.abspath(os.path.join(directory, sub_item[0]))
    return directory

def load_dicom_series(input_folder):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(input_folder)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

def save_as_nifti(image, output_file):
    sitk.WriteImage(image, output_file)

def reorient_image(image):
    # Reorienta a imagem para o sistema padrão RAS (Right, Anterior, Superior)
    return sitk.DICOMOrient(image, 'IAL')

def convert_dicom_to_nifti(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carrega a série DICOM
    image = load_dicom_series(input_folder)

    # Reorienta a imagem para o padrão RAS
    #image = reorient_image(image)

    # Formar nome de saída
    output_name = arq_nii(os.path.basename(input_folder))

    # Nome do arquivo NIfTI de saída
    output_file = os.path.abspath(os.path.join(output_folder, output_name))

    # Salva no formato NIfTI
    save_as_nifti(image, output_file)

# CONVERSÃO
input_folder = DIR_DICOM
output_folder = DIR_RAW

cont = 0

for folder in os.listdir(DIR_DICOM):
    item = os.path.join(DIR_DICOM, folder, 'MP-RAGE')
    item = get_f_dir(item)
    item = get_f_dir(item)
    
    convert_dicom_to_nifti(folder, DIR_RAW)

    cont+=1

print(f'Foram convertidas {cont} imagens!')