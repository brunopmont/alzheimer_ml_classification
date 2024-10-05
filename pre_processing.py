import os
import numpy as np
import ants

#FUNÇÕES
#Winsorize -> reduz outliers, litando os percentis inf e sup
def winsorize_image(image_data, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(image_data, lower_percentile)
    upper_bound = np.percentile(image_data, upper_percentile)
    winsorized_data = np.clip(image_data, lower_bound, upper_bound)
    return winsorized_data

#Normalization -> valores de voxels entre 0 e 1
def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return normalized_data

# DIRETÓRIOS
DIR_BASE = os.path.abspath('/mnt/d/ADNI/ADNI1')
DIR_DICOM = os.path.join(DIR_BASE, 'ADNI1 Screening')
DIR_RAW = os.path.join(DIR_BASE, 'ADNI_nii_raw')
DIR_OUTPUT = os.path.join(DIR_BASE, 'ADNI_nii_processed')
DIR_MASK = os.path.join(DIR_BASE, 'mni_icbm152_nlin_asym_09c')

template_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
mask_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

template = ants.image_read(template_path, reorient='IAL')
mask = ants.image_read(mask_path, reorient='IAL')

cont = 0

#MÃO NA MASSA
for file in os.listdir(DIR_RAW):
    #if (cont == 2): break

    cont += 1

    img_path = os.path.abspath(os.path.join(DIR_RAW, file))
    image = ants.image_read(img_path)
    data = image.numpy()

    print(f"\nIMAGEM nº {cont}\n")

    # Winsorizing
    data = winsorize_image(data)

    print("WINSOREZED")

    origin = image.origin
    spacing = image.spacing
    direction = image.direction

    image = ants.from_numpy(data, origin=origin, spacing=spacing, direction=direction)

    # Bias Field Correction -> reduzir "imperfeições do equipamento"
    image = ants.n4_bias_field_correction(image, shrink_factor=4)

    print("BIAS CORRECTED")

    # Additional Winsorize Image -> reduz outliers, litando os percentis inf e sup
    data = image.numpy()

    data = winsorize_image(data)

    image = ants.from_numpy(data, origin=origin,spacing=spacing, direction=direction)

    print("WINSOREZED AGAIN")

    # Translation Alignment -> alinhar as imagens entre si
    registration = ants.registration(
        fixed=template,  # Imagem fixa
        moving=image,  # Imagem q muda
        type_of_transform='Translation'  # Tipo de transformação
    )

    # Obter a imagem pós translation (registrada)
    warped_image = registration['warpedmovout']

    print("TRANSLATION")

    # Rigid Transform -> alinha as imagens entre si, além de rotacionar e ajustar, mantendo proporções
    registration = ants.registration(
        fixed=template,  # Imagem fixa
        moving=warped_image,  # Imagem q muda
        type_of_transform='Rigid' # Tipo de transformação
    )

    # Obter a imagem pós rigid (registrada)
    warped_image = registration['warpedmovout']
    
    print("RIGID")

    # Affine Transform
    registration = ants.registration(
        fixed=template,  # Imagem fixa
        moving=warped_image,  # Imagem q muda
        type_of_transform='Affine' # Tipo de transformação
    )

    # Obter a imagem pós affine (registrada)
    warped_image = registration['warpedmovout']

    print("AFFINE")

    # Deformable Symmetric Normalization (SyN)
    registration = ants.registration(
        fixed=template,  # Imagem fixa
        moving=warped_image,  # Imagem móvel
        type_of_transform='SyN' # Tipo de transformação
    )

    # Obter a imagem pós SyN (registrada)
    warped_image = registration['warpedmovout']

    print("SYN SYN")

    #Ajustar template e máscara
    template_copy = template
    mask_copy = mask

    registration = ants.registration(
        fixed=warped_image,
        moving=template_copy,
        type_of_transform='SyN',
    )

    template_registered = registration['warpedmovout']

    brain_mask = ants.apply_transforms(
        moving=mask_copy,
        fixed=registration['warpedmovout'],
        transformlist=registration['fwdtransforms'],
        interpolator='nearestNeighbor'
    )

    brain_mask_dilated = ants.morphology(brain_mask, radius=4, operation='dilate', mtype='binary')

    print("MASK DONE")

    # Application of Brain Mask
    brain_masked = ants.mask_image(warped_image, brain_mask_dilated)

    print("BRAIN EXTRACTED")

    # Normalizar o array numpy da imagem extraída
    normalized_data = normalize_image(brain_masked.numpy())

    print("NORMALIZED")

    # Converter o array numpy de volta para uma imagem ANTs
    normalized_image = ants.from_numpy(normalized_data, origin=brain_masked.origin, spacing=brain_masked.spacing, direction=brain_masked.direction)

    #Caminho saída
    output_path = os.path.abspath(os.path.join(DIR_OUTPUT, file))

    # Salvar a imagem normalizada
    ants.image_write(normalized_image, output_path)

    print("SALVOU BORA BILL CALABRESO")
print(f"\n\n{cont} IMAGENS PROCESSADAS COM SUCESSO!")