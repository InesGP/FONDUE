# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:19:55 2022

@author: walte
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import lpips
import torch
import nibabel as nib
import numpy as np
from pytorch_msssim import ssim, ms_ssim
from skimage import exposure
import SimpleITK as sitk
from typing import Optional, Type, Tuple, Union, Iterable, cast

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, device):
        mse = torch.nn.MSELoss()
        a = img1.to(device, dtype = torch.float32)
        b = img2.to(device, dtype = torch.float32)
        mse_total = mse(a,b)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_total))
        return psnr

def getvolmask(path_denoised, img_mask):
    img_denoised = load_and_conform_image(path_denoised) #Read the image
    img_denoised = (img_denoised - img_denoised.min())/(img_denoised.max() - img_denoised.min())
    img_denoised = img_mask*img_denoised
    return img_denoised

def getvol(path_denoised):
    img_denoised = load_and_conform_image(path_denoised) #Read the image
    img_denoised = 255.0*(img_denoised - img_denoised.min()) / \
        (img_denoised.max() - img_denoised.min())
    img_denoised = img_denoised.astype(np.single)
    return img_denoised

psnr = PSNR()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpipsdist = lpips.LPIPS(net='alex').to(device)
# flip = Hflip().to(device)

def is_conform(img, eps=1e-06):
    """
    Function to check if an image is already conformed or not (Dimensions: 256x256x256, Voxel size: 1x1x1, and
    LIA orientation.

    :param nibabel.MGHImage img: Loaded source image
    :param float eps: allowed deviation from zero for LIA orientation check (default 1e-06).
                      Small inaccuracies can occur through the inversion operation. Already conformed images are
                      thus sometimes not correctly recognized. The epsilon accounts for these small shifts.
    :return: True if image is already conformed, False otherwise
    """
    ishape = img.shape
    max_size = max(ishape)
    if len(ishape) > 3 and ishape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(img.shape[3]) + ') not supported!')

    # check dimensions
    if ishape[0] != max_size or ishape[1] != max_size or ishape[2] != max_size:
        return False

    # check voxel size
    izoom = img.header.get_zooms()
    min_zoom = min(izoom)
    # min_zoom = 1.2
    if izoom[0] != min_zoom or izoom[1] != min_zoom or izoom[2] != min_zoom:
        return False

    # check orientation LIA
    iaffine = img.affine[0:3, 0:3] + [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    if np.max(np.abs(iaffine)) > 0.0 + eps:
        return False

    return True

def check_affine_in_nifti(img, logger=None):
    """
    Function to check affine in nifti Image. Sets affine with qform if it exists and differs from sform.
    If qform does not exist, voxelsizes between header information and information in affine are compared.
    In case these do not match, the function returns False (otherwise returns True.

    :param nibabel.NiftiImage img: loaded nifti-image
    :return bool: True, if: affine was reset to qform
                            voxelsizes in affine are equivalent to voxelsizes in header
                  False, if: voxelsizes in affine and header differ
    """
    check = True
    message = ""

    if img.header['qform_code'] != 0 and np.max(np.abs(img.get_sform() - img.get_qform())) > 0.001:
        message = "#############################################################" \
                  "\nWARNING: qform and sform transform are not identical!\n sform-transform:\n{}\n qform-transform:\n{}\n" \
                  "You might want to check your Nifti-header for inconsistencies!" \
                  "\n!!! Affine from qform transform will now be used !!!\n" \
                  "#############################################################".format(img.header.get_sform(),
                                                                                         img.header.get_qform())
        # Set sform with qform affine and update best affine in header
        img.set_sform(img.get_qform())
        img.update_header()

    else:
        # Check if affine correctly includes voxel information and print Warning/Exit otherwise
        vox_size_head = img.header.get_zooms()
        aff = img.affine
        xsize = np.sqrt(aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0])
        ysize = np.sqrt(aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1])
        zsize = np.sqrt(aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2])

        if (abs(xsize - vox_size_head[0]) > .001) or (abs(ysize - vox_size_head[1]) > .001) or (abs(zsize - vox_size_head[2]) > 0.001):
            message = "#############################################################\n" \
                      "ERROR: Invalid Nifti-header! Affine matrix is inconsistent with Voxel sizes. " \
                      "\nVoxel size (from header) vs. Voxel size in affine: " \
                      "({}, {}, {}), ({}, {}, {})\nInput Affine----------------\n{}\n" \
                      "#############################################################".format(vox_size_head[0],
                                                                                             vox_size_head[1],
                                                                                             vox_size_head[2],
                                                                                             xsize, ysize, zsize,
                                                                                             aff)
            check = False

    if logger is not None:
        logger.info(message)

    else:
        print(message)

    return check

def conform(img, order=3, conform_type = 0, intensity_rescaling = False):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    # from nibabel.freesurfer.mghformat import MGHHeader
    
    # ishape = img.shape
    # s1 = ishape[0]
    # s2 = ishape[1]
    # s3 = ishape[2]
    # max_shape = max(s1, s2, s3)
    
    # izoom = img.header.get_zooms()
    # z1 = izoom[0]
    # z2 = izoom[1]
    # z3 = izoom[2]
    # min_zoom = min(z1, z2, z3)
    
    
    # cwidth = max_shape
    # csize = min_zoom
    
    # if conform_type == 2:
    #     csize = 0.25
    #     cwidth = 880
    
    # if len(ishape)>3:
    #     ishape = ishape[:3]
    # if len(izoom)>3:
    #     izoom = izoom[:3]
    # h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    # h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    # h1.set_zooms([csize, csize, csize])
    # h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    # h1['fov'] = cwidth
    # h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    # mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    
    img_ras = nib.funcs.as_closest_canonical(img, enforce_diag = True)
    mapped_data = img_ras.get_fdata()
    # new_img = nib.MGHImage(new_data, h1.get_affine(), h1)
    new_img = nib.Nifti1Image(mapped_data, img_ras.affine)
    new_img.set_data_dtype(np.uint8)

    return new_img

def map_image(
        img: nib.analyze.SpatialImage,
        out_affine: np.ndarray,
        out_shape: tuple,
        ras2ras: Optional[np.ndarray] = None,
        order: int = 1,
        dtype: Optional[Type] = None
) -> np.ndarray:
    """Map image to new voxel space (RAS orientation).

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        the src 3D image with data and affine set
    out_affine : np.ndarray
        trg image affine
    out_shape : tuple[int, ...], np.ndarray
        the trg shape information
    ras2ras : Optional[np.ndarray]
        an additional mapping that should be applied (default=id to just reslice)
    order : int
        order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    dtype : Optional[Type]
        target dtype of the resulting image (relevant for reorientation, default=same as img)

    Returns
    -------
    np.ndarray
        mapped image data array
    
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    if ras2ras is None:
        ras2ras = np.eye(4)

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image

    out_shape = tuple(out_shape)
    # if input has frames
    if image_data.ndim > 3:
        # if the output has no frames
        if len(out_shape) == 3:
            if any(s != 1 for s in image_data.shape[3:]):
                raise ValueError(
                    f"Multiple input frames {tuple(image_data.shape)} not supported!"
                )
            image_data = np.squeeze(image_data, axis=tuple(range(3, image_data.ndim)))
        # if the output has the same number of frames as the input
        elif image_data.shape[3:] == out_shape[3:]:
            # add a frame dimension to vox2vox
            _vox2vox = np.eye(5, dtype=vox2vox.dtype)
            _vox2vox[:3, :3] = vox2vox[:3, :3]
            _vox2vox[3:, 4:] = vox2vox[:3, 3:]
            vox2vox = _vox2vox
        else:
            raise ValueError(
                    f"Input image and requested output shape have different frames:"
                    f"{image_data.shape} vs. {out_shape}!"
                )

    if dtype is not None:
        image_data = image_data.astype(dtype)

    return affine_transform(
        image_data, inv(vox2vox), output_shape=out_shape, order=order
    )


def getscale(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.0,
        f_high: float = 0.999
) -> Tuple[float, float]:
    """Get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.

    Equivalent to how mri_convert conforms images.

    Parameters
    ----------
    data : np.ndarray
        image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    f_low : float
        robust cropping at low end (0.0 no cropping, default)
    f_high : float
        robust cropping at higher end (0.999 crop one thousandth of high intensity voxels, default)

    Returns
    -------
    float src_min
        (adjusted) offset
    float
        scale factor

    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0 !")

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cumulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print("ERROR: rescale upper bound not found")

    src_max = idx * bin_size + src_min

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print(
        "rescale:  min: "
        + format(src_min)
        + "  max: "
        + format(src_max)
        + "  scale: "
        + format(scale)
    )

    return src_min, scale


def scalecrop(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        src_min: float,
        scale: float
) -> np.ndarray:
    """Crop the intensity ranges to specific min and max values.

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    src_min : float
        minimal value to consider from source (crops below)
    scale : float
        scale value by which source will be shifted

    Returns
    -------
    np.ndarray
        scaled image data
    
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print(
        "Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max())
    )

    return data_new

def conform_fix(
        img, order=3, conform_type = 0, intensity_rescaling = False, keep_dims = False, dtype=None
) -> nib.MGHImage:
    """Python version of mri_convert -c.

    mri_convert -c by default turns image intensity values
    into UCHAR, reslices images to standard position, fills up slices to standard
    256x256x256 format and enforces 1mm or minimum isotropic voxel sizes.

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        loaded source image
    order : int
        interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    conform_vox_size : VoxSizeOption
        conform image the image to voxel size 1. (default), a
        specific smaller voxel size (0-1, for high-res), or automatically
        determine the 'minimum voxel size' from the image (value 'min').
        This assumes the smallest of the three voxel sizes.
    dtype : Optional[Type]
        the dtype to enforce in the image (default: UCHAR, as mri_convert -c)
    conform_to_1mm_threshold : Optional[float]
        the threshold above which the image is conformed to 1mm
        (default: ignore).

    Returns
    -------
    nib.MGHImage
        conformed image

    Notes
    -----
    Unlike mri_convert -c, we first interpolate (float image), and then rescale
    to uchar. mri_convert is doing it the other way around. However, we compute
    the scale factor from the input to increase similarity.

    """
    from nibabel.freesurfer.mghformat import MGHHeader

    
    ishape = img.shape
    s1 = ishape[0]
    s2 = ishape[1]
    s3 = ishape[2]
    max_shape = max(s1, s2, s3)
    conformed_img_size = max_shape
    izoom = img.header.get_zooms()
    z1 = izoom[0]
    z2 = izoom[1]
    z3 = izoom[2]
    min_zoom = min(z1, z2, z3)
    conformed_vox_size = min_zoom
    
    h1 = MGHHeader.from_header(
        img.header
    )  # may copy some parameters if input was MGH format
    h1.set_data_shape([s1, s2, s3, 1])
    if not keep_dims:
        h1.set_data_shape([conformed_img_size, conformed_img_size, conformed_img_size, 1])
        h1.set_zooms(
            [conformed_vox_size, conformed_vox_size, conformed_vox_size])  # --> h1['delta']  
        h1["Mdc"] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
        h1["fov"] = conformed_img_size * conformed_vox_size
        h1["Pxyz_c"] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
        # Here, we are explicitly using MGHHeader.get_affine() to construct the affine as
        # MdcD = np.asarray(h1['Mdc']).T * h1['delta']
        # vol_center = MdcD.dot(hdr['dims'][:3]) / 2
        # affine = from_matvec(MdcD, h1['Pxyz_c'] - vol_center)
    affine = h1.get_affine()

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # target scalar type and dtype
    if intensity_rescaling:
        sctype = np.uint8 if dtype is None else np.obj2sctype(dtype, default=np.uint8)
    else:
        sctype = np.float32
    target_dtype = np.dtype(sctype)
    
    
    src_min, scale = 0, 1.0
    # get scale for conversion on original input before mapping to be more similar to
    # mri_convert
    if (
        img.get_data_dtype() != np.dtype(np.uint8)
        or img.get_data_dtype() != target_dtype
    ):
        src_min, scale = getscale(np.asanyarray(img.dataobj), 0, 255)

    kwargs = {}
    if sctype != np.uint:
        kwargs["dtype"] = "float"
    mapped_data = map_image(img, affine, h1.get_data_shape(), order=order, **kwargs)
    
    if intensity_rescaling:
        if img.get_data_dtype() != np.dtype(np.uint8) or (
            img.get_data_dtype() != target_dtype and scale != 1.0
        ):
            scaled_data = scalecrop(mapped_data, 0, 255, src_min, scale)
            # map zero in input to zero in output (usually background)
            scaled_data[mapped_data == 0] = 0
            mapped_data = scaled_data

        if target_dtype == np.dtype(np.uint8):
            mapped_data = np.clip(np.rint(mapped_data), 0, 255)
    
    new_img = nib.MGHImage(sctype(mapped_data), affine, h1)

    # make sure we store uchar
    try:
        new_img.set_data_dtype(target_dtype)
    except nib.freesurfer.mghformat.MGHError as e:
        if "not recognized" in e.args[0]:
            codes = set(
                k.name
                for k in nib.freesurfer.mghformat.data_type_codes.code.keys()
                if isinstance(k, np.dtype)
            )
            print(
                f'The data type is not recognized for MGH images, switching '
                f'to "{new_img.get_data_dtype()}" (supported: {tuple(codes)}).'
            )

    return new_img

def load_and_conform_image(img_filename, interpol=3, logger=None, is_eval = False, intensity_rescaling=True, conform_type = 0):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0=min_vox_size+max_im_size, 1=std(1.0/256)) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    zoom = orig.header.get_zooms()
    ishape = orig.shape
    if len(orig.shape) == 4:
        orig = orig.slicer[:,:,:,0]
    max_shape = max(ishape)
    if not is_conform(orig):

        if logger is not None:
            if conform_type == 0:
                logger.info('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                logger.info('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        else:
            if conform_type == 0:
                print('Conforming image to UCHAR, RAS orientation, and '+str(zoom)+'mm isotropic voxels. Final Volume Shape: '+str(max_shape)+'x'+str(max_shape)+'x'+str(max_shape)+' voxels')
            else:
                print('Conforming image to UCHAR, RAS orientation, and 1.0mm isotropic voxels. Final Volume Shape: 256x256x256 voxels')
        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")
        orig = conform_fix(orig, interpol, intensity_rescaling=intensity_rescaling)
    orig = np.asanyarray(orig.dataobj)
    return orig

def load_and_conform_image_sitk(img_filename, interpol=3, logger=None, is_eval = False, conform_type = 2, intensity_rescaling = False,  sfs = 1.0, sfc = 1.0, sfa = 1.0):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param str conform_type: (0: Cubic shape of dims = max(img_filename.shape) and voxdim of minimum voxdim. 1: Cubic shape of dims 256^3. 2: Keep dimensions) 
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = sitk.ReadImage(img_filename)
    orig = sitk.DICOMOrient(orig, 'RAS')
    orig_img = sitk.GetArrayFromImage(orig)
    orig_img = np.transpose(orig_img, (2, 1, 0)) #Here should be AX,COR,SAG
    return orig_img


def get_mask(path_mask):
    img_mask = load_and_conform_image(path_mask, intensity_rescaling=False) #Read the image
    img_mask = img_mask >= 1 
    img_mask = img_mask.astype(np.single)
    return img_mask

def compute_metrics(path_denoised, path_gt_scaled, path_mask = None,
                    use_mask = True, exposure_match = False,
                    verbose = False):
    
    try: 
        if (path_mask is not None and use_mask):
            img_mask = get_mask(path_mask)
            igt = getvolmask(path_gt_scaled, img_mask)
            id1 = getvolmask(path_denoised, img_mask)
        else:
            igt = getvol(path_gt_scaled)
            id1 = getvol(path_denoised)
            mask = igt > (10)
            mask = mask.astype(np.single)
            id1 = id1 * mask 
            igt = igt * mask
        if exposure_match:
            id1 = exposure.match_histograms(id1, igt)
        
        # import matplotlib.pyplot as plt
        # import numpy as np
        
        # # Assuming you have your numpy arrays igt, id1, and id2 defined
        
        # # Slices for the first row
        # slice_1 = igt[:, :, 120]
        # slice_2 = id1[:, :, 120]
        
        # # Slices for the second row
        # slice_4 = igt[:, 120, :]
        # slice_5 = id1[:, 120, :]
        
        # # Slices for the third row
        # slice_7 = igt[120, :, :]
        # slice_8 = id1[120, :, :]
        
        # # Create a 3x3 grid
        # fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        # # Plotting the slices
        # axes[0, 0].imshow(slice_1)
        # axes[0, 0].set_title('igt[:,:,120]')
        # axes[0, 1].imshow(slice_2)
        # axes[0, 1].set_title('id1[:,:,120]')
        
        # axes[1, 0].imshow(slice_4)
        # axes[1, 0].set_title('igt[:,120,:]')
        # axes[1, 1].imshow(slice_5)
        # axes[1, 1].set_title('id1[:,120,:]')
        
        # axes[2, 0].imshow(slice_7)
        # axes[2, 0].set_title('igt[120,:,:]')
        # axes[2, 1].imshow(slice_8)
        # axes[2, 1].set_title('id1[120,:,:]')
        
        # plt.tight_layout()
        # plt.show()
        id1 = torch.from_numpy(id1).to(device).type(torch.float16)
        igt = torch.from_numpy(igt).to(device).type(torch.float16) 
        lpips_value = [0]
        slice_number = id1.shape[0]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i in range(slice_number):
                    
                    if verbose:
                        print("Slice i = "+ str(i))
                    
                    id1c = id1[:,:,i]
                    id1c = torch.unsqueeze(id1c,0)
                    id1c = torch.unsqueeze(id1c,0)
                    
                    igtc = igt[:,:,i]
                    igtc = torch.unsqueeze(igtc,0)
                    igtc = torch.unsqueeze(igtc,0)
                    
                    #  ==COMPUTING LPIPS======
                    if igtc.max() > 0:
                        lpd1 = lpipsdist(id1c,igtc).cpu().detach().numpy().item()
                        lpips_value[0] += lpd1
                    
                lpips_value[0] /= slice_number
                
                psnr_value = [0]
                ssim_value = [0]
                ms_ssim_value = [0]
                id1 = (id1 - id1.min()) / (id1.max() - id1.min())
                igt = (igt - igt.min()) / (igt.max() - igt.min())
                #======= COMPUTING PSNR ===========
                psnr_value[0] = psnr(id1,igt, device).cpu().detach().numpy().item()
                
                #======= COMPUTING SSIM ===========
                ssim_value[0] = ssim(torch.unsqueeze(id1,0),torch.unsqueeze(igt,0), data_range = 1).cpu().detach().numpy().item()
                
                #======= COMPUTING MSSSIM =========
                ms_ssim_value[0] = ms_ssim(torch.unsqueeze(id1,0), torch.unsqueeze(igt,0), data_range = 1).cpu().detach().numpy().item()
                
                # if verbose:
                #     print("v=v=v=v=v=v=v=v=v=LPIPS=v=v=v=v=v=v=v=v=v=v")
                #     print(lpips_value)
                #     print("v=v=v=v=v=v=v=v=v=PSNR=v=v=v=v=v=v=v=v=v=v")
                #     print(psnr_value)
                #     print("v=v=v=v=v=v=v=v=v=SSIM=v=v=v=v=v=v=v=v=v=v")
                #     print(ssim_value)
                #     print("v=v=v=v=v=v=v=v=v=MSSIM=v=v=v=v=v=v=v=v=v=v")
                #     print(ms_ssim_value)
                
                # # df = pd.DataFrame({'image' : image,'lpips': lpips_value, 'psnr': psnr_value, 'ssim': ssim_value, 'msssim': ms_ssim_value})
                # # # Insert the first row (variable names)
                # # df.loc[-1] = ['image','lpips', 'psnr', 'ssim', 'msssim']
                # # df.index = df.index + 1  # shifting index
                # # df = df.sort_index()  # sorting by index
                
                # # Save the dataframe to excel
                # df.to_excel("MEMPRAGE_010_metrics_finetuned.xlsx", index=False, header=False)
                torch.cuda.empty_cache()
                return lpips_value, psnr_value, ssim_value, ms_ssim_value
    except Exception as e:
        print(f"=== An error {e} ocurred during the computation of metrics for {path_denoised} subject")
        lpips_value[0] = 999.0
        psnr_value[0] = 999.0
        ssim_value[0] = 999.0
        ms_ssim_value[0] = 999.0
        return lpips_value, psnr_value, ssim_value, ms_ssim_value
    
    

