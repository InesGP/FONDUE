# IMPORTS
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import numpy as np
import time
import sys
import logging
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import SimpleITK as sitk
from data_loader.load_neuroimaging_data_final import OrigDataThickSlices, load_and_rescale_image_sitk, volshow
from data_loader.checkpoints import load_model, get_best_ckp_path
from pathlib import Path

import importlib
from utils import gzip_this, is_anisotropic, arguments_setup, filename_wizard, model_loading_wizard
HELPTEXT = """
Script to generate denoised.mgz using Deep Learning. /n

Dependencies:

    albumentations==1.3.0
    h5py==3.7.0
    imageio==2.19.3
    lpips==0.1.4
    matplotlib==3.5.2
    nibabel==5.1.0
    numpy==1.21.5
    opencv_python==4.7.0.72
    pandas==1.4.4
    Pillow==9.5.0
    PyYAML==6.0
    scikit_image==0.19.2
    scikit_learn==1.0.2
    scipy==1.9.1
    torch==1.13.1
    torchvision==0.14.1
    tqdm==4.64.1
    XlsxWriter==3.0.3
    torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

Original Author: Walter Adame Gonzalez
VANDAlab - Douglas Mental Health University Institute
PI - Mahsa Dadar, PhD., MSc.
PI - Yashar Zeighami, PhD.
Date: Apr-04-2024
"""

def options_parse():
    """
    Command line option parser
    """
    parser = argparse.ArgumentParser(description=HELPTEXT, epilog='$Id: fondue_denoising, v 1.0 2024-04-18$')
    # ======== INPUT/OUTPUT FILENAMES OPTIONS =============
    parser.add_argument('--in_name', '--input_name', dest='iname', type=str, 
                        help='name of file to process',
                        default=None)
    parser.add_argument('--out_name', '--output_name', dest='oname', type=str,
                        help='Output filename')
    parser.add_argument('--iname_new', type=str, default=None,
                        help='If save_new_input=True, it will store the preprocessed image. Default: orig.mgz')
    parser.add_argument('--csv_file', type=str, 
                        help='File that contains the list of volumes to denoise. Should be in csv format, one line per string',
                        default = None)
    parser.add_argument('--suffix_type', type=str,
                        help='Type of suffix to be used', 
                        default = "simple",
                        choices = ['simple', 'detailed'])
    parser.add_argument('--suffix', type=str,
                        help='Suffix of the denoised file')
    parser.add_argument('--ext', default = None, 
                        help = "Output file extension. By default is the same as the input extension")
    parser.add_argument('--save_new_input', type=bool, default = True, 
                        help = "If true, it will save the intensity-rescaled/reshaped/re-oriented (or a combination of these) version that was computed before producing the denoised image"
                        "Default is False")
    parser.add_argument('--noise_info_file', type=str, 
                        help='Name of the .txt file that will store the noise stdev information',
                        default=None)
    
    # ======== PRE-PROCESSING OPTIONS =======
    parser.add_argument('--intensity_range_mode', type=int, default = 0,
                        help = "Voxel intensity range for the generated images. 0 is for [0-255]. 2 is for using original intensity range (not recommended)."
                        "1 is for [0-1]")
    parser.add_argument('--robust_rescale_input', type=bool, default = True,
                        help = "Perform rescaling of input intensity between 0-255 using histogram robust rescaling. If False rescaling will be simple rescaling using maximum and minimum values.")
    
    # ======== NETWORKS OPTIONS ==========
    parser.add_argument('--name', type = str, default='FONDUE_LT',
                        choices=['FONDUE_A_BN', 'FONDUE_A_NOBN',
                                 'FONDUE_B_BN', 'FONDUE_B_NOBN',
                                 'FONDUE_LT',
                                 'FONDUE_B1_BN', 'FONDUE_B1_NOBN',
                                 'FONDUE_B2_BN', 'FONDUE_B2_NOBN',
                                 'FONDUE_LT_X2', 
                                 "UNETVINN", "MCDNCNN"
                                 ],
                        help='model name')
    parser.add_argument('--no_cuda', action='store_true', default=False, 
                        help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help="Batch size for inference. Default: 1")
    parser.add_argument('--model_path', type=str,
                        help='path to the training checkpoints')
    
    sel_option = parser.parse_args()
    if sel_option.iname is None and sel_option.csv_file is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------/nERROR: Please specify data directory or input volume/n')
    
    return arguments_setup(sel_option)



def run_network(img_filename, zoom, orig_data, denoised_image, plane, params_model, model, logger, args):
    """
    

    Parameters
    ----------
    img_filename : str
        Full path to the neuroimaging file.
    zoom : tuple
        Tuple of floats corresponding to the original voxel sizes retrieved from nibabel.header.get_zooms().
    orig_data : numpy array
        3D array containing the image to be denoised.
    denoised_image : torch tensor
        Tensor that will contain the denoised image.
    plane : str
        Plane to be used for denoising. "Axial", "Coronal" or "Sagittal".
    params_model : dict
        Values to be used to create the DataLoader variable that will be the input to the network.
    model : pytorch model (nn.Module)
        DCCR model containing the architecture to the network.
    logger : logger file
        DESCRIPTION.
    args : argument parser object
        Will contain all the arguments from the argument parser.

    Returns
    -------
    denoised_image : torch tensor
        Tensor that contains the denoised image.

    """
    # Set up DataLoader
    test_dataset = OrigDataThickSlices(img_filename, orig_data, plane=plane)
    test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                  batch_size=params_model["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if params_model["use_cuda"]:
        model = model.to(device)
    best_ckp_path = get_best_ckp_path(args)
    print('Loading checkpoint from '+best_ckp_path)
    model = load_model(best_ckp_path, model, device)
    print('======CHECKPOINT LOADED======')
    model.eval()
    model.to(device)
    logger.info("{} model loaded.".format(plane))
    with torch.no_grad():
        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):
            images_batch = Variable(sample_batch["image"])
            images_batch = images_batch.permute(0, 3, 1, 2) #Reshape from [BS, input_h, input_w, thickslice_size] to [BS, thickslice_size, input_h, input_w]
            images_batch = images_batch.float() #Transform to float to fit the network
            # if params_model["use_cuda"]:
            images_batch = images_batch.to(device)
            # with torch.cuda.amp.autocast():
            temp,_,_,_,_,_,_ = model(images_batch, zoom)
            # temp,_,_,_,_,_,_ = model(images_batch, zoom)
            output = temp
            temp = torch.squeeze(output)
            temp = output
            if plane == "Second":
                temp = temp.permute(3, 0, 2, 1)
                denoised_image[:, start_index:start_index + temp.shape[1], :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[1]
            elif plane == "Third":
                temp = temp.permute(2, 3, 0, 1)
                denoised_image[:, :, start_index:start_index + temp.shape[2], :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[2]
            else:
                temp = temp.permute(0, 2, 3, 1)
                denoised_image[start_index:start_index + temp.shape[0], :, :, :] += torch.mul(temp.cpu(), 1)
                start_index += temp.shape[0]
            logger.info("--->Batch {} {} Axis Testing Done.".format(batch_idx, plane))
    return denoised_image


def resunetcnn(img_filename, save_as, save_as_new_orig, logger, args):
    """
    

    Parameters
    ----------
    img_filename : str
        Full path to the input of the image (image to be denoised).
    save_as : str
        Full output filename (without the extension -e.g. without the ".mnc"-) where the denoised image will be written to.
    save_as_new_orig : str
        Full filename (without the file extension) where the new input image will be written to. This will be only used in case that the --keep_dims flag is False.
    logger : logger file
        File containing the log to the pipeline.
    args : argument parser object
        Contains the input arguments to the script.

    Returns
    -------
    None.

    """
    start_total = time.time()
    options = options_parse()
    txt_filename = options.noise_info_file
    archs = importlib.import_module("archs." + options.name)
    logger.info("Reading volume {}".format(img_filename))
    sitk_img = sitk.ReadImage(os.path.join(img_filename))
    basename_in, basename_out, basename_innew, ext_in, ext_out, ext_innew, is_gzip_in, is_gzip_out, is_gzip_innew = filename_wizard(img_filename, save_as, save_as_new_orig)
    orig_data, orig_img_interp_itk, orig_zoom, max_orig, min_orig = load_and_rescale_image_sitk(os.path.join(img_filename), logger=logger, is_eval = True, intensity_rescaling = options.robust_rescale_input)
    h, w, c = orig_data.shape
    orig_img_interp_itk
    orig_data = (orig_data - orig_data.min()) / (orig_data.max() - orig_data.min())
    orig_data_noisy = orig_data
    z1, z2, z3 = orig_zoom[0:3]
    h_out = int(h)
    w_out = int(w)
    c_out = int(c)
    anisotropic, irr_pos = is_anisotropic(z1, z2, z3)
    
    # =====  LOADING CONFIGURATION FOR NETWORK ====
    model, params_model = model_loading_wizard(options, args, archs, logger)
    

    # Generate the final tensor for storing denoised image
    num_planes_clean = 1
    denoised_image = torch.zeros((h_out, w_out, c_out, num_planes_clean), dtype=torch.float)

    # Axial Prediction
    # if (anisotropic and irr_pos == "Axial") or not anisotropic: 
    start = time.time()
    
    #run_network(img_filename, zoom, orig_data, denoised_image, plane, params_model, model, logger, args, anisotropic)
    
    denoised_image = run_network(img_filename, orig_zoom,
                            orig_data_noisy, denoised_image, "First",
                            params_model, model, logger, args)
    
    # volshow(denoised_image.cpu().numpy())
    # volshow(orig_data_noisy)
    
    logger.info("First axis tested in {:0.4f} seconds".format(time.time() - start))

    # Coronal Prediction
    # if (anisotropic and irr_pos == "Coronal") or not anisotropic:
    start = time.time()
    denoised_image = run_network(img_filename, orig_zoom,
                            orig_data_noisy, denoised_image, "Second",
                            params_model, model, logger, args)
    logger.info("Second axis tested in {:0.4f} seconds".format(time.time() - start))
    
    # volshow(denoised_image.cpu().numpy())
    # volshow(orig_data_noisy)

    # # Sagittal Prediction
    # if (anisotropic and irr_pos == "Sagittal") or not anisotropic:
    start = time.time()
    denoised_image = run_network(img_filename, orig_zoom,
                            orig_data_noisy, denoised_image, "Third",
                            params_model, model, logger, args)
    logger.info("Third axis tested in {:0.4f} seconds".format(time.time() - start))
    
    # volshow(denoised_image.cpu().numpy())
    # volshow(orig_data_noisy)
    
    denoised_image = denoised_image/3.0
    mask = denoised_image.squeeze().cpu().numpy() > 10
    mask = mask.astype(np.uint8)
    noise = (orig_data_noisy*2-1) - (denoised_image.squeeze().cpu().numpy() * 2 - 1)
    noise_std = noise.std()
    
    if int(args.intensity_range_mode) == 0:
        denoised_image = denoised_image * 255.
        orig_data_noisy = orig_data_noisy * 255.
    elif int(args.intensity_range_mode) == 2:
        denoised_image = (denoised_image*(max_orig - min_orig)) + min_orig
        orig_data_noisy = (orig_data_noisy*(max_orig - min_orig)) + min_orig
    elif int(args.intensity_range_mode) < 0 or int(args.intensity_range_mode) > 2:
        print("Warning: intensity_range_mode not valid. Valid values are only 0 (0-255), 1 (orig_min - orig_max), and 2 (0-1)")
        print("Storing output with intensity range mode 0 (0-255) ....")
        denoised_image = denoised_image * 255.
        orig_data_noisy = orig_data_noisy * 255.
        
    # Saving image
    denoised_image = torch.squeeze(denoised_image)
    denoised_image = denoised_image.detach().cpu().numpy()
    denoised_image = np.transpose(denoised_image, (2, 1, 0))
    denoised_image_itk = sitk.GetImageFromArray(denoised_image)
    denoised_image_itk.CopyInformation(sitk_img)
    
    orig_data_noisy = np.transpose(orig_data_noisy, (2, 1, 0))
    orig_data_noisy = sitk.GetImageFromArray(orig_data_noisy)
    orig_data_noisy.CopyInformation(sitk_img)
    
    den_fname = basename_out + ext_out
    # with open(txt_filename, "w") as file:
    #     file.write(str(noise_std))
    
    logger.info("========== SAVING DENOISED IMAGE IN {}".format(den_fname))
    sitk.WriteImage(denoised_image_itk, den_fname)
    # orig_fname = basename_innew + ext_innew
    # if options.save_new_input:
    #     logger.info("Saving original MRI rescaled  {}".format(orig_fname))
    #     sitk.WriteImage(orig_img_interp_itk, orig_fname)
    #     if is_gzip_innew:
    #         gzip_this(orig_fname)
    if is_gzip_out:
        gzip_this(den_fname)
        
    
    logger.info("Saved clean MRI to {}".format(den_fname))
    logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))
    
    
if __name__ == "__main__":
    options = options_parse()
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    if options.csv_file is not None:
        import csv
        # with open(options.csv_file, newline='', encoding='utf-8-sig') as f:
        with open(options.csv_file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                iname = row[0]
                fname = Path(iname)
                basename = os.path.join(fname.parent, fname.stem)
                output_dir = os.path.join(fname.parent.parent,"FONDUE_B")
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                suffix = options.suffix
                oname = os.path.join(output_dir,fname.stem + "_" + suffix)
                iname_new = basename + "_orig" 
                if not os.path.exists(oname+options.ext):
                    if not options.keep_dims or options.save_new_input:
                        resunetcnn(iname, os.path.join(oname), os.path.join(iname_new) , logger, options)
                    else:
                        resunetcnn(iname, os.path.join(oname), "_" , logger, options)
                else:
                    print("Skipping file ... "+oname+options.ext)
    else:
        if options.ext == ".nii" or options.ext == ".nii.gz" or options.ext == ".mgh" or options.ext == ".mgz" or options.ext == ".mnc":
            valid_extensions = True
        else:
            valid_extensions = False
            
        # Set up the logger
        
        
        if valid_extensions:
                resunetcnn(options.iname, os.path.join(options.oname), os.path.join(options.iname_new), logger, options)
        else:
            print("Invalid input file extension. Valid extensions are .nii, .nii.gz, .mgh, .mgz, .mnc")
            sys.exit(0)
    
        sys.exit(0)
