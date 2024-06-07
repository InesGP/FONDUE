# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:08:13 2024

@author: walte
"""
import gzip
import os
from pathlib import Path
import sys
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import torch
import torch.nn as nn

def gzip_this(in_file):
    in_data = open(in_file, "rb").read() # read the file as bytes
    out_gz = in_file + ".gz" # the name of the compressed file
    gzf = gzip.open(out_gz, "wb") # open the compressed file in write mode
    gzf.write(in_data) # write the data to the compressed file
    gzf.close() # close the file
    # If you want to delete the original file after the gzip is done:
    os.unlink(in_file)
    
def is_anisotropic(z1, z2, z3):
    # find the largest value
    largest = max(z1, z2, z3)
    # determine which axis has the largest value
    if largest == z1:
        irr_pos = "Sagittal"
    elif largest == z2:
        irr_pos = "Coronal"
    else:
        irr_pos = "Axial"
    # find the smallest value
    smallest = min(z1, z2, z3)
    # compare the largest and smallest values
    if largest >= 2 * smallest:
        print("WARNING: Voxel size is at least twice as large in the largest dimension than in the smallest dimension. Will perform denoising only using the "+irr_pos+" plane.")
        return True, irr_pos
    else:
        return False, 0
    
def arguments_setup(sel_option):
    in_name = getattr(sel_option, "iname")
    model_name = getattr(sel_option, "name")
    irm = getattr(sel_option, "intensity_range_mode")
    rri = getattr(sel_option, "robust_rescale_input")
    suffix_type = getattr(sel_option, "suffix_type")
    if getattr(sel_option, "ext") is None:
        fname = Path(in_name)
        basename = os.path.join(fname.parent, fname.stem)
        ext = fname.suffix
        if ext == ".gz":
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)
            ext = fname2.suffix + ext
        elif ext == ".gzpi":
            fname2 = Path(basename)
            basename = os.path.join(fname2.parent, fname2.stem)
            ext = fname2.suffix + ".gz"
        setattr(sel_option, "ext", ext)
    
    
    if irm == 0:
        suffix_3 = "_irm0"
    elif irm == 1:
        suffix_3 = "_irm1"
    elif irm == 2:
        suffix_3 = "_irm2"
    if rri:
        suffix_4 = "_rri1"
    else:
        suffix_4 = "_rri0"
    settings_suffix = suffix_3 + suffix_4
    if suffix_type == "detailed":
        suffix = model_name + settings_suffix + ext
    else:
        suffix = model_name + ext
    pathname = os.path.dirname(sys.argv[0])
    model_path = os.path.join(pathname,"model_checkpoints",model_name+".pt")
    
    # set the default suffix name if it was not parsed as an argument
    if getattr(sel_option, "suffix") is None:
        setattr(sel_option, "suffix", suffix)
    
    if getattr(sel_option, "oname") is None:
        setattr(sel_option, "oname", basename + "_" + suffix)
    
    if getattr(sel_option, "model_path") is None:
        setattr(sel_option, "model_path", model_path)
        
    if getattr(sel_option, "iname_new") is None:
        setattr(sel_option, "iname_new", basename + "_preprocessed_" + suffix)
        
    return sel_option

def add_noise(x, noise='.'):
        noise_type = noise[0]
        noise_value = float(noise[1:])/100
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            # noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = abs(x.astype(np.float64) + noises.astype(np.float64))
        return x_noise
    
def filename_wizard(img_filename, save_as, save_as_new_orig):
    fname_in = Path(img_filename)
    fname_out = Path(save_as)
    fname_innew = Path(save_as_new_orig)
    
    basename_in = os.path.join(fname_in.parent, fname_in.stem)
    basename_out = os.path.join(fname_out.parent, fname_out.stem)
    basename_innew = os.path.join(fname_innew.parent, fname_innew.stem)
    
    ext_in = fname_in.suffix
    ext_out = fname_out.suffix
    ext_innew = fname_innew.suffix
    
    if ext_in == ".gz":
        fname2 = Path(basename_in)
        basename_in = os.path.join(fname2.parent, fname2.stem)
        ext_in = fname2.suffix   
        is_gzip_in = True
    else:
        is_gzip_in = False
    if ext_out == ".gz":
        fname2 = Path(basename_out)
        basename_out = os.path.join(fname2.parent, fname2.stem)
        ext_out = fname2.suffix  
        is_gzip_out = True
    else:
        is_gzip_out = False
    if ext_innew == ".gz":
        fname2 = Path(basename_innew)
        basename_innew = os.path.join(fname2.parent, fname2.stem)
        ext_innew = fname2.suffix
        is_gzip_innew = True
    else:
        is_gzip_innew = False
    return basename_in, basename_out, basename_innew, ext_in, ext_out, ext_innew, is_gzip_in, is_gzip_out, is_gzip_innew

def model_loading_wizard(options, args, archs, logger):
    pathname = os.path.dirname(sys.argv[0])
    with open(pathname+'/models/%s/config.yml' % options.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    
    cudnn.benchmark = True
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           input_channels = config['input_channels'],
                                           deep_supervision = config['deep_supervision'])
    
    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    logger.info("Cuda available: {}, # Available GPUS: {}, "
                "Cuda user disabled (--no_cuda flag): {}, "
                "--> Using device: {}".format(torch.cuda.is_available(),
                                              torch.cuda.device_count(),
                                              args.no_cuda, device))

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)
    
    model.eval()

    params_model = {'device': device, "use_cuda": use_cuda, "batch_size": args.batch_size,
                    "model_parallel": model_parallel} #modifications needed?
    return model, params_model