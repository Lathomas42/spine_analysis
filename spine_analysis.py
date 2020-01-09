#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import scipy
from skimage.external.tifffile import TiffFile
import sys
import time
import logging
import tifffile as tf

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.ERROR)
import glob
import zarr
import multiprocessing
import tqdm
import caiman
import skimage
import sklearn
import scipy
from ScanImageTiffReader import ScanImageTiffReader
from skimage.external import tifffile as tif
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo


# In[ ]:
default_config = {
    "max_shifts": (10, 10),  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    "strides":  (48, 48),  # create a new patch every x pixels for pw-rigid correction
    "overlaps": (24, 24),  # overlap between pathes (size of patch strides+overlaps)
    "num_frames_split": 100,  # length in frames of each chunk of the movie (to be processed in parallel)
    "max_deviation_rigid": 3 ,  # maximum deviation allowed for patch with respect to rigid shifts
    "pw_rigid": False,  # flag for performing rigid or piecewise rigid motion correction
    "shifts_opencv": True,  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    "border_nan": 'copy',  # replicate values along the boundary (if True, fill in with NaN)
    # first do the structural data in the _2 folder
    "server_dir": "/home/silvio/recurrence/rotons/", #"/home/silvio/modulation/frankenshare/Silvio-Transfer/
    "prj": "20191209", #"20191128"
    "struct_nums":[2], #list of all _# for structural data
    "func_nums":[1], #list of all _# for functional data
    # functional data
    "nchannels" : 2,
    "nplanes" : 4,
    "nimgs" : 27,
}

def denoise_mov(movie,ncomp=1,batch_size=1000):
    """
    Denoises movie using PCA.
    """
    num_frames,h,w = movie.shape
    frame_size = h*w
    frame_samples = np.reshape(movie,(num_frames,frame_size)).T

    ipca_f = sklearn.decomposition.IncrementalPCA(n_components=ncomp,batch_size=batch_size)
    ipca_f.fit(frame_samples)
    proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frame_samples))
    eigenseries = ipca_f.components_.T
    eigenframes=np.dot(proj_frame_vectors, eigenseries)
    movie2 = caiman.movie(np.reshape(np.float32(proj_frame_vectors.T), movie.shape))
    return movie2, eigenframes

def motion_correct_file( fn, cfg = default_config, n_iter=0,max_iter=10):
    try:
        mc = MotionCorrect(fn, dview=None, max_shifts=cfg['max_shifts'],
                      strides=cfg['strides'], overlaps=cfg['overlaps'],
                      max_deviation_rigid=cfg['max_deviation_rigid'],
                      shifts_opencv=cfg['shifts_opencv'], nonneg_movie=True,
                      splits_els=1,splits_rig=1,
                      border_nan=cfg['border_nan'], save_dir=cfg['save_dir'])
        mc.motion_correct(save_movie=True)
        return mc.mmap_file
    except:
        if n_iter < max_iter:
            print("Retrying %s"%fn)
            return motion_correct_file(fn,cfg,n_iter+1)
        else:
            return Exception("Failed max_iter: %s times"%max_iter)


class AlignmentHelper(object):
    def __init__(self,cfg=default_config):
        if cfg is str:
            cfg = json.load(open(cfg,'r'))

        self.dview = None
        self.max_shifts = cfg["max_shifts"]
        self.strides = cfg["strides":]
        self.overlaps = cfg["overlaps"]
        self.num_frames_split = cfg["num_frames_split"]
        self.max_deviation_rigid = cfg["max_deviation_rigid"]
        self.pw_rigid = cfg["pw_rigid"]
        self.shifts_opencv = cfg["shifts_opencv"]
        self.border_nan = cfg["border_nan"]
        self.server_dir = cfg["server_dir"]
        self.prj = cfg["prj"]
        self.struct_nums = cfg["struct_nums"]
        self.func_nums = cfg["func_nums"]
        # functional data
        self.nchannels = cfg["nchannels"]
        self.nplanes = cfg["nplanes"]
        self.nimgs = cfg["nimgs"]

        self.prj_dir = os.path.join(server_dir,prj)

        self.struc_fnames = { x: glob.glob(os.path.join(prj_dir, "%s*_%s/*.tif"%(prj,x))) for x in self.struct_nums }
        self.func_fnames = { x: glob.glob(os.path.join(prj_dir, "%s*_%s/*.tif"%(prj,x))) for x in self.func_nums }

        self.base_folder = '/home/silvio/Documents/logan/spine_project/%s'%prj
        self.save_dir=os.path.join(base_folder,"struct_mmaps/")

        save_dir=os.path.join(base_folder,"struct_mmaps/")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        print("loading test movie")
        self.testm = caiman.load([struc_fnames[0][0]])


    def align_structural_data(self):
        # example usage:
        # iter through all struc files
        for n in self.struc_fnames:
            struc_fnames = self.struc_fnames[n]
            print("Processing structural dataset %s" % n)
            pool = multiprocessing.Pool(8)
            mmap_files = list(tqdm.tqdm(pool.imap(motion_correct_file,struc_fnames),total=len(struc_fnames)))

            # take median filters of the newly aligned stacks and register them
            mmap_files = glob.glob(os.path.join(self.save_dir,"*.mmap"))
            out_tiffname = os.path.join(self.base_folder,"%s_struc_%s.tiff"%(self.prj,n))

            denoise_ds = np.zeros((len(mmap_files),self.testm.shape[1],self.testm.shape[2]),dtype=testm.dtype)
            aligned_ds = np.zeros((len(mmap_files),self.testm.shape[1],self.testm.shape[2]),dtype=testm.dtype)

            # go through all the mmap files
            for f in mmap_files:
                if f is list:
                    f = f[0]
                ind = int(f.split('/')[-1].split('_')[5]) - 1
                ds = caiman.load(f)
                dnm, ev = denoise_mov(ds,1,1000)
                denoise_ds[ind,:,:] = dnm.mean(axis=0)

                # remove the file
                os.remove(f)

            # save the denoised dataset
            #print("saving denoised data to %s"%out_tiffname)
            #tf.imsave(out_tiffname,denoise_ds)

            #align the denoised items to eachother
            aligned_ds[0,:,:] = out_zarr[denoised_dsn][0,:,:]
            for i in range(1,len(aligned_ds)):
                s,e,d = skimage.feature.register_translation(aligned_ds[i-1,:,:],denoised_ds[i,:,:],100)
                print(s)
                oi = scipy.ndimage.fourier_shift(np.fft.fftn(denoised_ds[i,:,:]),s)
                aligned_ds[i,:,:] = np.fft.ifftn(oi)

            # save the aligned dataset
            print("saving aligned data to %s"%out_tiffname)
            tf.imsave(out_tiffname,aligned_ds)

        def align_functional_data(self):
            # now go through functional files and take the structural data
            # first go through and save a file for each plane and each channel

            #%%  TEST CODE: start the cluster (if a cluster already exists terminate it)
            if self.dview is not None:
                caiman.stop_server(dview=self.dview)
                self.dview = None
            c, self.dview, n_processes = caiman.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            for c in range(nchannels):
                for p in range(nplanes):
                    basedir=os.path.join(base_folder,str(c),str(p))
                    if not os.path.isdir(basedir):
                        os.makedirs(basedir)

            def process_func(fn):
                nTries = 10
                vol=None
                while nTries > 0:
                    try:
                        reader = ScanImageTiffReader(fn)
                        vol = reader.data()
                        reader.close()
                        nTries -= 1
                    except:
                        print("retrying %s"%fn)

                for c in range(nchannels):
                    for p in range(nplanes):
                        basedir=os.path.join(base_folder,str(c),str(p))
                        tif.imsave(os.path.join(basedir,os.path.basename(fn)),vol[(p*2+c)::8,:,:],bigtiff=True)

            pool = multiprocessing.Pool(4)
            #i = 0
            #for fn in func_fnames:
            #    print(i/len(func_fnames)*100)
            #    process_func(fn)
            list(tqdm.tqdm(pool.imap(process_func,func_fnames),total=len(func_fnames)))


            # In[ ]:


            # now go through and make one master channel for each channel on each plane
            for c in range(nchannels):
                for p in range(nplanes):
                    basedir=os.path.join(base_folder,str(c),str(p))
                    fns = glob.glob(os.path.join(basedir,"%s_*_*_*_*.tif"%prj))
                    fns.sort()
                    ofn = os.path.join(base_folder,"%s_c%s_p%s.tif"%(prj,c,p))
                    m = caiman.load_movie_chain(fns)
                    m.save(ofn)


            # In[ ]:


            func_fnames
            reader = ScanImageTiffReader(struc_fnames[2])
            print(reader.metadata())


            # In[ ]:


            reader = ScanImageTiffReader(func_fnames[2])
            print(reader.metadata())


            # In[ ]:


            #%% start the cluster (if a cluster already exists terminate it)
            if 'dview' in locals():
                caiman.stop_server(dview=dview)
            c, dview, n_processes = caiman.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            dummymov = caiman.load([os.path.join(base_folder,"%s_c1_p1.tif"%(prj))])

            out_zarr = zarr.open(os.path.join(base_folder,"%s.zarr"%prj),'a')
            out_zarr.require_dataset('func_red_avg',shape=(nplanes,testm.shape[1],testm.shape[2]),dtype=testm.dtype)
            out_zarr.require_dataset('func_green_mov',shape=(nplanes,dummymov.shape[0],testm.shape[1],testm.shape[2]),dtype=testm.dtype)
            del dummymov

            mmaps = []
            # motion correct in each of the c1 (red) channels
            for p in range(nplanes):
                c1fn = os.path.join(base_folder,"%s_c1_p%s.tif"%(prj,p))
                c0fn = os.path.join(base_folder,"%s_c0_p%s.tif"%(prj,p))
                print(c1fn)

                # motion correct on the red channel
                mc = MotionCorrect([c1fn], dview=dview, max_shifts=max_shifts,
                                      strides=strides, overlaps=overlaps,
                                      max_deviation_rigid=max_deviation_rigid,
                                      shifts_opencv=shifts_opencv, nonneg_movie=True,
                                      splits_els=30,splits_rig=30,
                                      border_nan=border_nan)
                ret = mc.motion_correct(save_movie=False)

                # save the mean image of the red channel after correction
                out_zarr['func_red_avg'][p,:,:] = np.mean(mc.templates_rig,axis=0)

                # apply shifts to the green channel
                mmgreen = ret.apply_shifts_movie([c0fn])

                # save the green channel movie to the zarr
                out_zarr['func_green_mov'][p,:,:,:] = mmgreen
                del mmgreen
                del mc

            caiman.stop_server(dview=dview) # stop the server


            # In[ ]:


            # now go through the avg red channels and find where they are in the larger volume
            out_zarr = zarr.open(os.path.join(base_folder,"%s.zarr"%prj),'a')
            #out_zarr.require_dataset('func_red_shifts',shape=(nplanes,2),dtype=np.float)

            rettot = []
            for p in range(4):
                rets = []

                func_red =  cv2.medianBlur(out_zarr['func_red_avg'][p,:,:],3)
                func_red = (func_red - func_red.min()) / (func_red.max()-func_red.min())
                func_red = skimage.transform.resize(func_red, np.asarray(func_red.shape)*7.8//9.0)
                for i in range(len(out_zarr['struct_aligned_2'])):
                    struct_red = out_zarr['struct_aligned_2'][i,:,:]
                    struct_red = (struct_red - struct_red.min()) / (struct_red.max()-struct_red.min())
                    ret=cv2.matchTemplate((func_red*255.0).astype(np.uint8), (struct_red*255.0).astype(np.uint8), cv2.TM_SQDIFF_NORMED)
                    rets.append(cv2.minMaxLoc(ret))
                rettot.append(rets)


            # In[ ]:


            # can do some confirmation here that the planes are ~30 um apart
            planes = np.argmin(rettot[:,:,0],axis=1)
            out_zarr.require_dataset('func_green_planes',shape=planes.shape)
            out_zarr['func_green_planes'] = planes
            out_zarr.require_dataset('func_green_shifts',shape=(4,2))
            for p in range(4):
                # load the green data and scale it and shift it
                out_zarr['func_green_shifts'][p,:] = rettot[p,planes[p],2]
                #skimage.transform.resize(mmgreen,np.asarray(mmgreen.shape)*[1,7.8,7.8]//[1,9,9])
