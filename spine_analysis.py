#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import argparse
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
import json

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
import configparser

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
    "server_dir": "/global/scratch/logan_thomas/",
    "output_dir": "/global/scratch/logan_thomas/output/",
    "prj": "20191209",
    "struct_nums":[2], #list of all _# for structural data
    "func_nums":[1], #list of all _# for functional data
}

class MetadataParser(object):
    def __init__(self, file):
        with ScanImageTiffReader(file) as reader:
            mds = reader.metadata()
            self.metadata = configparser.ConfigParser()
            self.metadata.read_string('[SIvars]\n'+mds[:mds.find('\n\n')])
            self.jsdata = json.loads(mds[mds.find('\n\n'):])

    def _get_var(self, varname):
        return self.metadata['SIvars'][varname]

    def get_nchannels(self):
        return self._get_var('SI.hChannels.channelSave').count(';')+1

    def get_channel_index(self,channel='red'):
        colors = self._get_var('SI.hChannels.channelMergeColor')[1:-1].split(';')
        for e,ci in enumerate(self._get_var('SI.hChannels.channelSave')[1:-1].split(';')):
            ind = int(ci)-1
            if colors[ind][1:-1] == channel:
                return ind
        return -1

    def get_num_slices(self):
        if self._get_var('SI.hFastZ.enable').lower() == 'true':
            return int(self._get_var('SI.hFastZ.numFramesPerVolume'))
        else:
            return int(self._get_var('SI.hStackManager.numSlices'))

    def get_frames_per_slice(self):
        if self._get_var('SI.hFastZ.enable').lower() == 'true':
            return int(self._get_var('SI.hFastZ.numVolumes'))
        else:
            return int(self._get_var('SI.hStackManager.framesPerSlice'))

    def get_pixelsize(self):
        return (int(self._get_var('SI.hRoiManager.linesPerFrame')),
                int(self._get_var('SI.hRoiManager.pixelsPerLine')))

    def get_zoom(self):
        return float(self._get_var('SI.hRoiManager.scanZoomFactor'))

def create_config(path):
    with open(path,'w') as outfile:
        json.dump(default_config,outfile,indent=2)

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

def motion_correct_file( fn, ind, save_dir, max_shifts, strides, overlaps, max_deviation_rigid, shifts_opencv, border_nan, subidx=slice(None,None,1), n_iter=0,max_iter=10):
    try:
        mc = MotionCorrect(fn, dview=None, max_shifts=max_shifts,
                      strides=strides, overlaps=overlaps,
                      max_deviation_rigid=max_deviation_rigid,
                      shifts_opencv=shifts_opencv, nonneg_movie=True,
                      splits_els=1,splits_rig=1,
                      border_nan=border_nan, subidx=subidx, save_dir=save_dir)
        mc.motion_correct(save_movie=False)
        return (ind, np.mean(mc.templates_rig,axis=0))
    except Exception as e:
        if n_iter < max_iter:
            print("Retrying %s due to %s"%(fn, e))
            return motion_correct_file(fn, save_dir, max_shifts, strides, overlaps, max_deviation_rigid, shifts_opencv, border_nan, n_iter+1)
        else:
            return Exception("Failed max_iter: %s times"%max_iter)


class AlignmentHelper(object):
    def __init__(self,cfg=default_config):
        if isinstance(cfg, str):
            cfg = json.load(open(cfg,'r'))

        self.dview = None
        self.max_shifts = cfg["max_shifts"]
        self.strides = cfg["strides"]
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
        self.num_proc=cfg["num_proc"]
        self.prj_dir = os.path.join(self.server_dir, self.prj)

        self.base_folder = os.path.join(cfg['output_dir'],self.prj)
        self.save_dir=os.path.join(self.base_folder,"struct_mmaps/")

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.fnames = dict()
        # get metadata
        self.cfgs = dict()
        for x in self.struct_nums + self.func_nums:
            self.cfgs[x] = dict()
            fns = glob.glob(os.path.join(self.prj_dir, "%s*_%s/*.tif"%(self.prj,x)))
            fns.sort()
            com_pre = os.path.commonprefix(fns)
            rev_fns = [f[::-1] for f in fns]
            com_suff = os.path.commonprefix(rev_fns)[::-1]
            inds = [ int(s[len(com_pre):-len(com_suff)]) for s in fns ]
            self.fnames[x] = list(zip(inds,fns))
            struct_metadata = MetadataParser(self.fnames[x][0])
            self.cfgs[x]['type'] = 'struct' if x in self.struct_nums else 'func'
            self.cfgs[x]['red_ind'] = struct_metadata.get_channel_index('red')
            self.cfgs[x]['green_ind'] = struct_metadata.get_channel_index('green')
            self.cfgs[x]['nchan'] = struct_metadata.get_nchannels()
            self.cfgs[x]['nz'] = struct_metadata.get_num_slices()
            self.cfgs[x]['frames'] = struct_metadata.get_frames_per_slice()
            self.cfgs[x]['shape'] = struct_metadata.get_pixelsize()
            self.cfgs[x]['zoom'] = struct_metadata.get_zoom()


    def align_structural_data(self):
        # example usage:
        # iter through all struc files
        for n in self.struct_nums:
            struc_fnames = self.fnames[n]
            print("Processing structural dataset %s" % n)
            # remove fnames that have been done before
            print("Running motion correction across %s files "%len(struc_fnames))
            st_slice = slice(self.cfgs[n]['red_ind'],None,self.cfgs[n]['nchan'])
            rets = []
            if len(struc_fnames) > 0:
                args = [(fn,  ind,
                          self.save_dir, self.max_shifts, self.strides, self.overlaps, 
                          self.max_deviation_rigid, self.shifts_opencv, self.border_nan, st_slice) for ind,fn in struc_fnames]

                pool = multiprocessing.Pool(min(self.num_proc,len(struc_fnames)))
                rets = pool.starmap(motion_correct_file,args)

            out_tiffname = os.path.join(self.base_folder,"%s_struc_%s.tif"%(self.prj,n))

            denoise_ds = np.zeros((self.cfgs[n]['nz'],self.cfgs[n]['shape'][0],self.cfgs[n]['shape'][1]),dtype=np.float32)
            aligned_ds = np.zeros((self.cfgs[n]['nz'],self.cfgs[n]['shape'][0],self.cfgs[n]['shape'][1]),dtype=np.float32)

            # go through all the mmap files
            for (ind, template) in rets:
                denoise_ds[ind,:,:] = template

            # save the denoised dataset
            #print("saving denoised data to %s"%out_tiffname)
            #tf.imsave(out_tiffname,denoise_ds)

            #align the denoised items to eachother
            aligned_ds[0,:,:] = denoise_ds[0,:,:]
            for i in range(1,len(aligned_ds)):
                s,_,_ = skimage.feature.register_translation(aligned_ds[i-1,:,:],denoise_ds[i,:,:],100)
                print(s)
                oi = scipy.ndimage.fourier_shift(np.fft.fftn(denoise_ds[i,:,:]),s)
                aligned_ds[i,:,:] = np.fft.ifftn(oi)

            # save the aligned dataset
            print("saving aligned data to %s"%out_tiffname)
            tf.imsave(out_tiffname,aligned_ds)

        def align_functional_data(self):
            # now go through functional files and take the structural data
            # first go through and save a file for each plane and each channel

            #%%  TEST CODE: start the cluster (if a cluster already exists terminate it)
            #if self.dview is not None:
            #    caiman.stop_server(dview=self.dview)
            #    self.dview = None
            #c, self.dview, n_processes = caiman.cluster.setup_cluster(
            #    backend='local', n_processes=None, single_thread=False)

            # for each functional imaging session
            for n in self.func_nums:
                # for each set of functional files
                # keep them around 100 frames per set
                n_files_per_set = round(100 / self.cfgs[n]['frames'])
                fname_sets = [ self.fnames[n][i:i+n_files_per_set] 
                               for i in range(0,len(self.fnames[n]),n_files_per_set)]
                for inds_fnames in fname_sets:
                    #unzip inds and fnames
                    inds, fnames = list(zip(*inds_fnames))
                    # go through each plane and motion correct the red channel
                    for p in range(self.cfgs[n]['nz']):
                        sliceRed = slice(self.cfgs[n]['nchan']*p+self.cfgs[n]['red_ind'],None,self.cfgs[n]['nchan']*self.cfgs[n]['nz'])
                        sliceGreen = slice(self.cfgs[n]['nchan']*p+self.cfgs[n]['green_ind'],None,self.cfgs[n]['nchan']*self.cfgs[n]['nz'])

                        # motion correct on the red channel
                        mc = MotionCorrect(fnames, dview=None,max_shifts=self.max_shifts, strides=self.strides, overlaps=self.overlaps, 
                          max_deviation_rigid=self.max_deviation_rigid, shifts_opencv=self.shifts_opencv, nonneg_movie=True, border_nan=self.border_nan,subidx=sliceRed)
                        ret = mc.motion_correct(save_movie=False)

                        # save the mean image of the red channel after correction
                        out_tiffname = os.path.join(self.base_folder,"%s_func_%s_p%s_%s_%s.tif"%(self.prj,n,p,min(inds),max(inds)))
                        tf.imsave(out_tiffname, np.mean(mc.templates_rig,axis=0))

                        # apply shifts to the green channel
                        mmgreen = ret.apply_shifts_movie(fnames,sliceGreen)

                        # save the green channel movie to the zarr
                        out_tiffname = os.path.join(self.base_folder,"%s_func_mov_%s_p%s_%s_%s.tif"%(self.prj,n,p,min(inds),max(inds)))

                        mmgreen.save(out_tiffname)

        def align_func_to_struc(self):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run alignment and registration of volumes using python")
    parser.add_argument('config_file', help='path to config file for this task')
    parser.add_argument('--struct', action='store_true', help='align structural data')
    parser.add_argument('--func', action='store_true', help='align functional data (requires struct data to have been aligned)')
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        create_config(args.config_file)
        sys.exit(0)

    print("Creating AlignmentHelper object with %s"%args.config_file)
    align_obj = AlignmentHelper(args.config_file)

    if args.struct:
        align_obj.align_structural_data()
    if args.func:
        align_obj.align_functional_data()
