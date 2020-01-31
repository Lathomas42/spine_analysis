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
    "max_deviation_rigid": 3 ,  # maximum deviation allowed for patch with respect to rigid shifts
    "shifts_opencv": True,  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    "border_nan": 'copy',  # replicate values along the boundary (if True, fill in with NaN)
    # first do the structural data in the _2 folder
    "server_dir": "/global/scratch/logan_thomas/",
    "output_dir": "/global/scratch/logan_thomas/output/",
    "prj": "20191209",
    "struct_nums":[2], #list of all _# for structural data
    "func_nums":[1], #list of all _# for functional data
    "num_proc": 4,
    "yaw_range": [-5,5,1], # min max step
    "tilt_range": [-5,5,1], # min_max_step
    "debug": False,
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
        cv = self._get_var('SI.hChannels.channelSave')
        if '[' in cv:
            for e,ci in enumerate(self._get_var('SI.hChannels.channelSave')[1:-1].split(';')):
                ind = int(ci)-1
                if colors[ind][1:-1] == channel:
                    return ind
        else:
            ind = int(cv) - 1
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

def motion_correct_file( args, n_iter=0,max_iter=10):
    try:
        in_fn, file_index, template, mc_args, \
            subidx, apply_subidx, apply_ofn = args
        max_shifts, strides, overlaps, \
            max_deviation_rigid, shifts_opencv, border_nan = mc_args
        mc = MotionCorrect(in_fn, dview=None, max_shifts=max_shifts,
                      strides=strides, overlaps=overlaps,
                      max_deviation_rigid=max_deviation_rigid,
                      shifts_opencv=shifts_opencv, nonneg_movie=True,
                      splits_els=1,splits_rig=1,
                      border_nan=border_nan, subidx=subidx)
        mc.motion_correct(template=template,save_movie=False)
        if apply_subidx is not None:
            applied_mov = mc.apply_shifts_movie(in_fn,apply_subidx)
            applied_mov.save(apply_ofn)
        return (file_index, np.mean(mc.templates_rig,axis=0))
    except Exception as e:
        if n_iter < max_iter:
            print("Retrying %s due to %s"%(in_fn, e))
            return motion_correct_file(args, n_iter=n_iter+1)
        else:
            return Exception("Failed max_iter: %s times"%max_iter)


class AlignmentHelper(object):
    def __init__(self,cfg=default_config):
        if isinstance(cfg, str):
            cfg = json.load(open(cfg,'r'))

        self.dview = None
        self.mc_args = [ cfg["max_shifts"], cfg["strides"], cfg["overlaps"],
                        cfg["max_deviation_rigid"],  cfg["shifts_opencv"], cfg["border_nan"]]
        self.server_dir = cfg["server_dir"]
        self.prj = cfg["prj"]
        self.struct_nums = cfg["struct_nums"]
        self.func_nums = cfg["func_nums"]
        self.num_proc=cfg["num_proc"]
        self.prj_dir = os.path.join(self.server_dir, self.prj)
        self.debug = False
        self.base_folder = os.path.join(cfg['output_dir'],self.prj)

        if not os.path.isdir(self.base_folder):
            os.makedirs(self.base_folder)

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
            inds = [ int(s[len(com_pre):-len(com_suff)]) - 1 for s in fns ]
            self.fnames[x] = list(zip(inds,fns))
            struct_metadata = MetadataParser(self.fnames[x][0][1])
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
                args = [(fn,  ind, None, self.mc_args, st_slice, None,None) for ind,fn in struc_fnames]

                pool = multiprocessing.Pool(min(self.num_proc,len(struc_fnames)))
                rets = pool.imap_unordered(motion_correct_file,args)


            denoise_ds = np.zeros((self.cfgs[n]['nz'],self.cfgs[n]['shape'][0],self.cfgs[n]['shape'][1]),dtype=np.float32)
            aligned_ds = np.zeros((self.cfgs[n]['nz'],self.cfgs[n]['shape'][0],self.cfgs[n]['shape'][1]),dtype=np.float32)

            # go through all the mmap files
            for (ind, template) in rets:
                denoise_ds[ind,:,:] = template

            #align the denoised items to eachother
            aligned_ds[0,:,:] = denoise_ds[0,:,:]
            for i in range(1,len(aligned_ds)):
                s,_,_ = skimage.feature.register_translation(aligned_ds[i-1,:,:],denoise_ds[i,:,:],100)
                print(s)
                oi = scipy.ndimage.fourier_shift(np.fft.fftn(denoise_ds[i,:,:]),s)
                aligned_ds[i,:,:] = np.fft.ifftn(oi)

            out_tiffname = os.path.join(self.base_folder,"%s_struc_%s.tif"%(self.prj,n))

            print("saving aligned data to %s"%out_tiffname)
            tf.imsave(out_tiffname,aligned_ds)

            out_jsname = os.path.join(self.base_folder,"%s_struc_%s.json"%(self.prj,n))
            with open(out_jsname,'w') as out_file:
                json.dump(self.cfgs[n],out_file)

    def align_functional_data(self):
        # now go through functional files and take the structural data
        # first go through and save a file for each plane and each channel
        # for each functional imaging session
        for n in self.func_nums:
            out_dir = os.path.join(self.base_folder,"func",str(n))
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            # for each set of functional files
            # keep them around 100 frames per set
            n_files_per_set = round(400 / self.cfgs[n]['frames'])
            fname_sets = [ self.fnames[n][i:i+n_files_per_set]
                            for i in range(0,len(self.fnames[n]),n_files_per_set)]

            out_jsname = os.path.join(out_dir,"%s_func_%s.json"%(self.prj,n))
            with open(out_jsname,'w') as out_file:
                json.dump(self.cfgs[n],out_file)

            for p in range(self.cfgs[n]['nz']):
                print("Processing functional data on plane %s" %p)

                sliceRed = slice(self.cfgs[n]['nchan']*p+self.cfgs[n]['red_ind'],None,self.cfgs[n]['nchan']*self.cfgs[n]['nz'])
                sliceGreen = slice(self.cfgs[n]['nchan']*p+self.cfgs[n]['green_ind'],None,self.cfgs[n]['nchan']*self.cfgs[n]['nz'])

                args = []
                template = None
                for inds_fnames in fname_sets:
                    inds, fnames = list(zip(*inds_fnames))
                    green_ofn = os.path.join(out_dir,"%s_func_mov_%s_p%s_%.4d_%.4d.tif"%(self.prj,n,p,min(inds),max(inds)))
                    arg = (list(fnames),(p,list(inds)), template, self.mc_args, sliceRed, sliceGreen, green_ofn)
                    if template is None:
                        # first iteration run motion correction to get a template
                        print("Processing first set of functional files to get a template")
                        inds,template = motion_correct_file(arg)
                        red_ofn = os.path.join(out_dir,"%s_func_red_%s_p%s_%.4d_%.4d.tif"%(self.prj,n,inds[0],min(inds[1]),max(inds[1])))
                        tf.imsave(red_ofn, template)
                    else:
                        args.append(arg)

                # now process the rest
                pool = multiprocessing.Pool(min(self.num_proc,len(args)))
                rets = pool.imap_unordered(motion_correct_file,args)

                for (inds, template) in rets:
                    red_ofn = os.path.join(out_dir,"%s_func_red_%s_p%s_%.4d_%.4d.tif"%(self.prj,n,inds[0],min(inds[1]),max(inds[1])))
                    tf.imsave(red_ofn, template)

    def align_func_to_struc(self):
        locs = dict()
        for sn in self.struct_nums:
            in_tiffname = os.path.join(self.base_folder,"%s_struc_%s.tif"%(self.prj,sn))
            if not os.path.exists(in_tiffname):
                return Exception("%s is not found, please run struct align first" %in_tiffname)

            in_data = tf.TiffFile(in_tiffname).asarray()
            d_min = in_data.min(axis=(1,2),keepdims=True)
            # hmm do some thresholding?
            pcs = np.percentile(in_data,99.95,axis=(1,2),keepdims=True)
            in_data = (in_data - d_min) / (pcs - d_min)
            in_data[in_data > 1.0] = 1.0
            in_data = (in_data * 255.0 ).astype(np.uint8)

            # now go through functional sets
            if sn not in locs:
                locs[sn] = dict()
            for n in self.func_nums:
                if n not in locs[sn]:
                    locs[sn][n] = dict()
                out_dir = os.path.join(self.base_folder,"func",str(n))
                mag = self.cfgs[sn]['zoom'] / self.cfgs[n]['zoom']
                for p in range(self.cfgs[n]['nz']):
                    fred_fns = glob.glob(os.path.join(out_dir,"%s_func_red_%s_p%s_*.tif"%(self.prj,n,p)))
                    for fred in fred_fns:
                        plane_data = tf.TiffFile(fred).asarray()
                        # normalize
                        pc = np.percentile(plane_data,99.95)
                        plane_data = (plane_data- plane_data.min()) / (pc - plane_data.min())
                        plane_data[plane_data > 1.0] = 1.0
                        zs = plane_data.shape[0]
                        plane_data = skimage.transform.resize(plane_data,np.round(np.asarray(plane_data.shape)*mag).astype(int))
                        plane_data[plane_data > 1.0] = 1.0
                        # -> uint8
                        plane_data = (plane_data*255.0).astype(np.uint8)
                        min_vals = []
                        min_locs = []
                        for z in range(in_data.shape[0]):
                            ret=cv2.matchTemplate(plane_data, in_data[z,:,:], cv2.TM_SQDIFF_NORMED)
                            m,_,ml,_ = cv2.minMaxLoc(ret)
                            min_vals.append(m)
                            min_locs.append(ml)
                        z = np.argmin(min_vals)
                        min_loc = [int(x) for x in min_locs[z]]
                        # put z at the beginning of coordinates
                        min_loc.insert(0,int(z))
                        mv = [min_vals[z]]
                        # val,z,y,x
                        mv.extend(min_loc)
                        locs[sn][n][fred] = mv

                        # for now lets save some images
                        if self.debug:
                            sd = os.path.join(self.base_folder,'func',str(n),'debug')
                            if not os.path.isdir(sd):
                                os.makedirs(sd)
                            testname = '_'.join(fred.split('_')[-3:-1])
                            fout = testname+'_%s_f.tif'%sn
                            sout = testname+'_%s_s.tif'%sn
                            tf1out = os.path.join(sd,fout)
                            tf.imsave(tf1out,plane_data)
                            tf2out = os.path.join(sd,sout)
                            tf.imsave(tf2out,in_data[mv[1],:,:])

        js_out = os.path.join(self.base_folder,"func","func_locs.json")
        with open(js_out,'w') as outfile:
            json.dump(locs, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run alignment and registration of volumes using python")
    parser.add_argument('config_file', help='path to config file for this task')
    parser.add_argument('--struct', action='store_true', help='align structural data')
    parser.add_argument('--func', action='store_true', help='align functional data (requires struct data to have been aligned)')
    parser.add_argument('--register', action='store_true',help='register the functional planes to the structural data, requires the other two operations to be run first')
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
    if args.register:
        align_obj.align_func_to_struc()

# %%
