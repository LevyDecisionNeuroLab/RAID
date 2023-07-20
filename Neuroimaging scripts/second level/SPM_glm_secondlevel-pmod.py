"""
Reference: https://github.com/poldracklab/ds003-post-fMRIPrep-analysis/blob/master/workflows.py
"""

import nipype.interfaces.io as nio  # Data i/o
from nipype.interfaces import spm
import nipype.pipeline.engine as pe  # pypeline engine
from nipype import Node, Workflow, MapNode
import nipype.interfaces.utility as util # utility
from nipype import SelectFiles
import os
from pathlib import Path

# General python libraries

import os
import pandas as pd
import numpy as np

# nipype libraries

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model specification

from nipype.interfaces import fsl
from nipype.interfaces import spm
from nipype.interfaces.utility import Function

from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/gpfs/gibbs/pi/levy_ifat/shared/MATLAB/spm12/')

#%% Group analysis - based on SPM
# OneSampleTTestDesign - creates one sample T-Test Design
onesamplettestdes = Node(spm.OneSampleTTestDesign(),
                         name="onesampttestdes")

# EstimateModel - estimates the model
level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                         name="level2conestimate")
cont1 = ['Group', 'T', ['mean'], [1]]
level2conestimate.inputs.contrasts = [cont1]

# Which contrasts to use for the 2nd-level analysis
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006', 'con_0007', 'con_0008', 'con_0009', 'con_0010', 'con_0011', 'con_0012', 'con_0013', 'con_0014', 'con_0015', 'con_0016', 'con_0017']

#subject_list = [11,12,13,15,16,17,19,20,21,22,24,25,27,28,29,30,31,32,36,39,40,41,42,43,45,46,47,48,50,51,55,56,57,61,62]
#ses_list = [1,2]

# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=False,
                              use_fwe_correction=True, # here we can use fwe or fdr
                              extent_threshold=10, #cluster extent
                              height_threshold= 0.005, #cluster-forming threshold
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                              name="level2thresh")
# First, an arbitrary voxel-level primary threshold defines clusters by retaining groups of suprathreshold voxels. Second, a cluster-level extent threshold, measured in units of contiguous voxels (k), is determined based on the estimated distribution of cluster sizes under the null hypothesis of no activation in any voxel in that cluster. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4214144/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5153601/

#Infosource - a function free node to iterate over the list of subject names
# infosource = Node(util.IdentityInterface(fields=['contrast_id', 'subject_id']),
#                   name="infosource")

# infosource.iterables = [('contrast_id', contrast_list), ('subject_id', subject_list)]
# infosource.inputs.subject_id = subject_list

infosource = pe.Node(util.IdentityInterface(fields=['contrast_id']),
                  name="infosource")

infosource.iterables = [('contrast_id', contrast_list)]
#infosource.inputs.subject_id = subject_list
#infosource.inputs.ses_id = ses_list

# SelectFiles - to grab the data (alternative to DataGrabber)
# templates = {'cons': os.path.join('/home/cyx3/scratch60/RAID/output/imaging/Sink/results/l1spm_pmod_bothsessions/_ses_id_*_subject_id_{subject_id}/contrastestimate/', 
#                          '{contrast_id}.nii')}

# selectfiles = MapNode(SelectFiles(templates,
#                     base_directory='/home/cyx3/scratch60/RAID/work/',
#                                   sort_filelist=True),
#                    name="selectfiles", 
#                    iterfield = ['subject_id'])

datasource = pe.Node(nio.DataGrabber(infields=['contrast_id'], outfields=['outfiles']),
                     name="datasource")
datasource.inputs.base_directory = '/gpfs/ysm/scratch60/levy_ifat/cyx3/'
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(outfiles = 'RAID/output/imaging/Sink/results/l1spm_pmod_bothsessions/_ses_id_*_subject_id_*/contrastestimate/%s.nii')
datasource.inputs.template_args = dict(outfiles=[['contrast_id']])
#datasource.inputs.ses_id = ses_list
#datasource.inputs.subject_id = subject_list
datasource.inputs.contrast_id = contrast_list
datasource.inputs.sort_filelist = False

datasink = Node(nio.DataSink(base_directory='/home/cyx3/scratch60/RAID/output/imaging/Sink/'),
                name="datasink")

l2analysis = Workflow(name='l2spm_pmod_bothsessions')

l2analysis.base_dir = '/home/cyx3/scratch60/RAID/work/'

l2analysis.connect([(infosource, datasource, [#('subject_id', 'subject_id'),
                                             # ('ses_id', 'ses_id'),
                                             ('contrast_id', 'contrast_id')]),

                    (datasource, onesamplettestdes, [('outfiles', 'in_files')]),
                    
                    (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                          'spm_mat_file')]),
                    (level2estimate, level2conestimate, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
                    (level2conestimate, datasink, [('spm_mat_file',
                        '2ndLevel_pmod_bothsessions.@spm_mat'),
                       ('spmT_images',
                        '2ndLevel_pmod_bothsessions.@T'),
                       ('con_images',
                        '2ndLevel_pmod_bothsessions.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel_pmod_bothsessions.@threshold')]),
                                                        ])
#%%                                                     
l2analysis.run('MultiProc', plugin_args={'n_procs': 4})
