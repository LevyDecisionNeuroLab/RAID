"""
Reference: https://github.com/poldracklab/ds003-post-fMRIPrep-analysis/blob/master/workflows.py
"""

import nipype.interfaces.io as nio  # Data i/o
from nipype.interfaces import spm
from nipype import Node, Workflow, MapNode
import nipype.interfaces.utility as util # utility
from nipype import SelectFiles
import os

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
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006', 'con_0007', 'con_0008',
                'con_0009', 'con_0010', 'con_0011', 'con_0012', 'con_0013', 'con_0014', 'con_0015', 'con_0016', 'con_0017']

subject_list = [11,12,13,15,16,17,19,20,21,22,24,25,27,28,29,30,31,32,36,39,40,41,42,43,45,46,47,48,50,51,55,56,57,61,62]

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
infosource = Node(util.IdentityInterface(fields=['contrast_id', 'subject_id']),
                  name="infosource")

infosource.iterables = [('contrast_id', contrast_list)]
infosource.inputs.subject_id = subject_list

# SelectFiles - to grab the data (alternative to DataGrabber)
templates = {'cons': os.path.join('/home/cyx3/scratch60/RAID/output/imaging/Sink/1stLevel_objv/_subject_id_{subject_id}/contrastestimate/', 
                         '{contrast_id}.nii')}

selectfiles = MapNode(SelectFiles(templates,
                    base_directory='/home/cyx3/scratch60/RAID/work/',
                                  sort_filelist=True),
                   name="selectfiles", 
                   iterfield = ['subject_id'])

datasink = Node(nio.DataSink(base_directory='/home/cyx3/scratch60/RAID/output/imaging/Sink/'),
                name="datasink")


l2analysis = Workflow(name='l2spm_objv')

l2analysis.base_dir = '/home/cyx3/scratch60/RAID/work/'

l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                               ('subject_id', 'subject_id')]),

                    (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                    
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
                        '2ndLevel_objv.@spm_mat'),
                       ('spmT_images',
                        '2ndLevel_objv.@T'),
                       ('con_images',
                        '2ndLevel_objv.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel_objv.@threshold')]),
                                                        ])
#%%                                                     
l2analysis.run('MultiProc', plugin_args={'n_procs': 15})
