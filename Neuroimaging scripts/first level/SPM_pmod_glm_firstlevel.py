#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First level analysis GLM using SPM.
@authors: nachshon - base script
chelsea - added DataGrabber instead of SelectFiles to allow for different task names
added parametric modulators to the GLM: add a 'pmod' field into the Bunch for model specification (at the node of runinfo) and add it into the contrast specification. Based off Ruonan's script https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/spm_pmod_glm_firstlevel.py
"""

#%%
# General python libraries

import os
import pandas as pd
import numpy as np

# nipype libraries

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model specification
from nipype import Node, Workflow, MapNode

from nipype.interfaces import fsl
from nipype.interfaces import spm
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.utility import Function

# SPM and FSL initiation

MatlabCommand.set_default_paths('/gpfs/gibbs/pi/levy_ifat/shared/MATLAB/spm12/') # set SPM12 path in the shared folder on the HPC 
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

#%%
# Adjust locations
data_dir =   '/gpfs/gibbs/pi/levy_ifat/Chelsea/RAID/'
output_dir =  '/home/cyx3/scratch60/RAID/output/imaging/Sink/results/' 
work_dir = '/home/cyx3/scratch60/work'

# subject list and task list

subject_list = [11,12,13,15,16,17,19,20,21,22,24,25,27,28,29,30,31,32,36,39,40,41,42,43,45,46,47,48,50,51,55,56,57,61,62] # Map field names to individual subject runs. 
task_list = ['gain1', 'gain2', 'loss1', 'loss2']
ses_list = [1,2]

# basic experiment properties

fwhm = 6 # full width at half maximum a.k.a smoothing in mm3
tr = 1 # Length of TR in seconds
removeTR = 10 # how many TRs should be removed from the beginning of the scan
highpass = 128. # high pass filter should be a float
motion_params = 6 # number of motion parameters to include in the GLM should be 0, 6 or 25
fd = 1 # Do you want to enter FD to the GLM? 1 yes 0 no 
dvars = 1 # Do you want std_dvars in the model? 1 yes 0 no 
n_comp_corr = 6 # how many comp corr do you want in your model? valid values are 0 and 6
n_procs = 4 # number of parallel process

## Building contrasts
# set contrasts, depend on the condition
# the condition names (cond_names) should correspond to the names of the event file.
# contrasts should be ('name of contrast - string', 'T (can also be F)', [a list of condition names], [a list of integers]) 
cond_names = ['Gain_amb', 'Gain_ambxGain_amb_sv^1', 'Gain_risk', 'Gain_riskxGain_risk_sv^1',
              'Loss_amb', 'Loss_ambxLoss_amb_sv^1', 'Loss_risk', 'Loss_riskxLoss_risk_sv^1']

#general activation
cont1 = ('Gain_Amb', 'T', cond_names, [1, 0, 0, 0, 0, 0, 0, 0])
cont2 = ('Gain_Risk', 'T', cond_names, [0, 0, 1, 0, 0, 0, 0, 0])
cont3 = ('Gain_Amb > Gain_Risk', 'T', cond_names, [1, 0, -1, 0, 0, 0, 0, 0]) 

cont4 = ('Loss_Amb', 'T', cond_names, [0, 0, 0, 0, 1, 0, 0, 0])
cont5 = ('Loss_Risk', 'T', cond_names, [0, 0, 0, 0, 0, 0, 1, 0])
cont6 = ('Loss_Amb > Loss_Risk', 'T', cond_names, [0, 0, 0, 0, 1, 0, -1, 0])

cont7 = ('Gain>Loss_Amb', 'T', cond_names, [1, 0, 0, 0, -1, 0, 0, 0])
cont8 = ('Gain>Loss_Risk', 'T', cond_names, [0, 0, 1, 0, 0, 0, -1, 0])

cont9 = ('Gain>Loss', 'T', cond_names, [1, 0, 1, 0, -1, 0, -1, 0])
cont10 = ('Amb>Risk', 'T', cond_names, [1, 0, -1, 0, 1, 0, -1, 0])

#sv
cont11 = ['Gain_Amb_SV', 'T', cond_names, [0, 1, 0, 0, 0, 0, 0, 0]]
cont12 = ['Gain_Risk_SV', 'T', cond_names, [0, 0, 0, 1, 0, 0, 0, 0]]
cont13 = ['Loss_Amb_SV', 'T', cond_names, [0, 0, 0, 0, 0, 1, 0, 0]]
cont14 = ['Loss_Risk_SV', 'T', cond_names, [0, 0, 0, 0, 0, 0, 0, 1]]

cont15 = ('All', 'T', cond_names, [1, 1, 1, 1, 1, 1, 1, 1])

cont16 = ['Gain_SV', 'T', cond_names, [0, 1, 0, 1, 0, 0, 0, 0]]
cont17 = ['Loss_SV', 'T', cond_names, [0, 0, 0, 0, 0, 1, 0, 1]]
contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, cont14, cont15, cont16, cont17]

#%% regressors for design matrix

def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None, removeTR = 10,
                     decimals=3, amplitude=1.0):
    
    from pathlib import Path
    from nipype.interfaces.base.support import Bunch
    import pandas as pd
    import numpy as np

    # Process the events file
    events = pd.read_csv(events_file, sep = r'\s+') #read events files csv
    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    np.savetxt(out_motion, regress_data[motion_columns].values[removeTR:,], '%g')
    
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    domain = list(set(events.condition.values))[0] #loss or gain
    trial_types = list(set(events.trial_type.values)) #risk or amb
    
    # add parametric modulator, bunch field name = 'pmod'
    bunch_fields += ['pmod']
    
    runinfo = Bunch(
        scans=in_file,
        conditions=[domain + '_' + trial_type for trial_type in trial_types], #conditions = ['Gain_amb', 'Gain_risk', 'Loss_amb', 'Loss_risk']
        **{k: [] for k in bunch_fields})
    
    for condition in runinfo.conditions:        
        
        event = events[events.trial_type.str.match(condition[5:])] #risk or amb
        runinfo.onsets.append(np.round(event.onset.values-removeTR,3).tolist()) # added -removeTR to align to the onsets after removing X number of TRs from the scan
        runinfo.durations.append(np.round(event.duration.values, 3).tolist()) #6 seconds
        
        # parametric modulator, name it as the condition name + '_sv'
        runinfo.pmod.append(Bunch(
            name = [condition + '_sv'], #name of modulator for each condition
            param = [np.round(event.zsvs.values,3).tolist()], # values of modulator for each condition
            poly = [1] #degree of modulation, 1-linear
        ))
        
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))

#     # response predictor regardless of condition
#     runinfo.conditions.append('Resp')
            
# #     response predictor when there is a button press
#     resp_mask = events.resp != 2    
#     resp_onset = np.round(events.resp_onset.values[resp_mask] - removeTR, 3).tolist()
#     runinfo.onsets.append(resp_onset)
#     resp_duration = np.round(events.resp_duration.values[resp_mask], 3).tolist()
#     runinfo.durations.append(resp_duration)
# #    runinfo.durations.append([0] * len(resp_onset))
#     runinfo.amplitudes.append([amplitude] * len(resp_onset))

    # runinfo.pmod.append(None)
    
    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[removeTR:,].T.tolist() # adding removeTR to cut the first rows
        
    return runinfo, str(out_motion)

#%% infosource
infosource = pe.Node(util.IdentityInterface(fields=['subject_id', 'ses_id']),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list),
                       ('ses_id',  ses_list)]

# Flexibly collect data from disk to feed into flows.
# selectfiles = pe.Node(nio.SelectFiles(templates,
#                       base_directory=data_dir),
#                       name="selectfiles")
# #selectfiles.iterables = [('task_id', task_ids)]       
# selectfiles.inputs.task_id = task_ids

#%# datasource with datagrabber
datasource = pe.Node(nio.DataGrabber(infields=['subject_id', 'task_id', 'ses_id'], outfields=['func', 'mask', 'regressors', 'events']),
                     name="datasource")
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(func = 'R_A_ID_BIDS/derivatives/sub-%s/ses-%s/func/sub-%s_ses-%s_task-%s_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
                                        mask = 'R_A_ID_BIDS/derivatives/sub-%s/ses-%s/func/sub-%s_ses-%s_task-%s_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz',
                                        regressors = 'R_A_ID_BIDS/derivatives/sub-%s/ses-%s/func/sub-%s_ses-%s_task-%s_desc-confounds_timeseries.tsv',
                                        events = 'event_files/sub-%s_ses-%s_task-%s_cond.csv')
datasource.inputs.template_args = dict(func=[['subject_id', 'ses_id', 'subject_id', 'ses_id', 'task_id']],
                                       mask=[['subject_id', 'ses_id', 'subject_id', 'ses_id', 'task_id']],
                                       regressors=[['subject_id', 'ses_id', 'subject_id', 'ses_id', 'task_id']],
                                       events=[['subject_id', 'ses_id', 'task_id']])
datasource.inputs.subject_id = subject_list
datasource.inputs.task_id = task_list
datasource.inputs.ses_id = ses_list
datasource.inputs.sort_filelist = False

#%# runinfo
runinfo = MapNode(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'removeTR', 'motion_columns'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo',
    iterfield = ['in_file', 'events_file', 'regressors_file'])
regressors = ['std_dvars'*dvars, 'framewise_displacement'*fd] + ['a_comp_cor_%02d' % i for i in range(n_comp_corr)]
while('' in regressors) :
    regressors.remove('')
runinfo.inputs.regressors_names = regressors
runinfo.inputs.removeTR = removeTR                                  
 
motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'] + \
         ['trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1', 'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'] + \
         ['trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2', 'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2'] + \
         ['trans_x_power2', 'trans_y_power2', 'trans_z_power2', 'rot_x_power2', 'rot_y_power2', 'rot_z_power2']

runinfo.inputs.motion_columns   = motion[:motion_params]

#%# extract
extract = pe.MapNode(fsl.ExtractROI(), name="extract", iterfield = ['in_file'])
extract.inputs.t_min = removeTR
extract.inputs.t_size = -1
extract.inputs.output_type='NIFTI' 

#%# smooth
smooth = Node(spm.Smooth(), name="smooth", fwhm = fwhm)

#%# modelspec
modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec") 
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'scans' # supposedly it means tr
modelspec.inputs.output_units = 'scans'
modelspec.inputs.time_repetition = 1.  # make sure its with a dot 
modelspec.inputs.high_pass_filter_cutoff = highpass

#%# level 1 design
level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = 1.
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

#%# level 1 estimate
level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical': 1}

#%% contrasts
contrastestimate = pe.Node(
    interface=spm.EstimateContrast(), name="contrastestimate")
contrastestimate.overwrite = True
contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}
contrastestimate.inputs.contrasts = contrasts      
         
#%% Connect workflow
wfSPM = Workflow(name="l1spm_pmod_bothsessions", base_dir=output_dir)
wfSPM.connect([
        (infosource,     datasource,      [('subject_id',     'subject_id'),
                                          ('ses_id', 'ses_id')]),
        (datasource,    runinfo,          [('events',         'events_file'),
                                            ('regressors',     'regressors_file')]),
        (datasource,    extract,          [('func',           'in_file')]),
        (extract,        smooth,           [('roi_file',       'in_files')]),
        (smooth,         runinfo,          [('smoothed_files', 'in_file')]),
        (smooth,         modelspec,        [('smoothed_files', 'functional_runs')]),   
        (runinfo,        modelspec,        [('info',           'subject_info'), 
                                            ('realign_file',   'realignment_parameters')]),
        (modelspec,      level1design,     [('session_info',   'session_info')]),
        (level1design,   level1estimate,   [('spm_mat_file',   'spm_mat_file')]),
        (level1estimate, contrastestimate, [('spm_mat_file',   'spm_mat_file'), 
                                            ('beta_images',    'beta_images'),
                                            ('residual_image', 'residual_image')]),
        ])

#%% Run workflow
wfSPM.run('MultiProc', plugin_args={'n_procs': n_procs})                                
