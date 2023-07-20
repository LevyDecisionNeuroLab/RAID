%% This script is meant to take parameters files and print some data to Excel
clearvars
close all


fitparwave = 'Behavior data fitpar_1130012017';
root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis';
data_path = fullfile(root, 'PTB Behavior Log/'); % Original log from PTB
subjects = getSubjectsInDir(data_path, 'subj'); %function
exclude = [2581]; % TEMPORARY: subjects incomplete data (that the script is not ready for)
subjects = subjects(~ismember(subjects, exclude));


cd (['D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\Behavior fitpar files\' fitparwave]);
path = ['D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\Behavior fitpar files\' fitparwave];


output_file1 = ['param_subjRate_uncstr_' fitparwave '.txt'];


% results file
fid1 = fopen([output_file1],'w')
fprintf(fid1,'subject\tmonetary\t\t\t\t\t\t\t\t\t\t\t\tmedical\n')
fprintf(fid1,['\talpha\talphase\tbeta\tbetase\tgamma\tgammase\tr2\tAIC\tBIC\tmodel\texitFlag\toptimizer',...
              '\talpha\talphase\tbeta\tbetase\tgamma\tgammase\tr2\tAIC\tBIC\tmodel\texitFlag\toptimizer\n'])

% Fill in subject numbers separated by commas
% subjects = {'87','88'};
for s = 1:length(subjects)
    
    subject = subjects(s); 
    % load gains file for subject and extract params & choice data
    load(['MDM_MON_' num2str(subject) '_fitpar.mat']);
    aP = Datamon.mfRUncstr.alpha;
    aPse = Datamon.mfRUncstr.MLE.se(3);
    bP = Datamon.mfRUncstr.beta;
    bPse = Datamon.mfRUncstr.MLE.se(2);
    gP = Datamon.mfRUncstr.gamma;
    gPse = Datamon.mfRUncstr.MLE.se(1);
    r2P = Datamon.mfRUncstr.r2;
    AICP = Datamon.mfRUncstr.MLE.AIC;
    BICP = Datamon.mfRUncstr.MLE.BIC;
    modelP = Datamon.mfRUncstr.MLE.model;
    exitFlagP = Datamon.mfRUncstr.MLE.exitflag;
    optimizerP = Datamon.mfRUncstr.MLE.optimizer;
    
    % load gains file for subject and extract params & choice data
    load(['MDM_MED_' num2str(subject) '_fitpar.mat']);
    aN = Datamed.mfRUncstr.alpha;
    aNse = Datamed.mfRUncstr.MLE.se(3);
    bN = Datamed.mfRUncstr.beta;
    bNse = Datamed.mfRUncstr.MLE.se(2);
    gN = Datamed.mfRUncstr.gamma;
    gNse = Datamed.mfRUncstr.MLE.se(1);
    r2N = Datamed.mfRUncstr.r2;
    AICN = Datamed.mfRUncstr.MLE.AIC;
    BICN = Datamed.mfRUncstr.MLE.BIC;
    modelN = Datamed.mfRUncstr.MLE.model;
    exitFlagN = Datamed.mfRUncstr.MLE.exitflag;
    optimizerN = Datamed.mfRUncstr.MLE.optimizer;


    %write into param text file
    fprintf(fid1,'%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%f\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%f\t%s\n',...
            num2str(subject),aP,aPse,bP,bPse,gP,gPse,r2P,AICP,BICP,modelP,exitFlagP,optimizerP,...
                             aN,aNse,bN,bNse,gN,gNse,r2N,AICN,BICN,modelN,exitFlagN,optimizerN)
        
end

fclose(fid1);
