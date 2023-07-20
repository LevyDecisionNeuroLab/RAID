%% This script calculate choice proportion by breaking down trial types by outcome levels
clear all
close all

fitparwave = '09300219';

root = 'E:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis';
data_path = fullfile(root, 'PTB Behavior Log/'); % Original log from PTB
subjects = getSubjectsInDir(data_path, 'subj'); %function
exclude = [2581 2587]; % TEMPORARY: subjects incomplete data (that the script is not ready for)
subjects = subjects(~ismember(subjects, exclude));
% subjects = [2654 2655 2656 2657 2658 2659 2660 2661 2662 2663 2664 2665 2666]

path = fullfile(root, 'Behavior fitpar files', ['Behavior data fitpar_' fitparwave], filesep);
cd(path);

% defining unique values
valueLevel = [5 8 12 25];
riskLevel = [0.25 0.5 0.75];
ambigLevel = [0.24 0.5 0.74];

% results file
output_file1 = ['chocie_prob_byoutcome.txt'];
fid1 = fopen([output_file1],'w')

fprintf(fid1,'id\tr5_mon\tr8_mon\tr12_mon\tr25_mon\ta5_mon\ta8_mon\ta12_mon\ta25_mon\tr5_med\tr8_med\tr12_med\tr25_med\ta5_med\ta8_med\ta12_med\ta25_med\n')


domain = {'MON','MED'};

rng('shuffle')

% Fill in subject numbers separated by commas
% subjects = {'87','88'};
for s = 1:length(subjects)
    
    subject = subjects(s); 
    
    % load monetary file for subject and extract params & choice data
    %ChoicesP stands for monetary, ChoicesN stands for medical
    % monetary
    load(['MDM_MON_' num2str(subject) '_fitpar.mat']);
    choiceMatrixP = Datamon.choiceMatrix;

    riskyChoicesP = choiceMatrixP.riskProb;
    ambigChoicesP = choiceMatrixP.ambigProb;
    riskyChoicesPC = choiceMatrixP.riskCount;
    ambigChoicesPC = choiceMatrixP.ambigCount;
    
    % calculate choice prob by outcome levels
    risk_sumP = riskyChoicesP .* riskyChoicesPC;
    risk_mon = sum(risk_sumP, 1) ./ sum(riskyChoicesPC, 1);
    ambig_sumP = ambigChoicesP .* ambigChoicesPC;
    ambig_mon = sum(ambig_sumP, 1) ./ sum(ambigChoicesPC,1);
    
    % medical
    load(['MDM_MED_' num2str(subject) '_fitpar.mat']);
    choiceMatrixN = Datamed.choiceMatrix;

    riskyChoicesN = choiceMatrixN.riskProb;
    ambigChoicesN = choiceMatrixN.ambigProb;
    riskyChoicesNC = choiceMatrixN.riskCount;
    ambigChoicesNC = choiceMatrixN.ambigCount;
    
    % calculate choice prob by outcome levels
    risk_sumN = riskyChoicesN .* riskyChoicesNC;
    risk_med = sum(risk_sumN, 1) ./ sum(riskyChoicesNC, 1);
    ambig_sumN = ambigChoicesN .* ambigChoicesNC;
    ambig_med = sum(ambig_sumN, 1) ./ sum(ambigChoicesNC,1);

    %% calculate choice prob by outcome levels
    fprintf(fid1,'%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',...
                num2str(subject),risk_mon(1),risk_mon(2),risk_mon(3),risk_mon(4),...
                ambig_mon(1),ambig_mon(2),ambig_mon(3),ambig_mon(4),...
                risk_med(1),risk_med(2),risk_med(3),risk_med(4),...
                ambig_med(1),ambig_med(2),ambig_med(3),ambig_med(4))
         
end

fclose(fid1)