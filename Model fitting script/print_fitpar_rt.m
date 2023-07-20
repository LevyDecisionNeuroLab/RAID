clearvars
close all

root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis';
data_path = fullfile(root, 'PTB Behavior Log/'); % Original log from PTB
subjects = getSubjectsInDir(data_path, 'subj'); %function
exclude = [2581]; % TEMPORARY: subjects incomplete data (that the script is not ready for)
subjects = subjects(~ismember(subjects, exclude));

fitparwave = 'Behavior data fitpar_10172017';
cd (['D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\Behavior fitpar files\' fitparwave]);
path = ['D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\Behavior fitpar files\' fitparwave];

% defining monetary values
value = [5 8 12 25];
risk = [0.25 0.5 0.75 0.5 0.5 0.5];
amb = [0 0 0 0.24 0.5 0.74];

trials = 21;
trial2include = [1:trials];

% Fill in subject numbers separated by commas
% subjects = {'87','88'};
for s = 1:length(subjects)  
    subject = subjects(s); 
    % load mon file for subject and extract params & choice data
    %ChoicesP stands for monetary, ChoicesN stands for medical
    load(['MDM_MON_' num2str(subject) '_fitpar.mat']);       
    valsP = Datamon.vals;
    probsP = Datamon.probs;
    ambigsP = Datamon.ambigs;
    rtP = Datamon.rt;
    
    valsP = valsP(trial2include);
    probsP = probsP(trial2include);
    ambigsP = ambigsP(trial2include);
    rtP = rtP(trial2include);
        
    % load med file for subject and extract params & choice data
    load(['MDM_MED_' num2str(subject) '_fitpar.mat']);
    valsN = Datamed.vals;
    probsN = Datamed.probs;
    ambigsN = Datamed.ambigs;
    rtN = Datamed.rt;
    
    valsN = valsN(trial2include);
    probsN = probsN(trial2include);
    ambigsN = ambigsN(trial2include);
    rtN = rtN(trial2include);

        
    %% for Excel file - rt by lottery type
    % monetary first, then medical
    % reaction time matrix by lottery types
    rtByLottP = zeros(length(risk), length(value));
    for i = 1:length(value)
        for j = 1:length(risk)
            rtByLottP(j,i) = nanmean(rtP(valsP == value(i) & probsP == risk(j) & ambigsP == amb(j)));
        end
    end
    
    rtByLottN = zeros(length(risk), length(value));
    for i = 1:length(value)
        for j = 1:length(risk)
            rtByLottN(j,i) = nanmean(rtN(valsN == value(i) & probsN == risk(j) & ambigsN == amb(j)));
        end
    end
    
    all_data_subject = [value; rtByLottP ;value; rtByLottN];    
    xlFile = ['rt_by_lotteries_' num2str(trials) 'trials.xls'];
    dlmwrite(xlFile, subject , '-append', 'roffset', 1, 'delimiter', ' ');  
    dlmwrite(xlFile, all_data_subject, 'coffset', 1, '-append', 'delimiter', '\t');
    
    %% for Excel file - rt by uncertainty level
    
    %exclude choices with $5 or slight improvement
    %P is monetary, N is medical

    % monetary
    rtByUncertP = zeros(1,length(risk)+3);
    for i = 1:length(risk)
        rtByUncertP(i) = nanmean(rtP(valsP ~= 5 & probsP == risk(i) & ambigsP == amb(i)));
    end
    rtByUncertP(length(risk)+1) = nanmean(rtP(valsP ~= 5 & ambigsP == 0)); % all risky trials
    rtByUncertP(length(risk)+2) = nanmean(rtP(valsP ~= 5 & ambigsP ~= 0)); % all ambiguous trials
    rtByUncertP(length(risk)+3) = nanmean(rtP(valsP ~= 5)); % all monetary trials
    
    %Medical
    rtByUncertN = zeros(1,length(risk)+3);
    for i = 1:length(risk)
        rtByUncertN(i) = nanmean(rtN(valsN ~= 5 & probsN == risk(i) & ambigsN == amb(i)));
    end
    rtByUncertN(length(risk)+1) = nanmean(rtN(valsN ~= 5 & ambigsN == 0)); % all risky trials
    rtByUncertN(length(risk)+2) = nanmean(rtN(valsN ~= 5 & ambigsN ~= 0)); % all ambiguous trials
    rtByUncertN(length(risk)+3) = nanmean(rtN(valsN ~= 5)); % all medical trials

   cp_title = {'subject ID', 'r25', 'r50','r75','a24','a50','a74','rAll','aAll','All',...
                             'r25', 'r50','r75','a24','a50','a74','rAll','aAll','All'};
   cp_data_subject = [subject,rtByUncertP,rtByUncertN];
                          
    xlFile = ['rt_by_uncertainty_without5_' num2str(trials) 'trials.xls'];
    dlmwrite(xlFile, cp_data_subject, '-append', 'delimiter', '\t');

end
