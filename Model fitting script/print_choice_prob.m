%% This script is meant to take parameters files and print some data to Excel
clear all
close all
%cd 'C:\Users\lr382\Desktop\Lital\RISK-VA\Behavior for PTB\';
%addpath(genpath('C:\Users\lr382\Desktop\Lital\RISK-VA\Behavior for PTB\'));

root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis';
data_path = fullfile(root, 'PTB Behavior Log/'); % Original log from PTB
subjects = getSubjectsInDir(data_path, 'subj'); %function
exclude = [2581 2587]; % TEMPORARY: subjects incomplete data (that the script is not ready for)
subjects = subjects(~ismember(subjects, exclude));
subjects = [2654 2655 2656 2657 2658 2659 2660 2661 2662 2663 2664 2665 2666]

fitparwave = 'Behavior data fitpar_0520012018';
cd (['D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\Behavior fitpar files\' fitparwave]);
path = ['D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\Behavior fitpar files\' fitparwave];

% defining unique values
valueLevel = [8 12 25];
riskLevel = [0.25 0.5 0.75];
ambigLevel = [0.24 0.5 0.74];

domain = {'MON','MED'};

rng('shuffle')

% Fill in subject numbers separated by commas
% subjects = {'87','88'};
for s = 1:length(subjects)
    
    subject = subjects(s); 
    % load monetary file for subject and extract params & choice data
    %ChoicesP stands for monetary, ChoicesN stands for medical
    for d = 1:length(domain);
        load(['MDM_' domain{d} '_' num2str(subject) '_fitpar.mat']);
        
        ismon = strcmp(domain{d}, 'MON');
        
        if ismon
            data = Datamon;
            clear Datamon
        else
            data = Datamed;
            clear Datamed
        end
        
        % load data
        probs = data.probs;
        ambigs = data.ambigs;
        choices = data.choiceLott;
        vals = data.vals;
        
        % exclude trials with val==5
        include = find(vals ~=5);
        probs = probs(include);
        ambigs = ambigs(include);
        choices = choices(include);
        vals = vals(include);
        
 
        %% trials into halfs
        % There are two ways of doing this: 
        % First: randmly seperate trials, but risk and ambig seperately.
        % Second: For the four repeats of each lottery type, randomly
        %         assign two into fist/second halfe. PREFERRED
        
%         % frist way:
%         % seperate risk and ambig trials
%         riskIndex = find(ambigs == 0);
%         ambigIndex = find(ambigs ~= 0);
%         
%         riskProbs = probs(riskIndex);
%         riskAmbigs = ambigs(riskIndex);
%         riskVals = vals(riskIndex);
%         riskChoices = choices(riskIndex);
%         
%         ambigProbs = probs(ambigIndex);
%         ambigAmbigs = ambigs(ambigIndex);
%         ambigVals = vals(ambigIndex);
%         ambigChoices = choices(ambigIndex);
%         
%         % randomize the trials into two halfs
%         
%         % Risk
%         riskRand = randperm(36);
%         % Goes to first half
%         riskProbs1 = riskProbs(riskRand(1:18));
%         riskAmbigs1 = riskAmbigs(riskRand(1:18));
%         riskVals1 = riskVals(riskRand(1:18));
%         riskChoices1 = riskChoices(riskRand(1:18));
%         % Goes to second half
%         riskProbs2 = riskProbs(riskRand(19:36));
%         riskAmbigs2 = riskAmbigs(riskRand(19:36));
%         riskVals2 = riskVals(riskRand(19:36));
%         riskChoices2 = riskChoices(riskRand(19:36));
%         
%         % Ambig
%         ambigRand = randperm(36);
% 
%         ambigProbs1 = ambigProbs(ambigRand(1:18));
%         ambigAmbigs1 = ambigAmbigs(ambigRand(1:18));
%         ambigVals1 = ambigVals(ambigRand(1:18));
%         ambigChoices1 = ambigChoices(ambigRand(1:18));
% 
%         ambigProbs2 = ambigProbs(ambigRand(19:36));
%         ambigAmbigs2 = ambigAmbigs(ambigRand(19:36));
%         ambigVals2 = ambigVals(ambigRand(19:36));
%         ambigChoices2 = ambigChoices(ambigRand(19:36));
%         
        
        % Second way:
        separate = zeros(72,1); % stores 1 and 2
        
        for v = 1:length(valueLevel);
            for r = 1:length(riskLevel);
                currentRepeats = find(vals==valueLevel(v) & probs==riskLevel(r) & ambigs==0); %the index for the 4 repeats of current trial types
                index = randperm(4);
                separate(currentRepeats(index(1:2))) = 1;
                separate(currentRepeats(index(3:4))) = 2;
            end
            
            for a = 1:length(ambigLevel);
                currentRepeats = find(vals==valueLevel(v) & ambigs==ambigLevel(a) & probs==0.5);
                index = randperm(4);
                separate(currentRepeats(index(1:2))) = 1;
                separate(currentRepeats(index(3:4))) = 2;
            end
        end
                
            
        
        %% Choice probability matrix and print, loop through first and second half
        for i = 1:2 % loop through first and second half
            valsHalf = vals(separate==i);
            probsHalf = probs(separate==i);
            ambigsHalf = ambigs(separate==i);
            choicesHalf = choices(separate==i);
            
            %Sum and counts of choices
            %ambig trials
            ambigChoicesS = zeros(length(ambigLevel), length(valueLevel)); % choice sum, each row an ambiguity level
            ambigChoicesC = zeros(length(ambigLevel), length(valueLevel)); % count of trials, each row an ambiguity level
            for a = 1:length(ambigLevel)
                for v = 1:length(valueLevel)
                    selection = find(ambigsHalf == ambigLevel(a) & valsHalf == valueLevel(v));
                    if ~isempty(selection)
                        ambigChoicesC(a,v) = length(selection);
                        ambigChoicesS(a, v) = nansum(choicesHalf(selection)); % Have already got rid of NaN choices, do not need to worry about nanmean.
                    else
                        ambigChoicesC(a, v) = NaN;
                    end
                end
            end
            %risk trials
            riskChoicesS = zeros(length(riskLevel), length(valueLevel)); % choice sum, each row an riskuity level
            riskChoicesC = zeros(length(riskLevel), length(valueLevel)); % count of trials, each row an riskuity level
            for a = 1:length(riskLevel)
                for v = 1:length(valueLevel)
                    selection = find(probsHalf == riskLevel(a) & valsHalf == valueLevel(v) & ambigsHalf ==0);
                    if ~isempty(selection)
                        riskChoicesC(a,v) = length(selection);
                        riskChoicesS(a, v) = nansum(choicesHalf(selection)); % Have already got rid of NaN choices, do not need to worry about nanmean.
                    else
                        riskChoicesC(a, v) = NaN;
                    end
                end
            end
            
            % Choice prob matrix
            % ambig
            ambigChoicesP = ambigChoicesS ./ ambigChoicesC;             
            byAmbigChoicesP = nansum(ambigChoicesS,2) ./ nansum(ambigChoicesC,2); % will be a column vector, each row is an ambiguity level
            ambigAllP = nansum(ambigChoicesS(:)) ./ nansum(ambigChoicesC(:));
            % risk
            riskChoicesP = riskChoicesS ./ riskChoicesC;
            byRiskChoicesP = nansum(riskChoicesS, 2) ./ nansum(riskChoicesC,2);
            riskAllP = nansum(riskChoicesS(:)) ./ nansum(riskChoicesC(:));
            % ambig-risk50
            ambigAtt = ambigChoicesP - riskChoicesP(2,:);
            byAmbigAmbigAtt = nanmean(ambigAtt,2);
            ambigAttAll = nanmean(byAmbigAmbigAtt);
            % All
            AllP = (nansum(ambigChoicesS(:)) + nansum(riskChoicesS(:))) ./ (nansum(ambigChoicesC(:)) + nansum(riskChoicesC(:)));

            %% for Excel file - choice matrix prob by lottery type
            % four sheet: MON1, MON2, MED1, MED2
            % prepare matrix for print
            choicesP_all = [8 12 25; riskChoicesP; ambigChoicesP; ambigAtt];

            xlFile = [domain{d} num2str(i) '_choice_data.xls'];
            dlmwrite(xlFile, subject , '-append', 'roffset', 1, 'delimiter', ' ');  
            dlmwrite(xlFile, choicesP_all, 'coffset', 1, '-append', 'delimiter', '\t');
            
            %% for Excel file - choice prob by uncertainty level

            %exclude choices with $5 or slight improvement


           cp_title = {'subject ID', 'r25', 'r50','r75','rAll','a24', 'a50','a74','aAll','a24-r50', 'a50-r50','a74-r50','a-r50 All','All'};
           cp_data_subject = [subject,byRiskChoicesP.',riskAllP,byAmbigChoicesP.',ambigAllP,byAmbigAmbigAtt.',ambigAttAll,AllP];

            xlFile = [domain{d} num2str(i) '_choice_prob_without5.xls'];
                       
            dlmwrite(xlFile, cp_data_subject, '-append', 'delimiter', '\t');
            
            
            
                      
        end    
        
        


    end
    
    

end

