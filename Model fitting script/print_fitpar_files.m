% This script is meant to take parameters files and print some data to Excel
clearvars
close all

%% Define conditions
fitparwave = '12012022loss2';
fitbywhat = 'value';
includeAmbig = true;

%% Setup
root = 'D:\Chelsea\Projects_in_the_lab\RAID';
function_path = fullfile(root,'scripts','Model fitting', 'Model fitting script');
addpath(function_path)
data_path = fullfile(root, 'behavioral');

subjects = [11,12,13,15,16,17,19,20,21,22,24,25,27,28,29,30,31,32,36,39,40,41,42,43,45,46,47,48,50,51,55,56,57,61,62];
%exclude = []; % TEMPORARY: subjects incomplete data (that the script is not ready for)
%subjects = subjects(~ismember(subjects, exclude));
% subjects = [2585];
% subjects = [2654 2655 2656 2657 2658 2659 2660 2661 2662 2663 2664 2665 2666];

path = fullfile(root, 'model_results', ['Behavior data fitpar_' fitparwave], filesep);
cd(path);

% defining monetary values
valueP = [5 8 12 25];

output_file1 = ['param_' fitparwave '.txt'];
% might not need
% output_file2 = 'choice_data.txt';
% output_file3 = 'choice_prob.txt';

% results file
fid1 = fopen([output_file1],'w')

if strcmp(fitbywhat, 'value')
%     fprintf(fid1,'subject\tmonetary\n')
    fprintf(fid1,'id\talpha_mon\tbeta_mon\tgamma_mon\tLL_mon\tr2_adj_mon\tAIC_mon\tBIC_mon\tmodel_mon\texitFlag_mon\toptimizer_mon\n')
else
%     fprintf(fid1,'subject\tmonetary\t\t\t\t\t\t\t\t\t\t\t\t\tmedical\n')
    fprintf(fid1,'id\talpha_mon\tbeta_mon\tgamma_mon\tLL_mon\tr2_adj_mon\tAIC_mon\tBIC_mon\tmodel_mon\texitFlag_mon\toptimizer_mon\talpha_med\tbeta_med\tgamma_med\tLL_med\tr2_adj_med\tAIC_med\tBIC_med\tmodel_med\texitFlag_med\toptimizer_med\n')
end

% Fill in subject numbers separated by commas
% subjects = {'87','88'};
for s = 1:length(subjects)
    domains = {'LOSS'};
    for domain_idx = 1:length(domains)

%% Print parameters
        subject = subjects(s); 
        domain = domains(domain_idx);
        
        if strcmp(domain, 'GAINS') == 1
            load(['RA_GAINS_' num2str(subject) '_fitpar.mat']);
            choiceMatrixP = Datagain.choiceMatrix;

            riskyChoicesP = choiceMatrixP.riskProb;
            ambigChoicesP = choiceMatrixP.ambigProb;
            riskyChoicesPC = choiceMatrixP.riskCount;
            ambigChoicesPC = choiceMatrixP.ambigCount;

            svByLottP = Datagain.svByLott;
            svRefP = Datagain.svRef;

            alphaP = Datagain.alpha;
        %     alphaseP = Datamon.MLE.se(3);
            betaP = Datagain.beta;
        %     betaseP = Datamon.MLE.se(2);
            gammaP = Datagain.slope;
        %     gammaseP = Datamon.MLE.se(1);
            LLP = Datagain.MLE.LL;
            r2_adjP = Datagain.MLE.r2_adj;
            AICP = Datagain.MLE.AIC;
            BICP = Datagain.MLE.BIC;
            modelP = Datagain.MLE.model;
            exitFlagP = Datagain.MLE.exitflag;
            optimizerP = Datagain.MLE.optimizer;
            
            fprintf(fid1,'%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%f\t%s\n',...
                num2str(subject),alphaP,betaP,gammaP,LLP,r2_adjP,AICP,BICP,modelP,exitFlagP,optimizerP)
            
        elseif strcmp(domain, 'LOSS') == 1

            load(['RA_LOSS_' num2str(subject) '_fitpar.mat']);
            choiceMatrixN = Dataloss.choiceMatrix;

            riskyChoicesN = choiceMatrixN.riskProb;
            ambigChoicesN = choiceMatrixN.ambigProb;
            riskyChoicesNC = choiceMatrixN.riskCount;
            ambigChoicesNC = choiceMatrixN.ambigCount;
            svByLottN = Dataloss.svByLott;
            svRefN = Dataloss.svRef;

            alphaN = Dataloss.alpha;
        %         alphaseN = Datamed.MLE.se(3);
            betaN = Dataloss.beta;
        %         betaseN = Datamed.MLE.se(2);
            gammaN = Dataloss.slope;
        %         gammaseN = Datamed.MLE.se(1);
            LLN = Dataloss.MLE.LL;
            r2_adjN = Dataloss.MLE.r2_adj;
            AICN = Dataloss.MLE.AIC;
            BICN = Dataloss.MLE.BIC;
            modelN = Dataloss.MLE.model;
            exitFlagN = Dataloss.MLE.exitflag;
            optimizerN = Dataloss.MLE.optimizer;

        %write into param text file
        fprintf(fid1,'%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%f\t%s\n',...
                num2str(subject),alphaN,betaN,gammaN,LLN,r2_adjN,AICN,BICN,modelN,exitFlagN,optimizerN)
    end
end
  
    %% print true and model fitting choice prob by trial types
%     choiceMatrixModelP = create_choice_matrix(Datamon.vals,Datamon.ambigs,Datamon.probs,Datamon.choiceModeled);
%     
%     % Plot risky choice prob
%     screensize = get(groot, 'Screensize');
%     fig = figure('Position', [screensize(3)/4 screensize(4)/22 screensize(3)/2 screensize(4)*10/12]);
%     ax1 = subplot(2,1,1);
%     % actual choice
%     barplot = bar(ax1,[choiceMatrixP.riskProb(1,:),choiceMatrixP.riskProb(2,:),choiceMatrixP.riskProb(3,:)],'FaceColor','y');
%     hold on
%     xticklabels({'r25-$5','r25-$8','r25-$12','r25-$25','r50-$5','r50-$8','r50-$12','r50-$25','r75-$5','r75-$8','r75-$12','r75-$25'})
%     ylim([0 1.1])
%     yticks([0:0.1:1.1])
%     % predicted by model
%     plot(ax1,[choiceMatrixModelP.riskProb(1,:),choiceMatrixModelP.riskProb(2,:),choiceMatrixModelP.riskProb(3,:)],'LineStyle','none','Marker','o' )
%     title(ax1,['Subject ' num2str(subject) ' monetary risky choice probability, model:' Datamon.MLE.model])
%     
%     if includeAmbig
%         % Plot ambiguous choice prob
%         ax2 = subplot(2,1,2);
%         % actual choice
%         barplot = bar(ax2,[choiceMatrixP.ambigProb(1,:),choiceMatrixP.ambigProb(2,:),choiceMatrixP.ambigProb(3,:)],'FaceColor','y');
%         hold on
%         xticklabels({'a24-$5','a24-$8','a24-$12','a24-$25','a50-$5','a50-$8','a50-$12','a50-$25','a74-$5','a74-$8','a74-$12','a74-$25'})
%         ylim([0 1.1])
%         yticks([0:0.1:1.1])
%         % predicted by model
%         plot(ax2,[choiceMatrixModelP.ambigProb(1,:),choiceMatrixModelP.ambigProb(2,:),choiceMatrixModelP.ambigProb(3,:)],'LineStyle','none','Marker','o' )
%         title(ax2,['Subject ' num2str(subject) ' monetary ambiguous choice probability, model:' Datamon.MLE.model])
%     
%         % save figure
%         saveas(fig,['Subject ' num2str(subject), ' mon choice prob-' Datamon.MLE.model])
%     end
%     
%     if ~strcmp(fitbywhat, 'value')
%         choiceMatrixModelN = create_choice_matrix(Datamed.vals,Datamed.ambigs,Datamed.probs,Datamed.choiceModeled);
% 
%         % Plot risky choice prob
%         screensize = get(groot, 'Screensize');
%         fig = figure('Position', [screensize(3)/4 screensize(4)/22 screensize(3)/2 screensize(4)*10/12]);
%         ax1 = subplot(2,1,1);
%         % actual choice
%         barplot = bar(ax1,[choiceMatrixN.riskProb(1,:),choiceMatrixN.riskProb(2,:),choiceMatrixN.riskProb(3,:)],'FaceColor','y');
%         hold on
%         xticklabels({'r25-sl','r25-mod','r25-maj','r25-rec','r50-sl','r50-mod','r50-maj','r50-rec','r75-sl','r75-mod','r75-maj','r75-rec'})
%         ylim([0 1.1])
%         yticks([0:0.1:1.1])
%         % predicted by model
%         plot(ax1,[choiceMatrixModelN.riskProb(1,:),choiceMatrixModelN.riskProb(2,:),choiceMatrixModelN.riskProb(3,:)],'LineStyle','none','Marker','o' )
%         title(ax1,['Subject ' num2str(subject) ' medical risky choice probability, model:' Datamed.MLE.model])
% 
%         if includeAmbig
%             % Plot ambiguous choice prob
%             ax2 = subplot(2,1,2);
%             % actual choice
%             barplot = bar(ax2,[choiceMatrixN.ambigProb(1,:),choiceMatrixN.ambigProb(2,:),choiceMatrixN.ambigProb(3,:)],'FaceColor','y');
%             hold on
%             xticklabels({'a24-sl','a24-mod','a24-maj','a24-rec','a50-sl','a50-mod','a50-maj','a50-rec','a74-sl','a74-mod','a74-maj','a74-rec'})
%             ylim([0 1.1])
%             yticks([0:0.1:1.1])
%             % predicted by model
%             plot(ax2,[choiceMatrixModelN.ambigProb(1,:),choiceMatrixModelN.ambigProb(2,:),choiceMatrixModelN.ambigProb(3,:)],'LineStyle','none','Marker','o' )
%             title(ax2,['Subject ' num2str(subject) ' medical ambiguous choice probability, model:' Datamed.MLE.model])
%         end
%         
%         % save figure
%         saveas(fig,['Subject ' num2str(subject), ' med choice prob-' Datamed.MLE.model])
% 
%     end
    
    
    %% for Excel file - choice prob by lottery type
    
%     % Firt, combine choice data with and without $4
%     choices_allP = [riskyChoicesP; ambigChoicesP];
%     if strcmp(fitbywaht, 'value') == 0
%         choices_allN = [riskyChoicesN; ambigChoicesN];
%     end
%     
%     all_data_subject = [valueP; choices_allP ;valueP; choices_allN];
%     
%     xlFile = ['choice_data.xls'];
%     dlmwrite(xlFile, subject , '-append', 'roffset', 1, 'delimiter', ' ');  
%     dlmwrite(xlFile, all_data_subject, 'coffset', 1, '-append', 'delimiter', '\t');
    
    %% for Excel file - subjective values
%     xlFile = ['SV_unconstrained_by_lottery.xls'];
%     dlmwrite(xlFile, subject, '-append', 'roffset', 1, 'delimiter', ' '); 
%     dlmwrite(xlFile, svRefUncstr, '-append', 'coffset', 1, 'delimiter', '\t');
%     dlmwrite(xlFile, svUncstrByLott, 'coffset', 1, '-append', 'delimiter', '\t');
    
    %% for Excel file - choice prob by uncertainty level
    
    %exclude choices with $5 or slight improvement
    %P is monetary, N is medical
%     riskyChoicesP = riskyChoicesP(:,2:size(riskyChoicesP,2));
%     riskyChoicesPC = riskyChoicesPC(:,2:size(riskyChoicesPC,2));
%     ambigChoicesP = ambigChoicesP(:,2:size(ambigChoicesP,2));
%     ambigChoicesPC = ambigChoicesPC(:,2:size(ambigChoicesPC,2));
%     riskyChoicesN = riskyChoicesN(:,2:size(riskyChoicesN,2));
%     riskyChoicesNC = riskyChoicesNC(:,2:size(riskyChoicesNC,2));
%     ambigChoicesN = ambigChoicesN(:,2:size(ambigChoicesN,2));
%     ambigChoicesNC = ambigChoicesNC(:,2:size(ambigChoicesNC,2));
% 
%     % monetary
%     riskyChoicesPT = riskyChoicesP .* riskyChoicesPC; % choice total counts = choice prob * trial counts
%     cpByRiskP = zeros(size(riskyChoicesP,1),1); % choice prob by risk level
%     for i = 1:size(cpByRiskP,1);
%       cpByRiskP(i) = sum(riskyChoicesPT(i,:))/sum(riskyChoicesPC(i,:));
%     end
%     cpRiskAllP = sum(riskyChoicesPT(:))/sum(riskyChoicesPC(:));
%     
%     ambigChoicesPT = ambigChoicesP .* ambigChoicesPC; % choice total counts = choice prob * tial counts
%     cpByAmbigP = zeros(size(ambigChoicesP,1),1); % choice prob by ambig level
%     for i = 1:size(cpByAmbigP,1);
%       cpByAmbigP(i) = sum(ambigChoicesPT(i,:))/sum(ambigChoicesPC(i,:));
%     end
%     cpAmbigAllP = sum(ambigChoicesPT(:))/sum(ambigChoicesPC(:));
%     
%     ambigAttP = ambigChoicesP - riskyChoicesP(2,:); 
%     ambigAttByAmbigP = nanmean(ambigAttP.'); % model free ambig attitude by ambig level
%     ambigAttAllP = nanmean(ambigAttP(:));
%     
%     AllPT = sum(riskyChoicesPT(:)) + sum(ambigChoicesPT(:)); % choice total counts = choice prob * tial counts
%     AllPC = sum(riskyChoicesPC(:)) + sum(ambigChoicesPC(:));
%     cpAllP = sum(AllPT(:))/AllPC;
% 
%    
%     
%     %Medical
%     riskyChoicesNT = riskyChoicesN .* riskyChoicesNC; % choice total counts = choice prob * tial counts
%     cpByRiskN = zeros(size(riskyChoicesN,1),1); % choice prob by risk level
%     for i = 1:size(cpByRiskN,1);
%       cpByRiskN(i) = sum(riskyChoicesNT(i,:))/sum(riskyChoicesNC(i,:));
%     end
%     cpRiskAllN = sum(riskyChoicesNT(:))/sum(riskyChoicesNC(:));
%     
%     ambigChoicesNT = ambigChoicesN .* ambigChoicesNC; % choice total counts = choice prob * tial counts
%     cpByAmbigN = zeros(size(ambigChoicesN,1),1); % choice prob by ambig level
%     for i = 1:size(cpByAmbigN,1);
%       cpByAmbigN(i) = sum(ambigChoicesNT(i,:))/sum(ambigChoicesNC(i,:));
%     end
%     cpAmbigAllN = sum(ambigChoicesNT(:))/sum(ambigChoicesNC(:));
%     
%     ambigAttN = ambigChoicesN - riskyChoicesN(2,:); 
%     ambigAttByAmbigN = nanmean(ambigAttN.'); % model free ambig attitude by ambig lavel
%     ambigAttAllN = nanmean(ambigAttN(:));
%     
%     AllNT = sum(riskyChoicesNT(:)) + sum(ambigChoicesNT(:)); % choice total counts = choice prob * tial counts
%     AllNC = sum(riskyChoicesNC(:)) + sum(ambigChoicesNC(:));
%     cpAllN = sum(AllNT(:))/AllNC;
% 
%     
% 
%    cp_title = {'subject ID', 'r25', 'r50','r75','rAll','a24', 'a50','a74','aAll','a24-r50', 'a50-r50','a74-r50','a-r50 All','All',...
%                              'r25', 'r50','r75','rAll','a24', 'a50','a74','aAll','a24-r50', 'a50-r50','a74-r50','a-r50 All','All'};
%    cp_data_subject = [subject,cpByRiskP.',cpRiskAllP,cpByAmbigP.',cpAmbigAllP,ambigAttByAmbigP,ambigAttAllP,cpAllP...
%                               cpByRiskN.',cpRiskAllN,cpByAmbigN.',cpAmbigAllN,ambigAttByAmbigN,ambigAttAllN,cpAllN];
%                           
%     xlFile = ['choice_prob_without5.xls'];
%     dlmwrite(xlFile, cp_data_subject, 'coffset', 1, '-append', 'delimiter', '\t');

end

fclose(fid1);
