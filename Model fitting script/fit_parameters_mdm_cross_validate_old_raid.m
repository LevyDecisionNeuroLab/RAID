% NOTE: Requires MATLAB optim library

% run this file to fit all possible models to each individual subject
% model fitting results saved in MLE structures
% subjective ratings are also saved in the *_fitpar.mat

clearvars
close all

%% start par pool for parallel computing
poolobj = parpool('local', 6);

%% attach function so that workers can find and use it
addAttachedFiles(gcp,["fit_ambigNrisk_model.m" "ambig_utility.m" "choice_prob_ambigNrisk.m" "create_choice_matrix.m"...
    "exportfig.m" "find_extreme.m" "fit_ambigNrisk_model_Constrained.m" "getSubjectsData.m" "getSubjectsInDir.m" ...
    "individual_example_plot.m" "load_mat.m" "print_choice_prob.m" "print_choice_prob_outcome.m" "print_cross_validation_mse.m" ...
    "print_fitpar_files.m" "print_fitpar_files_rating.m" "print_fitpar_rt.m" "save_mat.m" "simulation.m" "fminsearchcon.m"])

%% Define conditions
fitparwave = 'Behavior data fitpar_12012022loss2'; % folder to save all the fitpar data structures
fitbywhat = 'value'; % what to use as values 'value', 'rating', 'arbitrary'(0,1,2,3,4)
model = 'ambigNrisk'; % which utility function 'ambigNrisk'
includeAmbig = true; %true or false
search = 'grid'; % 'grid', 'single'
optimizer = 'fminunc';
k = 5; % # of folds of k-fold cross validation
use_cv = false; % true or false

%% set up fitting parameters

% grid search
grid_step = 0.5;

if strcmp(search, 'grid')
    % grid search
    % range of each parameter
    if strcmp(model,'ambigNrisk')
        slopeRange = -4:grid_step:-0.01;
        bRange = -2:grid_step:2;
        aRange = 0.01:grid_step:4;
    else
        slopeRange = -4:grid_step:1;
        bRange = -2:grid_step:2;
        aRange = -2:grid_step:2;
    end
    % three dimensions
    [b1, b2, b3] = ndgrid(slopeRange, bRange, aRange);
    % all posibile combinations of three parameters
    b0 = [b1(:) b2(:) b3(:)];
elseif strcmp(search,'single')
    b0 = [-1 0.5 0.5]; % starting point of the search process, [gamma, beta, alpha]
end

fixed_ambig = 0;
fixed_valueP = 5; % Value of fixed reward
fixed_prob = 1;   % prb of fixed reward 

%% Set up loading & subject selection
root = 'D:\Chelsea\Projects_in_the_lab\RAID\';
data_path = fullfile(root, 'behavioral'); % root of folders is sufficient
fitpar_out_path = fullfile(root, 'model_results', fitparwave);
%exclude_list = [14,19,23,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,48,49];
subjects = [11,12,13,15,16,17,19,20,21,22,24,25,27,28,29,30,31,32,36,39,40,41,42,43,45,46,47,48,50,51,55,56,57,61,62];
% if folder does not exist, create folder
if exist(fitpar_out_path)==0
    mkdir(fitpar_out_path)
end

addpath(genpath(data_path));

% load subjective ratings
% column1-subj ID, c2-$0, c3-$5,c4-$8,c5-$12,c6-$25,c7-no effect, c8-slight,c9-moderate,c10-major,c11-recovery.
%rating = csvread(rating_filename,1,0); %reads data from the file starting at row offset R1 and column offset C1. For example, the offsets R1=0, C1=0 specify the first value in the file.

%% Individual subject fitting
tic

% matrix to store MSE, row (subject number x k folds) x column (2 domains)
% does not work for parfor
% err_mse = zeros(length(subjects)*k,2);

% use structure, dimension # of subjects x 1
err_mse = struct('id',cell(length(subjects),1),'mse',cell(length(subjects),1));
%PerSubjectData = struct();

parfor subj_idx = 1:length(subjects)
    %parfor sub_idx = 1:10
  domains = {'LOSS'};
  
  % mse for a single subject's k fold and 2 domains
  err_domain = zeros(k, 2);
  
  for domain_idx = 1:length(domains)
    subjectNum = subjects(subj_idx);
    domain = domains{domain_idx};
    
    data = load_mat(subjectNum,domain);
    
    % data = struct;
    % load/save does not work in parfor, instead, using a function
    %data= load_mat(subjectNum, domain);
    
%     %% Load subjective ratings
%     % prepare subjective rating for each trial
%     if strcmp(domain, 'MON') ==1 % Monetary block
%         subjRefRatings = rating(find(rating(:,1)==subjectNum),3) * ones(length(data.choice), 1);
%         %values = data.val(include_indices);
%         subjRatings = ones(length(data.val),1);
%         for i=1:length(subjRatings)
%             subjRatings(i) = rating(find(rating(:,1)==subjectNum),1+find(rating(1,2:6)==data.val(i)));
%         end
%     else % Medical block
%         subjRefRatings = rating(find(rating(:,1)==subjectNum),8) * ones(length(data.choice), 1);
%         %values = data.val(include_indices);
%         subjRatings = ones(length(data.val),1);
%         for i=1:length(subjRatings)
%             subjRatings(i) = rating(find(rating(:,1)==subjectNum),6+find(rating(1,7:11)==data.val(i)));
%         end
%     end
    
    %% Refine variables
    
    if includeAmbig
        % Exclude non-responses
%         include_indices = data.choice ~= 0;
        include_indices = ~isnan(data.choice);
    else
        % Exclude ambiguious trials (fit only risky trials)
%        include_indices = data.ambigs' ~= 0 & data.choice ~= 0;
        include_indices = data.ambigs ~= 0 & ~isnan(data.choice);
    end

    choice = data.choice(include_indices);
    values = data.vals(include_indices);
    ambigs = data.ambigs(include_indices);
    probs  = data.probs(include_indices);
    %ratings = subjRatings(include_indices);
    %refRatings = subjRefRatings(include_indices);
    
    % Side with lottery is counterbalanced across subjects 
    % code 0 as reference choice, 1 as lottery choice
    if data.refSide == 2
        choice(choice == 2) = 0;
        choice(choice == 1) = 1;
    elseif data.refSide == 1 % Careful: rerunning this part will make all choices 0
        choice(choice == 1) = 0;
        choice(choice == 2) = 1;
    end
    
    % choice data for $5 only, for rationality check only
    idx_only5 = and(data.choice ~= 0, data.vals' == 5);
    choice5 = data.choice(idx_only5);
    values5 = data.vals(idx_only5);
    ambigs5 = data.ambigs(idx_only5);
    probs5  = data.probs(idx_only5);
    
    if data.refSide == 2
        choice5(choice5 == 2) = 0;
        choice5(choice5 == 1) = 1;
    elseif data.refSide == 1 % Careful: rerunning this part will make all choices 0
        choice5(choice5 == 1) = 0;
        choice5(choice5 == 2) = 1;
    end    
    
    choice_prob_5= sum(choice5)/length(choice5);
    
    %% split training and testing, by k-fold cross-validation
    n_trial = length(choice);
   
    cv = cvpartition(choice,'KFold',k);
       
    %% Fitting, whether to use outcome magnitude or rating is conditioned on the variable 'fitbyrating' 
    
    % define fitting values if fitting by rating
%     if strcmp(fitbywhat,'rating')     
%         fixed_valueP = refRatings(1);
%         fitrefVal = refRatings;
%         fitVal = ratings;
%     end
    
    % define fitting values if fitting by arbitrary units
%     if strcmp(fitbywhat,'arbitrary')     
%         fixed_valueP = 1;
%         fitrefVal = fixed_valueP * ones(length(choice), 1);
%         fitVal = values;
%         % change to arbitrary units: 5->1, 8->2, 12->3, 25->4
%         fitVal(values==5) = 1;
%         fitVal(values==8) = 2;
%         fitVal(values==12) = 3;
%         fitVal(values==25) = 4;
%     end
    
    % define fitting values if fit by objective value in the monetary domain
    if strcmp(fitbywhat,'value') %&& strcmp(domain, 'GAINS') ==1 
        fixed_valueP = 5; % Value of fixed reward
        fitrefVal = fixed_valueP * ones(length(choice), 1);
        fitVal = values;
%     elseif strcmp(fitbywhat,'value') && strcmp(domain, 'LOSS') ==1
%         fixed_valueP = -5; % Value of fixed reward
%         fitrefVal = fixed_valueP * ones(length(choice), 1);
%         fitVal = -values;        
    end
         
    % fit the model if not cross-validating
    if ~use_cv
        if strcmp(fitbywhat,'value') || strcmp(fitbywhat,'rating') || strcmp(fitbywhat,'arbitrary')
            fixed_prob = 1;   % prb of fixed reward
            refProb = fixed_prob  * ones(length(choice), 1);
            fixed_ambig = 0;
            ambig = unique(ambigs(ambigs > 0)); % All non-zero ambiguity levels
            prob = unique(probs); % All probability levels
            base = 0;

            % Two versions of function:
            %       fit_ambgiNrisk_model: unconstrained
            %       fit_ambigNrisk_model_Constrained: constrained on alpha and beta
        
            % Unconstrained fitting
%             [info, p] = fit_ambigNrisk_model(choice, ...
%                 fitrefVal', ...
%                 fitVal', ...
%                 refProb', ...
%                 probs', ...
%                 ambigs', ...
%                 model, ...
%                 b0, ...
%                 base, ...
%                 optimizer);
        
            % Constrained fitting
            % !Need to correct the value, because the design is 100-fold value
          [info, p] = fit_ambigNrisk_model_Constrained(choice, ...
              fitrefVal', ...
              fitVal', ...
              refProb', ...
              probs', ...
              ambigs', ...
              model, ...
              b0, ...
              base, ...
              optimizer);

            slope = info.b(1);
            a = info.b(3);
            b = info.b(2);
            r2 = info.r2;    
        
            model_fit = struct('id', subjectNum,...
                'info', info,...
                'p', p,...
                'fitby', fitbywhat,...
                'search', search)
        end
    
        % choice probability for each trial based on fitted model parameters
        % should not using the model fitting inputs, but rather also
        % include missing response trials. So IMPORTANTLY, use all trials!
        if (strcmp(fitbywhat,'value') && strcmp(domain, 'GAINS') ==1)
            choiceModeled = choice_prob_ambigNrisk(base,fixed_valueP * ones(length(data.vals), 1)',data.vals',...
                fixed_prob  * ones(length(data.vals), 1)',data.probs',data.ambigs',info.b,model);
        elseif (strcmp(fitbywhat,'value') && strcmp(domain, 'LOSS') ==1)
            choiceModeled = choice_prob_ambigNrisk(base,fixed_valueP * ones(length(data.vals), 1)',-data.vals',...
                fixed_prob  * ones(length(data.vals), 1)',data.probs',data.ambigs',info.b,model);            
%         elseif strcmp(fitbywhat,'rating')
%             choiceModeled = choice_prob_ambigNrisk(base,fixed_valueP * ones(length(subjRatings), 1)',subjRatings',...
%                 fixed_prob  * ones(length(data.vals), 1)',data.probs',data.ambigs',info.b,model);
%         elseif strcmp(fitbywhat,'arbitrary')
%             % transform vals into arbitrary units
%             aUnits = data.vals; 
%             aUnits(data.vals == 5) = 1;
%             aUnits(data.vals == 8) = 2;
%             aUnits(data.vals == 12) = 3;
%             aUnits(data.vals == 25) = 4;
%             
%             choiceModeled = choice_prob_ambigNrisk(base,fixed_valueP * ones(length(data.vals), 1)',aUnits',...
%                 fixed_prob  * ones(length(data.vals), 1)',data.probs',data.ambigs',info.b,model);            
        end      
        
        sv = zeros(length(data.choice),1);
        svRef = 0;
        
%         calculate subject values by unconstrained fit
        if strcmp(fitbywhat,'value') && strcmp(domain, 'GAINS') ==1 
            for reps = 1:length(data.choice)
              sv(reps, 1) = ambig_utility(0, ...
                  data.vals(reps), ...
                  data.probs(reps), ...
                  data.ambigs(reps), ...
                  a, ...
                  b, ...
                  model);
            end
        elseif strcmp(fitbywhat,'value') && strcmp(domain, 'LOSS') ==1 
            for reps = 1:length(data.choice)
              sv(reps, 1) = ambig_utility(0, ...
                  -data.vals(reps), ...
                  data.probs(reps), ...
                  data.ambigs(reps), ...
                  a, ...
                  b, ...
                  model);
            end            
%         elseif strcmp(fitbywhat,'rating')
%            for reps = 1:length(data.choice)
%               sv(reps, 1) = ambig_utility(0, ...
%                   subjRatings(reps), ...
%                   data.probs(reps), ...
%                   data.ambig(reps), ...
%                   a, ...
%                   b, ...
%                   model);
%            end   
%         elseif strcmp(fitbywhat,'arbitrary')
%            for reps = 1:length(data.choice)
%               sv(reps, 1) = ambig_utility(0, ...
%                   aUnits(reps), ...
%                   data.probs(reps), ...
%                   data.ambig(reps), ...
%                   a, ...
%                   b, ...
%                   model);
%             end            
            
        end
        
        svRef = ambig_utility(0, ...
              fixed_valueP, ...
              fixed_prob, ...
              fixed_ambig, ...
              a, ...
              b, ...
              model);
          
    %% Chosen SV (cv), chosen reward magnitude (cr), chosen subjective rating (CRating)

    if strcmp(fitbywhat,'value') && strcmp(domain, 'GAINS') ==1 
        valuesAll = data.vals;
        refValue = 5;
    elseif strcmp(fitbywhat,'value') && strcmp(domain, 'LOSS') ==1
        valuesAll = -data.vals;
        refValue = -5;
    end
    
    % All choices
    choiceAll = data.choice;
    ambigsAll = data.ambigs;
    probsAll  = data.probs;
    
%    ratingsAll = subjRatings;
%    refRatingsAll = subjRefRatings;
    
    if strcmp(fitbywhat,'value') || strcmp(fitbywhat,'rating') || strcmp(fitbywhat,'arbitrary')
        %chosen subjective value of all trials
        cv = zeros(length(choiceAll),1);
        cv(cv == 0) = NaN; % this makes sure the missed-trial is marked by NaN
        cv(choiceAll == 0) = svRef;
        cv(choiceAll ==1 ) = sv(choiceAll == 1);
    end
      
     % chosen reward magnitude for both monetary and medical
     cr = zeros(length(choiceAll),1);
     cr(cr == 0) = NaN;
     cr(choiceAll ==0) = refValue;
     cr(choiceAll ==1) = valuesAll(choiceAll ==1 );

     % chosen subjective rating for both monetary and medical
%      cRating = zeros(length(choiceAll),1);
%      cRating(cRating == 0) = NaN;
%      cRating(choiceAll ==0) = refRatingsAll(1);
%      cRating(choiceAll ==1) = ratingsAll(choiceAll ==1 );

    %% Create choice matrices
    % One matrix per condition. Matrix values are binary (0 for sure
    % choice, 1 for lottery). Matrix dimensions are prob/ambig-level
    % x payoff values. Used for graphing and some Excel exports.

    choiceMatrix = create_choice_matrix(values,ambigs,probs,choice);        
          
    %% Create matrix for subjective value
    valueP = unique(values(ambigs == 0));

    if strcmp(fitbywhat,'value')
        if strcmp(domain, 'LOSS') ==1
            valueP = -valueP;
        end
        svByLott = zeros(length(prob)+length(ambig), length(valueP));
        for i = 1:length(prob)+length(ambig)
            for j = 1:length(valueP)
                if i < length(prob)+1
                   svByLott(i,j) = ambig_utility(base, ...
                                               valueP(j), ...
                                               prob(i), ...
                                               0, ...
                                               a, ...
                                               b, ...
                                               model); 
                else
                   svByLott(i,j) = ambig_utility(base, ...
                                               valueP(j), ...
                                               0.5, ...
                                               ambig(i-length(prob)), ...
                                               a, ...
                                               b, ...
                                               model);  
                end
            end
        end
    end    
    %% Save generated values
    PerSubjectData = struct();
    
    PerSubjectData.choiceMatrix = choiceMatrix;
    PerSubjectData.choiceProb5 = choice_prob_5;
            
    %choices per each trial, 0-ref, 1-lottery
    PerSubjectData.choiceLott = choiceAll;
    PerSubjectData.choiceModeled = choiceModeled;
    
    % chosen reward magnitude and chosen subjective value
    PerSubjectData.chosenVal = cr;
%    PerSubjectData.chosenRating = cRating;

    if strcmp(fitbywhat,'value') || strcmp(fitbywhat,'rating') || strcmp(fitbywhat,'arbitrary')
        PerSubjectData.MLE = info;
        PerSubjectData.slope = info.b(1);
        PerSubjectData.alpha = info.b(3);
        PerSubjectData.beta = info.b(2);
        PerSubjectData.r2s = info.r2;
        PerSubjectData.sv = sv;
        PerSubjectData.svRef = svRef;
        PerSubjectData.svChosen = cv;
        PerSubjectData.id = subjectNum;
        PerSubjectData.p = p;
        PerSubjectData.fitby = fitbywhat;
        PerSubjectData.search = search;
        PerSubjectData.model_fit = model_fit;
        PerSubjectData.use_CV = false;
    end
    
    if (strcmp(fitbywhat,'value'))
        PerSubjectData.svByLott = svByLott;
    end

    % save data struct for the two domains
    % load/save does not work in parfor, instead, using a function
    save_mat(PerSubjectData, subjectNum, domain, fitbywhat, fitpar_out_path)    
    %% K-fold cross validation, fit the model on training trials
    else
        slopes = zeros(k, 1);
        alphas = zeros(k, 1);
        betas = zeros(k, 1);
        r2s = zeros(k, 1);
        for fold_idx = 1:k
            % index of training and testing data
            index_train = training(cv,fold_idx);
            index_test = test(cv, fold_idx);

            if strcmp(fitbywhat,'value') || strcmp(fitbywhat,'rating') || strcmp(fitbywhat,'arbitrary')
                % mask the training and testing trials
                choice_train = choice(index_train);
                fitrefVal_train = fitrefVal(index_train);
                fitVal_train = fitVal(index_train);
                probs_train = probs(index_train);
                ambigs_train = ambigs(index_train);

                choice_test = choice(index_test)';
                fitrefVal_test = fitrefVal(index_test);
                fitVal_test = fitVal(index_test);
                probs_test = probs(index_test);
                ambigs_test = ambigs(index_test);


                fixed_prob = 1;   % prb of fixed reward 
                refProb_train = fixed_prob  * ones(length(choice_train), 1);
                fixed_ambig = 0;
                ambig = unique(ambigs_train(ambigs_train > 0)); % All non-zero ambiguity levels 
                prob = unique(probs_train);
                %prob = unique(probs_train); % All probability levels
                base = 0; 

                % if using rating to fit, change base into rating of the null
                % outcome
%                 if strcmp(fitbywhat,'rating')
%                     % monetary
%                     if strcmp(domain, 'MON') == 1
%                         base = rating(rating(:,1)==subjectNum,2);
%                     elseif strcmp(domain, 'MED') == 1
%                         % medical
%                         base = rating(rating(:,1)==subjectNum,7);
%                     end
%                 end

                % Two versions of function:
                %       fit_ambgiNrisk_model: unconstrained
                %       fit_ambigNrisk_model_Constrained: constrained on alpha and beta

                % Unconstrained fitting
                [info, p] = fit_ambigNrisk_model(choice_train', ...
                    fitrefVal_train', ...
                    fitVal_train', ...
                    refProb_train', ...
                    probs_train', ...
                    ambigs_train', ...
                    model, ...
                    b0, ...
                    base,...
                    optimizer);

%                 slope = info.b(1);
                a = info.b(3);
                b = info.b(2);
                r2 = info.r2;
                
                slopes(fold_idx) = info.b(1);
                alphas(fold_idx) = info.b(3);
                betas(fold_idx) = info.b(2);
                r2s(fold_idx) = info.r2;

                %% choice probability for testing trials based on fitted model parameters
                choiceModeled = choice_prob_ambigNrisk(base,fixed_valueP * ones(length(choice_test), 1)',fitVal_test',...
                    fixed_prob  * ones(length(choice_test), 1)',probs_test',ambigs_test',info.b,model);

                % calculate difference between fitted and real choice
                % MSE, mean squared error, per subject per domain
                err_domain(fold_idx, domain_idx) = mean((choiceModeled - choice_test).^2);

            end
        end
    
        %% Save generated values
        PerSubjectData.choiceModeled = choiceModeled;

        if strcmp(fitbywhat,'value') || strcmp(fitbywhat,'rating') || strcmp(fitbywhat,'arbitrary')
            PerSubjectData.MLE = info;
            PerSubjectData.slopes = slopes;
            PerSubjectData.alphas = alphas;
            PerSubjectData.betas = betas;
            PerSubjectData.r2s = r2s;
            PerSubjectData.err_mse = err_domain;
            PerSubjectData.id = subjectNum;
            PerSubjectData.p = p;
            PerSubjectData.fitby = fitbywhat;
            PerSubjectData.search = search;
            PerSubjectData.use_CV = true;
        end

        % save data struct for the two domains
        % load/save does not work in parfor, instead, using a function
        save_mat(PerSubjectData, subjectNum, domain, fitbywhat, fitpar_out_path)

        err_mse(subj_idx).id = subjectNum;
        err_mse(subj_idx).mse = err_domain;
        err_mse(subj_idx).mon_mse = err_domain(:,1);
        err_mse(subj_idx).med_mse = err_domain(:,2);
        %save(fullfile(fitpar_out_path, [model, '_', fitbywhat,'_cross_validation_mse.mat']), 'err_mse')

    
    end
  end
end

toc

% delete(poolobj);
