
clearvars
close all
%% set up trial types

% MDM
value = [8 12 25];
prob = [0.25 0.5 0.75];
ambig = [0.74 0.5 0.24];

value_r = 5;
prob_r = 1;
ambig_r = 0;
prob_ambig = 0.5;

%% risky trials only

alpha_all = zeros(length(prob) * length(value),1);
count = 0;

for prob_idx = 1:length(prob)
    for value_idx = 1:length(value)
        
        count = count + 1;
        
        prob_trial = prob(prob_idx);
        value_trial = value(value_idx);
        
        alpha = log(prob_r/prob_trial)/log(value_trial/value_r);
        alpha_all(count) = alpha;
        
        fprintf('Value %4.2f Prob %4.2f, alpha equals to %6.4f \n', value_trial, prob_trial, alpha);
    end
end

fprintf('\n')

figure
histogram(alpha_all, 20);

alpha_max = max(alpha_all);
alpha_min = min(alpha_all);

%% ambig trials
% from previous step, alpha_max, alpha_min

% for each alpha, calcualte beta
beta_all = zeros(length(alpha_all)*length(value)*length(ambig),1);
count = 0;

for alpha_idx = 1:length(alpha_all)
    for ambig_idx = 1:length(ambig)
        for value_idx = 1:length(value)
            
            count = count+1;
            
            ambig_trial = ambig(ambig_idx);
            value_trial = value(value_idx);

            alpha = alpha_all(alpha_idx);
            
            beta = (prob_ambig - (prob_r*value_r^alpha)/(value_trial^alpha))*2/ambig_trial;
            
            beta_all(count) = beta;
            
            fprintf('Alpha %6.4f Value %4.2f Ambig %4.2f, beta equals to %6.4f \n', alpha, value_trial, ambig_trial, beta);
        end
    end
end

fprintf('\n')

figure
histogram(beta_all, 20);

beta_min = min(beta_all);
beta_max = max(beta_all);