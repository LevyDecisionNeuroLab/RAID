clearvars
close all

%% Define conditions and load data
fitparwave = 'Behavior data fitpar_07062021'; % folder to save all the fitpar data structures
fitbywhat = 'rating'; % what to use as values 'value', 'rating', 'arbitrary'(0,1,2,3,4)
model = 'ambigNrisk'; % which utility function 'ambigNrisk'
k = 5; % # of folds of k-fold cross validation

root = 'E:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis';
fitpar_out_path = fullfile(root, 'Behavior fitpar files',fitparwave);

load(fullfile(fitpar_out_path, [model, '_', fitbywhat,'_cross_validation_mse.mat']));

%% write into csv
err2write_mon = zeros(length(err_mse) * k , 3);
err2write_med = zeros(length(err_mse) * k , 3);

for i=1:length(err_mse)
    id = err_mse(i).id;
    mse = err_mse(i).mse;
    
    err2write_mon(((i-1)*k +1):(i*k),1) = id;
    err2write_mon(((i-1)*k +1):(i*k),2) = 0;
    err2write_mon(((i-1)*k +1):(i*k),3) = mse(:,1);
    
    
    err2write_med(((i-1)*k +1):(i*k),1) = id;
    err2write_med(((i-1)*k +1):(i*k),2) = 1;
    err2write_med(((i-1)*k +1):(i*k),3) = mse(:,2);    
end

% concatenate
err2write = vertcat(err2write_mon, err2write_med);

% turn into table
err_table = array2table(err2write);
err_table.Properties.VariableNames = {'id', 'is_med', 'mse'};
err_table.model = repmat({model}, size(err2write, 1), 1);
err_table.fitby = repmat({fitbywhat}, size(err2write, 1), 1);

writetable(err_table,...
    fullfile(fitpar_out_path,[model, '_', fitbywhat,'_cross_validation_mse.csv']));


