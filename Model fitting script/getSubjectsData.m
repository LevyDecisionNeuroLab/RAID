function [id_unique, data_raw] = getSubjectsData(file_name)
%Extracts subjects data and ids
% Input:
%      - file_name: name of the data sheet
% Output:
%      - id_unique: ids of all subjects
%      - data_raw: data sheet, matlab table

% read .csv
data_raw = readtable(file_name, 'Delimiter', ',','TreatAsEmpty','N/A');

% get rid of NaN lines
% toDelete = isnan(data_raw.is_med);
% data_raw(toDelete,:) = [];

% turn string into numeric
for i = 1:height(data_raw)
    data_raw.ambig{i} = str2num(data_raw.ambig{i});
    data_raw.prob{i} = str2num(data_raw.prob{i});
    data_raw.id{i} = str2num(data_raw.id{i});
end

data_raw.ambig = cell2mat(data_raw.ambig);
data_raw.prob = cell2mat(data_raw.prob);
data_raw.id = cell2mat(data_raw.id);

% take our all subject ID
id = data_raw.id;

% get rid of repetition
id_unique = unique(id);


end