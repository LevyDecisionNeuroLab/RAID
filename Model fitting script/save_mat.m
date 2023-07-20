function save_mat(Data, subjectNum, domain, fitbywhat, fitpar_out_path)

    if strcmp(domain, 'LOSS') ==1
        Dataloss = Data;
        save(fullfile(fitpar_out_path, ['RA_' domain '_' num2str(subjectNum) '_fitpar.mat']), 'Dataloss')
    elseif strcmp(domain, 'GAINS') ==1
        Datagain = Data;
        save(fullfile(fitpar_out_path, ['RA_' domain '_' num2str(subjectNum) '_fitpar.mat']), 'Datagain')
    end