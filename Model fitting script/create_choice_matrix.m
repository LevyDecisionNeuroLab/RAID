function choiceMatrix = create_choice_matrix(vals,ambigs,probs,choices)

    % Chreate choice probability matrix per condition.Matrix dimensions are prob/ambig-level
    % x payoff values. Used for graphing and some Excel exports.

    % Input: 
    %       vals:       values by trial
    %       ambigs:     ambiguity levels by trial
    %       probs:      probability levels by trial
    %       choices:    choices prob by trial
    %
    % Output:
    %  choiceMatrix
    %       ambigProb:      ambiguity trials, dimension = ambig level * val level
    %       riskProb:       riskty trials, dimension = risk level * val level
    %       ambigCount:     ambiguity trials, count of registered trial number 
    %                       in case subject missed trials
    %       riskCount:      risky trials, count of registered trial number
    %                       in case subject missed trials
    %
    % Histroy
    % ruonan 12.08.2017 written
    
    ambigUniq = unique(ambigs(ambigs > 0)); % All non-zero ambiguity levels 
    probUniq = unique(probs); % All probability levels
    value = unique(vals); % each lottery payoff value under ambiguity

    % Ambiguity levels by payoff values
    ambigChoicesProb = zeros(length(ambigUniq), length(value)); % each row an ambiguity level
    ambigChoicesCount = zeros(length(ambigUniq), length(value)); % count of trials, each row an ambiguity level
    for i = 1:length(ambigUniq)
        for j = 1:length(value)
            selection = find(ambigs == ambigUniq(i) & vals == value(j));
            if ~isempty(selection)
                ambigChoicesCount(i,j) = length(selection);
                ambigChoicesProb(i, j) = mean(choices(selection)); % Have already got rid of NaN choices, do not need to worry about nanmean.
            else
                ambigChoicesProb(i, j) = NaN;
            end
        end
    end
    
    
    % Risk levels by payoff values
    value = unique(vals(ambigs == 0));
    riskyChoicesProb = zeros(length(probUniq), length(value));
    riskyChoicesCount = zeros(length(probUniq), length(value));    
    for i = 1:length(probUniq)
        for j = 1:length(value)
            selection = find(probs == probUniq(i) & vals == value(j) & ambigs == 0);
            if ~isempty(selection)
                riskyChoicesCount(i, j) = length(selection);
                riskyChoicesProb(i, j) = mean(choices(selection));
            else
                riskyChoicesProb(i, j)=NaN;
            end
        end
    end
    
    % output
    choiceMatrix = struct;
    choiceMatrix.ambigProb = ambigChoicesProb;
    choiceMatrix.riskProb = riskyChoicesProb;
    choiceMatrix.ambigCount = ambigChoicesCount;
    choiceMatrix.riskCount = riskyChoicesCount;
end