% This script 

value = [5,8,12,25];
probLevel = [.25,.5,.75];
ambigLevel = [.24,.5,.74];
trialRep = 4;

fixedVal = 5;
fixedProb = 1;
fixedAmbig = 0;

base = 0;
model = 'ambigNrisk';

permNum = 1;
simNum = 100; % generate a parameter set for each simulation ( equals to  subject number)

p = [probLevel, 0.5.*ones(size(ambigLevel))]';
a = [zeros(size(probLevel)), ambigLevel]';
v = repmat(value,1,trialRep)';
[val, prob] = ndgrid(v,p);
[~, ambig] = ndgrid(v,a);
val = val(:);
prob = prob(:);
ambig = ambig(:);

trialNum = length(val);

for i = 1:permNum
    %% true parameter distribution
    alpha = [0.8,0.2]; % [mean, standard deviation]
    beta = [0.5, 0.2];
    gamma = [-1,0.2];
    
    multisubjChoice = zeros(trialNum,simNum); % big matrix of all subjects choice data, each column is a subject
 
    %% generate choice data for all subjects
    
    for j = 1:simNum % for each subject
        % generate parameters from normal distribution
        b = [normrnd(gamma(1),gamma(2)), normrnd(beta(1),beta(2)), normrnd(alpha(1),alpha(2))];
        
        choice = zeros(size(val));
        
        for t = 1:trialNum
           if ambig_utility(base,val(t),prob(t),ambig(t),b(3),b(2),model) > ambig_utility(base,fixedVal,fixedProb,fixedAmbig,b(3),b(2),model)
               choice(t) = 1;
           elseif ambig_utility(base,val(t),prob(t),ambig(t),b(3),b(2),model) < ambig_utility(base,fixedVal,fixedProb,fixedAmbig,b(3),b(2),model)
               choice(t) = 0;
           elseif ambig_utility(base,val(t),prob(t),ambig(t),b(3),b(2),model) == ambig_utility(base,fixedVal,fixedProb,fixedAmbig,b(3),b(2),model)
               choice(t) = binornd(1,0.5); %equal chance of choosing either option
           end
        end
        multisubjChoice(:,j) = choice;
        clear choice
    end
    
    %% fit model
    slope = zeros(simNum,1);
    a = zeros(simNum,1);
    b = zeros(simNum,1);
    r2 = zerps(simNum,1);
    info = cell(simNum,1);
    
    for s = 1:simNum
        choice = multisubjChoice(:,s);
        
        % grid search
        if strcmp(model,'ambigNrisk')
            slopeRange = -4:0.2:1;
            bRange = -2:0.2:2;
            aRange = 0:0.2:4;
        else
            slopeRange = -4:0.2:1;
            bRange = -2:0.2:2;
            aRange = -2:0.2:2;
        end
        % three dimenstions
        [b1, b2, b3] = ndgrid(slopeRange, bRange, aRange);
        % grid, all posibile combinatinos of three parameters
        b0 = [b1(:) b2(:) b3(:)];
        
        fitrefVal = fixedVal .* ones(size(val));
        refProb = fixedProb .* ones(size(val));
        
        % Unconstrained fitting
        [info{s}, p] = fit_ambigNrisk_model(choice, ...
            fitrefVal, ...
            val, ...
            refProb, ...
            prob, ...
            ambig, ...
            model, ...
            b0, ...
            base);

        slope(s) = info{s}.b(1);
        a(s) = info{s}.b(3);
        b(s) = info{s}.b(2);
        r2(s) = info{s}.r2;        
    end
    
    %% if recover parameters
    alphaFit = [mean(a),std(a)];
    betaFit = [mean(b),std(b)];
    gammaFit = [mean(slope),std(slope)];
    
    % Plot
    fig = figure;
    

end