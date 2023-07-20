% FIT_AMBIGNRISK_MODEL      Fit a variety of probabilistic ambiguity models
% 
%     [info,p] = fit_ambigNrisk_model(choice,vF,vA,pF,pA,AL,model);
%
%     Fits a binary logit model by maximum likelihood.
%
%     INPUTS
%     choice      - Dependent variable. The data should be *ungrouped*,
%                   such that CHOICE is a column of 0s and 1s, where 1 indicates 
%                   a choice of the FIXED option.
%     vF          - value of fixed lottery
%     vA          - value of ambiguous lottery
%     pF          - probability of fixed lottery
%     pA          - probability of ambiguous lottery
%     AL          - ambiguity level
%     model       - String indicating which model to fit; currently valid are:
%                   'ambigNrisk'   - power with subjective probability, estimates both risk and ambiguity coefficients 
%                    
%                   Multiple models can be fit by passing in a cell array
%                   of strings. 
%
%     OUTPUTS
%     info       - data structure with following fields:
%                     .nobs      - number of observations
%                     .nb        - number of parameters
%                     .optimizer - function minimizer used
%                     .exitflag  - see FMINSEARCH
%                     .b         - fitted parameters; note that for all the
%                                  available models, the first element of B
%                                  is a noise term for the logistic
%                                  function, the remaining elements are
%                                  parameters for the selected discount
%                                  functions. eg., for model='exp', B(2) is
%                                  the time constant of the exponential
%                                  decay.
%                     .LL        - log-likelihood evaluated at maximum
%                     .LL0       - restricted (minimal model) log-likelihood
%                     .AIC       - Akaike's Information Criterion 
%                     .BIC       - Schwartz's Bayesian Information Criterion 
%                     .r2        - pseudo r-squared
%                   This is a struct array if multiple models are fit.
%     p           - Estimated choice probabilities evaluated at the values
%                   delays specified by the inputs vS, vR, dS, dL. This is
%                   a cell array if multiple models are fit.
%
%     EXAMPLES
%     see TEST_FAKE_DATA_AMBIGUITTY, TEST_FAKE_DATA, TEST_JOE_DATA, and TEST_KENWAY_DATA

%
%     REVISION HISTORY:
%     brian 03.10.06 written
%     brian 03.14.06 added fallback to FMINSEARCH, multiple fit capability
%     ifat  12.01.06 adapted for ambiguity and risk + CI

function [info,p] = fit_ambigNrisk_model_Constrained(choice,vF,vA,pF,pA,AL,model,b0,base,optimizer)
% If multiple model fits requested, loop and pack

if iscell(model)
   for i = 1:length(model)
      [info(i),p{i}] = fit_ambigNrisk_model_Constrained(choice,vF,vA,pF,pA,AL,model{i},b0,base,optimizer);
   end
   return;
end

thresh = 0.05;
nobs = length(choice);

%% Fit model, attempting to use FMINUNC first, then falling back to FMINSEARCH
% For RAID, risk levels are 0.25, 0.5, 0.75. ambig levels are 0.24, 0.5,
% 0.74. 20 outcome levels include 5,6,7,8,10...101, 120.
MIN_LOTTERY = 6;
MAX_LOTTERY = 120;
maxalpha = log(0.25)/log(5/MIN_LOTTERY)
minalpha = log(0.75)/log(5/MAX_LOTTERY)
maxbeta = (0.5 - (5/MAX_LOTTERY)^maxalpha)/(0.24/2)
minbeta = (0.5 - (5/MAX_LOTTERY)^minalpha)/(0.24/2)

if strcmp(optimizer, 'fminunc')
    
    % try multiple starting positions
    for i = 1 : size(b0,1)
        b00 = b0(i,:)'; % search starting point
        
        if exist('fminunc','file')
           try
              optimizer = 'fminunc';
              OPTIONS = optimset('Display','off','LargeScale','off');
                  
              [b,negLL,exitflag,convg] = fmincon(@local_negLL,b00,[],[],[],[],[-inf minbeta minalpha],[inf maxbeta maxalpha],[],OPTIONS,choice,vF,vA,pF,pA,AL,model,base);
%              [b,negLL,exitflag,convg,g,H] = fminunc(@local_negLL_fminunc,b00,OPTIONS,choice,vF,vA,pF,pA,AL,model,base);
              % X = fmincon(fun,x0,A,B,Aeq,Beq,lb,ub,nonlcon,Opts,a,b)  

              if exitflag ~= 1 % trap occasional linesearch failures
                 optimizer = 'fminsearch';
                 fprintf('FMINUNC failed to converge, switching to FMINSEARCH\n');
              end         
           catch
              optimizer = 'fminsearch';
              fprintf('Problem using FMINUNC, switching to FMINSEARCH\n');
           end
        else
           optimizer = 'fminsearch';
        end

        if strcmp(optimizer,'fminsearch')
           optimizer = 'fminsearch';
           OPTIONS = optimset('Display','off','TolCon',1e-6,'TolFun',1e-5,'TolX',1e-5,...
              'DiffMinChange',1e-4,'Maxiter',100000,'MaxFunEvals',20000);
           [b,negLL,exitflag,convg] = fminsearchcon(@local_negLL,b00,[-inf minbeta minalpha],[inf maxbeta maxalpha],[],[],[],OPTIONS,choice,vF,vA,pF,pA,AL,model,base);
%           [b,negLL,exitflag,convg] = fminsearch(@local_negLL_fminunc,b00,OPTIONS,choice,vF,vA,pF,pA,AL,model,base);
        end

        if exitflag ~= 1
           fprintf('Optimization FAILED, #iterations = %g\n',convg.iterations);
        else
           fprintf('Optimization CONVERGED, #iterations = %g\n',convg.iterations);
        end
        
        % Unrestricted log-likelihood
        LL = -negLL;
        if i == 1
            info.LL = LL;
        end

        if i == 1 || (i ~=1 && LL > info.LL)% first iteration; and if a later iteration renders larger likelihood, replace info
            % Choice probabilities (for VARIED)
            p = choice_prob_ambigNrisk(base,vF,vA,pF,pA,AL,b,model);

            % Restricted log-likelihood
            LL0 = sum((choice==1).*log(0.5) + (1 - (choice==1)).*log(0.5)); % assuming no predictors, the chance of choosing and not choosing the lottery are both 50%

            % Confidence interval, requires Hessian from FMINUNC
            try
                invH = inv(-H);
                se = sqrt(diag(-invH));
            catch
            end

            info.nobs = nobs;
            info.nb = length(b);
            info.model = model;
            info.optimizer = optimizer;
            info.exitflag = exitflag;
            info.b = b;

            try
                info.se = se;
                info.ci = [b-se*norminv(1-thresh/2) b+se*norminv(1-thresh/2)]; % Wald confidence
                info.tstat = b./se;
            catch
            end

            info.LL = LL;
            info.LL0 = LL0;
            info.AIC = -2*LL + 2*length(b);
            info.BIC = -2*LL + length(b)*log(nobs);
            info.r2 = 1 - LL/LL0; % McFadden's Pseudo r squared = 1-LLmodel/LLwithoutModel
            info.r2_adj = 1 - ((LL-length(b))/LL0);
        end       
    end
    
end


%----- LOCAL FUNCTIONS
% This is the function to minimize, sum of -log-likelihood.
% there are two functions, because of row/column vector 
%% for fminunc
function sumerr = local_negLL(beta,choice,vF,vA,pF,pA,AL,model,base)
    % estimated likelihood of chooseing the lottery
    p = choice_prob_ambigNrisk(base,vF,vA,pF,pA,AL,beta,model); 

       % Trap log(0)
    ind = p == 1;
    p(ind) = 0.9999;
    ind = p == 0;
    p(ind) = 0.0001;

    % Log-likelihood
    % If lottery is chosen, err=log(p), log likelihood of choosing the lottery
    % If reference is chosen, err=log(1-p), log likelihood of choosing the reference
    % Because 0<p<1, log(p) and log(1-p) < 0. sum of these likelihoods is negative 
    % The sum of these likelihood should be maximized
    err = (choice==1).*log(p) + (1 - (choice==1)).*log(1-p);

    % Sum of -log-likelihood. sumerr is a positive value, should me minimized
    sumerr = -sum(err);