%% History
% Ruonan 10.23.2017: Add financial utility model


function y = ambig_utility(base,v,p,AL,alpha,beta,model)

if (strcmp(model,'ambiguity') || strcmp(model,'ambigNrisk')) || strcmp(model,'ambigNriskFixSlope')
    % the model we are using
    y = -((p - beta .* (AL./2)) .* v .^alpha + (1-p - beta .* (AL./2)) .* base .^alpha);
elseif strcmp(model,'ambigPower')
    y = p .^ (1+beta.*AL) .* v .^alpha; % change that
elseif strcmp(model,'discounting')
    %y = v ./ (1 + alpha.*log(1+(1-p+beta.*AL./2)./(p-beta.*AL./2)));
    y = v ./ (1 + alpha.*(1-p+beta.*AL./2)./(p-beta.*AL./2));
    %y = v ./ (1 + alpha.*(1-p)./p);
elseif strcmp(model,'ambigSubjRate')
    y = p .^ (1+beta.*AL) .* v;
elseif strcmp(model,'riskAmbigPremiumVar')
    y = v .* p - alpha .* v .^ 2 .*p .* (1 - p) - beta .* AL .^ 2; %sv = ev - risk premium on risk - risk premium on ambig
elseif strcmp(model,'riskAmbigPremiumStd')
    y = v .* p - alpha .* v .* sqrt(p .* (1 - p)) - beta .* AL; %sv = ev - risk premium on risk - risk premium on ambig
elseif strcmp(model,'riskPremium')
    y = v .* p - alpha .* v .^ 2 .*(p-beta.*AL./2) .* (1 - p + beta.*AL./2); %sv = ev - risk premium, assume a single guess of probability
elseif strcmp(model,'riskAmbigPremium')
    y = v .* p - alpha .* p .* (1 - p) - beta .* AL .^ 2; %sv = ev - risk premium on risk - risk premium on ambig, but no quadratic term of V
elseif strcmp(model, '')
    % another model, distribution of risk? 
end
end


