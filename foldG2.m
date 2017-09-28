function [ g2,g2err,folded_numer,folded_denom ] = foldG2( numer,denom,max_bin)
%FOLDG2 Summary of this function goes here
%   Detailed explanation goes here
    %Fold the numerator and denominator
    folded_numer = zeros(1,max_bin+1);
    folded_denom = zeros(1,max_bin+1);
    max_pulse_distance = int32(4);
    %For tau = 0
    folded_numer(1) = numer(max_bin+1);
    folded_denom(1) = denom;
    for i=1:max_bin
        folded_numer(i+1) = numer(max_bin+1+i) + numer(max_bin+1-i);
        folded_denom(i+1) = 2*denom;
    end
    g2 = double(max_pulse_distance) * 2 * double(folded_numer)./double(folded_denom);
    g2err = double(max_pulse_distance) * 2*(folded_numer.*(folded_denom+folded_numer)./(folded_denom.^3)).^(1/2);
end

