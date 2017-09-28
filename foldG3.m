function [ g3,g3err,folded_numer,folded_denom ] = foldG3( numer,denom,max_bin )
%FOLDG3 Summary of this function goes here
%   Detailed explanation goes here
            %Fold the numerator and denominator
            folded_numer = zeros(max_bin+1,max_bin+1);
            folded_denom = zeros(max_bin+1,max_bin+1);
            max_pulse_distance = 2;
            %For tau1,tau2 = 0
            folded_numer(1,1) = numer(max_bin+1,max_bin+1);
            folded_denom(1,1) = denom;
            %For tau1 = 0 , tau2=/=0
            for j=1:max_bin
                folded_numer(1,1+j) = numer(max_bin+1,max_bin+1+j)+numer(max_bin+1,max_bin+1-j);
                folded_denom(1,1+j) = 2*denom;
            end
            %For tau1=/= 0 , tau2=0
            for i=1:max_bin
                folded_numer(1+i,1) = numer(max_bin+1+i,max_bin+1)+numer(max_bin+1-i,max_bin+1);
                folded_denom(1+i,1) = 2*denom;
            end
            %For all other tau1 & tau2
            for i=1:max_bin
                for j=1:max_bin
                    folded_numer(1+i,1+j) = numer(max_bin+1+i,max_bin+1+j) + numer(max_bin+1+i,max_bin+1-j) + numer(max_bin+1-i,max_bin+1+j) + numer(max_bin+1-i,max_bin+1-j);
                    folded_denom(1+i,1+j) = 4*denom;
                end
            end
            g3 = ((double(max_pulse_distance) * 2)^2 - (double(max_pulse_distance) * 2))*double(folded_numer)./double(folded_denom);
            g3err = ((double(max_pulse_distance) * 2)^2 - (double(max_pulse_distance) * 2))*(folded_numer.*(folded_denom+folded_numer)./(folded_denom.^3)).^(1/2);

end

