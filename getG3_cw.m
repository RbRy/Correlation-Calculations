function [ g3, tau , running_numer, running_denom] = getG3_cw( bin_width, max_time)
    %%For use with non-pulsed data taking    

    pulse_spacing = int32(round(100e-6/bin_width));
    max_bin = int32(round(max_time/bin_width));
    max_pulse_distance = int32(2);
    %Get directory of files
    folder_name = uigetdir;
    %Get all h5 files in folder
    file_struct = dir(sprintf('%s\\*.h5',folder_name));
    running_numer = int32(zeros(max_bin*2+1,max_bin*2+1));
    running_denom = int32(0);
    tic
    parfor i = 1:length(file_struct)
        filename = sprintf('%s\\%s',folder_name,file_struct(i).name);
        try
            [tags,clocks] = readCorrelationTags_cw(filename);
            bins = tagsToBins(tags,bin_width);
            clock_bins = int64(ceil(double(clocks)*82.3e-12/bin_width));
            [numer,denom] = binsToCoincidences_g3_cw(bins,max_bin,pulse_spacing,max_pulse_distance,clock_bins);
            running_numer = running_numer + numer;
            running_denom = running_denom + denom;
        catch
            disp(filename);
        end
        %if rem(i,100) == 0
        %    disp(double(i)/double(length(file_struct))*100)
        %end
    end
    toc
    g3 = ((double(max_pulse_distance) * 2)^2 - (double(max_pulse_distance) * 2)) * double(running_numer)./double(running_denom);
    tau = [-max_time:bin_width:max_time];
    
end