function [ g3, tau , reshaped_numer, running_denom] = getG3_cw_cuda_file( bin_width, max_time)
    %%For use with non-pulsed data taking

    pulse_spacing = int32(round(100e-6/bin_width));
    max_bin = int32(round(max_time/bin_width));
    max_pulse_distance = int32(2);
    %Get directory of files
    folder_name = uigetdir;
    %Get all h5 files in folder
    file_struct = dir(sprintf('%s\\*.h5',folder_name));
    filelist = cell(length(file_struct),1);
    for i = 1:length(file_struct)
        filelist{i} = sprintf('%s\\%s',folder_name,file_struct(i).name);
    end
    tic
    [running_numer,running_denom] = fileToCoincidences_g3_cw_cuda(filelist, bin_width, max_time, 100e-6, max_pulse_distance);
    toc
    reshaped_numer = reshape(running_numer,max_bin*2+1,max_bin*2+1);
    g3 = ((double(max_pulse_distance) * 2)^2 - (double(max_pulse_distance) * 2)) * double(reshaped_numer)./double(running_denom);
    tau = [-max_time:bin_width:max_time];
    clear fileToCoincidences_g3_cw_cuda;
end

