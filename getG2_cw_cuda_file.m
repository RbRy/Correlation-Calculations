function [ g2, tau , running_numer, running_denom] = getG2_cw_cuda_file( bin_width, max_time)
    %%For use with non-pulsed data taking

    pulse_spacing = int32(round(100e-6/bin_width));
    max_bin = int32(round(max_time/bin_width));
    max_pulse_distance = int32(4);
    %Get directory of files
    folder_name = uigetdir;
    %Get all h5 files in folder
    file_struct = dir(sprintf('%s\\*.h5',folder_name));
    filelist = cell(length(file_struct),1);
    for i = 1:length(file_struct)
        filelist{i} = sprintf('%s\\%s',folder_name,file_struct(i).name);
    end
    tic
    [running_numer,running_denom] = fileToCoincidences_g2_cw_cuda(filelist, bin_width, max_time, 100e-6, max_pulse_distance);
    toc
    g2 = double(max_pulse_distance) * 2 * double(running_numer)./double(running_denom);
    tau = [-max_time:bin_width:max_time];
    clear fileToCoincidences_g2_cw_cuda;
end

