function [ g3, tau , reshaped_numer, tot_denom] = getG3_cw_cuda_file( bin_width, max_time)
    %%For use with non-pulsed data taking
    num_gpu = 2;
    tagger_resolution = 82.3e-12;
    pulse_spacing = 100e-6;
    
    % Adjust all times to be integer multiples of the tagger resolution
    tagger_bin_width = round(bin_width/tagger_resolution) * tagger_resolution;
    tagger_max_time = round(max_time/tagger_bin_width) * tagger_bin_width;
    tagger_pulse_spacing = round(pulse_spacing/tagger_bin_width) * tagger_bin_width;
    
    max_bin = int32(round(tagger_max_time/tagger_bin_width));
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
    running_numer = cell(num_gpu,1);
    running_denom = cell(num_gpu,1);
    filelist_per_gpu = cell(num_gpu,1);
    for i = 1:num_gpu
        filelist_per_gpu{i} = filelist(i:num_gpu:end);
    end
    parfor i = 1:num_gpu
        [running_numer{i},running_denom{i}] = fileToCoincidences_g3_cw_cuda(filelist_per_gpu{i}, tagger_bin_width, tagger_max_time, tagger_pulse_spacing, max_pulse_distance,int32(i-1));
    end
    toc
    tot_numer = double(running_numer{1});
    tot_denom = double(running_denom{1});
    for i = 2:num_gpu
        tot_numer = tot_numer + double(running_numer{i});
        tot_denom = tot_denom + double(running_denom{i});
    end
    
    reshaped_numer = reshape(tot_numer,max_bin*2+1,max_bin*2+1);
    g3 = ((double(max_pulse_distance) * 2)^2 - (double(max_pulse_distance) * 2)) * double(reshaped_numer)./double(tot_denom);
    tau = [-tagger_max_time:tagger_bin_width:tagger_max_time];
    clear fileToCoincidences_g3_cw_cuda;
end

