bin_width = 5e-9;
max_time = 50e-6;
max_pulse_distance = int32(4);
folder_name = 'C:\Data\g2_test';
file_struct = dir(sprintf('%s\\*.h5',folder_name));
filelist = cell(length(file_struct),1);
for i = 1:length(file_struct)
    filelist{i} = sprintf('%s\\%s',folder_name,file_struct(i).name);
end
fileToCoincidences_g2_cw_cuda(filelist, bin_width, max_time, 100e-6, max_pulse_distance);
clear mex;
exit