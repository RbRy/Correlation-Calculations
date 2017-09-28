folder_name = 'C:\Data\g2_test';
file_struct = dir(sprintf('%s\\*.h5',folder_name));
for i = 1:length(file_struct)
    file_list{i} = sprintf('%s\\%s',folder_name,file_struct(i).name);
end
fileToCoincidences_g2_cw_cuda(file_list,5e-9,1e-6,100e-6, int32(4));
clear mex
exit