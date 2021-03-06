
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "H5Cpp.h"
#include <vector>
#include <H5Exception.h>
#include <map>
#include <omp.h>

// The following lines must be located BEFORE '#include <mex.h>'
#ifdef _MSC_VER
#define DLL_EXPORT_SYM __declspec(dllexport)
#else
#define DLL_EXPORT_SYM
#endif
#include <mex.h>

const int max_tags_length = 200000;
const int max_clock_tags_length = 5000;
const int max_channels = 3;
const size_t return_size = 3;
const int file_block_size = 64;
const double tagger_resolution = 82.3e-12;

struct shotData {
	bool file_load_completed;
	std::vector<short int> channel_list;
	std::map<short int, short int> channel_map;
	std::vector<long long int> start_tags;
	std::vector<long long int> end_tags;
	std::vector<long long int> photon_tags;
	std::vector<long long int> clock_tags;
	std::vector<std::vector<long long int>> sorted_photon_tags;
	std::vector<std::vector<long int>> sorted_photon_bins;
	std::vector<std::vector<long long int>> sorted_clock_tags;
	std::vector<std::vector<long int>> sorted_clock_bins;
	std::vector<long int> sorted_photon_tag_pointers;
	std::vector<long int> sorted_clock_tag_pointers;

	shotData() : sorted_photon_tags(max_channels, std::vector<long long int>(max_tags_length, 0)), sorted_photon_bins(max_channels, std::vector<long int>(max_tags_length, 0)), sorted_photon_tag_pointers(max_channels, 0), sorted_clock_tags(2, std::vector<long long int>(max_clock_tags_length, 0)), sorted_clock_bins(2, std::vector<long int>(max_clock_tags_length, 0)), sorted_clock_tag_pointers(2, 0) {}
};

struct gpuData {
	long int *coinc_gpu;
	long int *photon_bins_gpu;
	long int *start_and_end_clocks_gpu;
	int *max_bin_gpu, *pulse_spacing_gpu, *max_pulse_distance_gpu, *photon_bins_length_gpu;
	int *offset_gpu;
};

__global__ void calculateCoincidencesGPU_g2(long int *coinc, long int *photon_bins, long int *start_and_end_clocks, int *max_bin, int *pulse_spacing, int *max_pulse_distance, int *offset, int *photon_bins_length, int num_channels, int shot_file_num) {
	//Get numerator step to work on
	int id = threadIdx.x;
	int block = blockIdx.x;
	int block_size = blockDim.x;

	//Check we're not calculating something out of range
	if (block * block_size + id < ((*max_bin * 2 + 1) + (*max_pulse_distance * 2))) {
		int pulse_shift_measurment = (block * block_size + id >= *max_bin * 2 + 1) && (block * block_size + id < *max_bin * 2 + 1 + (*max_pulse_distance * 2));
		int pulse_shift = ((block * block_size + id - (*max_bin * 2 + 1) - (*max_pulse_distance)) + ((block * block_size + id - (*max_bin * 2 + 1) - (*max_pulse_distance)) >= 0)) * (pulse_shift_measurment);
		int tau = (block * block_size + id - (*max_bin)) * (!pulse_shift_measurment);
		tau += pulse_shift * (*pulse_spacing);
		for (int channel_1 = 0; channel_1 < num_channels; channel_1++) {
			for (int channel_2 = channel_1 + 1; channel_2 < num_channels; channel_2++) {
				int i = 0;
				int j = 0;
				int running_tot = 0;
				while ((i < photon_bins_length[channel_1 + shot_file_num * max_channels]) && (j < photon_bins_length[channel_2 + shot_file_num * max_channels])) {
					int dummy_i = 0;
					int dummy_j = 0;
					int out_window = 0;
					//Increment i if we're outside the window of interest
					out_window = (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0 + shot_file_num * 2])) || (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] > (start_and_end_clocks[1 + shot_file_num * 2] - (*max_bin + *max_pulse_distance * *pulse_spacing)));
					//in_window = (photon_bins[offset[channel_1] + i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0]));
					dummy_i += out_window;
					//Increment i if chan_1 < chan_2
					dummy_i += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] < (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - tau));
					//Increment j if chan_1 > chan_2
					dummy_j += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] > (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - tau));
					//If we have a common element increment the running_total and increment both i and j
					dummy_i += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] == (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - tau));
					dummy_j += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] == (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - tau));
					running_tot += !out_window && photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] == (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - tau);
					//running_tot += in_window;
					i += dummy_i;
					j += dummy_j;
				}
				coinc[block * block_size + id + shot_file_num * ((*max_bin * 2 + 1) + (*max_pulse_distance * 2))] += running_tot;
			}
		}
	}
}

__global__ void calculateDenominatorGPU_g2(long int *denom, long int *photon_bins, long int *start_and_end_clocks, int *max_bin, int *pulse_spacing, int *max_pulse_distance, int *offset, int *photon_bins_length, int num_channels, int shot_file_num) {
	//Get denominator step to work on
	int id = threadIdx.x;
	int block = blockIdx.x;
	int block_size = blockDim.x;

	if (block * block_size + id < (*max_pulse_distance * 2 + 1)) {
		//Determine the pulse_shift to work on
		int pulse_shift = block * block_size + id - (*max_pulse_distance);
		if (pulse_shift != 0) {
			for (int channel_1 = 0; channel_1 < num_channels; channel_1++) {
				for (int channel_2 = channel_1 + 1; channel_2 < num_channels; channel_2++) {
					int i = 0;
					int j = 0;
					int running_tot = 0;
					while ((i < photon_bins_length[channel_1 + shot_file_num * max_channels]) && (j < photon_bins_length[channel_2 + shot_file_num * max_channels])) {
						int dummy_i = 0;
						int dummy_j = 0;
						int out_window = 0;
						//Increment i if we're outside the window of interest
						out_window = (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0 + shot_file_num * 2])) || (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] > (start_and_end_clocks[1 + shot_file_num * 2] - (*max_bin + *max_pulse_distance * *pulse_spacing)));
						//in_window = (photon_bins[offset[channel_1] + i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0]));
						dummy_i += out_window;
						//Increment i if chan_1 < chan_2
						dummy_i += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] < (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - pulse_shift*(*pulse_spacing)));
						//Increment j if chan_1 > chan_2
						dummy_j += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] > (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - pulse_shift*(*pulse_spacing)));
						//If we have a common element increment the running_total and increment both i and j
						dummy_i += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] == (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - pulse_shift*(*pulse_spacing)));
						dummy_j += !out_window && (photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] == (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - pulse_shift*(*pulse_spacing)));
						running_tot += !out_window && photon_bins[offset[channel_1 + shot_file_num * max_channels] + i] == (photon_bins[offset[channel_2 + shot_file_num * max_channels] + j] - pulse_shift*(*pulse_spacing));
						//running_tot += in_window;
						i += dummy_i;
						j += dummy_j;
					}
					denom[block * block_size + id + shot_file_num * (*max_pulse_distance * 2 + 1)] += running_tot;
				}
			}
		}
	}
}

//Function grabs all tags and channel list from file
void fileToShotData(shotData *shot_data, char* filename) {
	//Open up file
	H5::H5File file(filename, H5F_ACC_RDONLY);
	//Open up "Tags" group
	H5::Group tag_group(file.openGroup("Tags"));
	//Find out how many tag sets there are, should be 4 if not something is fucky
	hsize_t numTagsSets = tag_group.getNumObjs();
	if (numTagsSets != 4) {
		mexPrintf("There should be 4 sets of Tags, found %i\n", numTagsSets);
		delete filename;
		exit;
	}
	//Read tags to shotData structure
	//First the clock tags
	H5::DataSet clock_dset(tag_group.openDataSet("ClockTags0"));
	H5::DataSpace clock_dspace = clock_dset.getSpace();
	hsize_t clock_length[1];
	clock_dspace.getSimpleExtentDims(clock_length, NULL);
	shot_data->clock_tags.resize(clock_length[0]);
	clock_dset.read(&(*shot_data).clock_tags[0u], H5::PredType::NATIVE_UINT64, clock_dspace);
	clock_dspace.close();
	clock_dset.close();
	//Then start tags
	H5::DataSet start_dset(tag_group.openDataSet("StartTag"));
	H5::DataSpace start_dspace = start_dset.getSpace();
	hsize_t start_length[1];
	start_dspace.getSimpleExtentDims(start_length, NULL);
	shot_data->start_tags.resize(start_length[0]);
	start_dset.read(&(*shot_data).start_tags[0u], H5::PredType::NATIVE_UINT64, start_dspace);
	start_dspace.close();
	start_dset.close();
	//Then end tags
	H5::DataSet end_dset(tag_group.openDataSet("EndTag"));
	H5::DataSpace end_dspace = end_dset.getSpace();
	hsize_t end_length[1];
	end_dspace.getSimpleExtentDims(end_length, NULL);
	shot_data->end_tags.resize(end_length[0]);
	end_dset.read(&(*shot_data).end_tags[0u], H5::PredType::NATIVE_UINT64, end_dspace);
	end_dspace.close();
	end_dset.close();
	//Finally photon tags
	H5::DataSet photon_dset(tag_group.openDataSet("TagWindow0"));
	H5::DataSpace photon_dspace = photon_dset.getSpace();
	hsize_t photon_length[1];
	photon_dspace.getSimpleExtentDims(photon_length, NULL);
	shot_data->photon_tags.resize(photon_length[0]);
	photon_dset.read(&(*shot_data).photon_tags[0u], H5::PredType::NATIVE_UINT64, photon_dspace);
	photon_dspace.close();
	photon_dset.close();
	//And close tags group
	tag_group.close();
	//Open up "Inform" group
	H5::Group inform_group(file.openGroup("Inform"));
	//Grab channel list
	H5::DataSet chan_dset(inform_group.openDataSet("ChannelList"));
	H5::DataSpace chan_dspace = chan_dset.getSpace();
	hsize_t chan_length[1];
	chan_dspace.getSimpleExtentDims(chan_length, NULL);
	shot_data->channel_list.resize(chan_length[0]);
	chan_dset.read(&(*shot_data).channel_list[0u], H5::PredType::NATIVE_UINT16, chan_dspace);
	chan_dspace.close();
	chan_dset.close();
	//Close Inform group
	inform_group.close();
	//Close file
	file.close();

	//Populate channel map
	for (short int i = 0; i < shot_data->channel_list.size(); i++) {
		shot_data->channel_map[shot_data->channel_list[i]] = i;
	}
}

//Reads relevant information for a block of files into shot_block
void populateBlock(std::vector<shotData> *shot_block, std::vector<char *> *filelist, int block_num) {
	//Loop over the block size
	for (int i = 0; i < file_block_size; i++) {
		//Default to assuming the block is corrupted
		(*shot_block)[i].file_load_completed = false;
		//Figure out the file id within the filelist
		int file_id = block_num * file_block_size + i;
		//Check the file_id isn't out of range of the filelist
		if (file_id < filelist->size()) {
			//Try to load file to shot_block
			try {
				fileToShotData(&(*shot_block)[i], (*filelist)[file_id]);
				(*shot_block)[i].file_load_completed = true;
			}
			//Will catch if the file is corrupted, print corrupted filenames to command window
			catch (...) {
				printf("%s appears corrupted\n", (*filelist)[file_id]);
			}
		}
	}
}


//Process the time tags, assigning them to the correct channel, binning them appropriately and removing tags which do not fall in the clock mask
void sortTags(shotData *shot_data) {
	long int i;
	int high_count = 0;
	//Loop over all tags in clock_tags
	for (i = 0; i < shot_data->clock_tags.size(); i++) {
		//Check if clock tag is a high word
		if (shot_data->clock_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Determine whether it is the rising (start) or falling (end) slope
			int slope = ((shot_data->clock_tags[i] >> 28) & 1);
			//Put tag in appropriate clock tag vector and increment the pointer for said vector
			shot_data->sorted_clock_tags[slope][shot_data->sorted_clock_tag_pointers[slope]] = ((shot_data->clock_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			shot_data->sorted_clock_tag_pointers[slope]++;
		}
	}
	high_count = 0;
	//Clock pointer
	int clock_pointer = 0;
	//Loop over all tags in photon_tags
	for (i = 0; i < shot_data->photon_tags.size(); i++) {
		//Check if photon tag is a high word
		if (shot_data->photon_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Figure out if it fits within the mask
			long long int time_tag = ((shot_data->photon_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			bool valid = true;
			while (valid) {
				//printf("%i\t%i\t%i\t", time_tag, shot_data->sorted_clock_tags[1][clock_pointer], shot_data->sorted_clock_tags[0][clock_pointer - 1]);
				//Increment dummy pointer if channel tag is greater than current start tag
				if ((time_tag >= shot_data->sorted_clock_tags[1][clock_pointer]) & (clock_pointer < shot_data->sorted_clock_tag_pointers[1])) {
					//printf("up clock pointer\n");
					clock_pointer++;
				}
				//Make sure clock_pointer is greater than 0, preventing an underflow error
				else if (clock_pointer > 0) {
					//Check if tag is lower than previous end tag i.e. startTags[j-1] < channeltags[i] < endTags[j-1]
					if (time_tag <= shot_data->sorted_clock_tags[0][clock_pointer - 1]) {
						//printf("add tag tot data\n");
						//Determine the index for given tag
						int channel_index; 
						//Bin tag and assign to appropriate vector
						channel_index = shot_data->channel_map.find(((shot_data->photon_tags[i] >> 29) & 7) + 1)->second;
						shot_data->sorted_photon_tags[channel_index][shot_data->sorted_photon_tag_pointers[channel_index]] = time_tag;
						shot_data->sorted_photon_tag_pointers[channel_index]++;
						//printf("%i\t%i\t%i\n", channel_index, time_tag, shot_data->sorted_photon_tag_pointers[channel_index]);
					}
					//Break the valid loop
					valid = false;
				}
				// If tag is smaller than the first start tag
				else {
					valid = false;
				}
			}
		}
	}
}

void tagsToBins(shotData *shot_data, double bin_width) {
	int tagger_bins_per_bin_width = (int)round(bin_width / tagger_resolution);
	#pragma omp parallel for
	for (int channel = 0; channel < shot_data->sorted_photon_bins.size(); channel++) {
		#pragma omp parallel for
		for (int i = 0; i < shot_data->sorted_photon_tag_pointers[channel]; i++) {
			shot_data->sorted_photon_bins[channel][i] = shot_data->sorted_photon_tags[channel][i] / tagger_bins_per_bin_width;
		}
	}
	for (int slope = 0; slope <= 1; slope++) {
		#pragma omp parallel for
		for (int i = 0; i < shot_data->sorted_clock_tag_pointers[slope]; i++) {
			shot_data->sorted_clock_bins[slope][i] = shot_data->sorted_clock_tags[slope][i] / tagger_bins_per_bin_width;
		}
	}
}

//Sorts photons and bins them for each file in a block
void sortAndBinBlock(std::vector<shotData> *shot_block, double bin_width) {
	#pragma omp parallel for
	for (int shot_file_num = 0; shot_file_num < file_block_size; shot_file_num++) {
		if ((*shot_block)[shot_file_num].file_load_completed) {
			sortTags(&(*shot_block)[shot_file_num]);
			tagsToBins(&(*shot_block)[shot_file_num], bin_width);
		}
	}
}

void printShotChannelBins(shotData *shot_data, int channel) {
	for (int i = 0; i < shot_data->sorted_photon_tag_pointers[channel]; i++) {
		printf("%i\t%i\t%i\n",i, shot_data->sorted_photon_tags[channel][i], shot_data->sorted_photon_bins[channel][i]);
	}
}

void mexFunction(int nlhs, mxArray* plhs[], int nrgs, const mxArray* prhs[]) {
	//Get list of files to process
	mxArray *cell_element_ptr;
	mwSize total_num_files, buflen;
	//Figure out how many files there are and allocate a vector to hold strings
	total_num_files = mxGetNumberOfElements(prhs[0]);
	std::vector<char *> filelist(total_num_files);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < total_num_files; i++) {
		cell_element_ptr = mxGetCell(prhs[0], i);
		buflen = mxGetN(cell_element_ptr) * sizeof(mxChar) + 1;
		filelist[i] = (char *)mxMalloc(buflen);
		mxGetString(cell_element_ptr, filelist[i], buflen);
	}

	double *max_time;
	max_time = (double *)mxGetData(prhs[2]);
	double *bin_width;
	bin_width = (double *)mxGetData(prhs[1]);
	double *pulse_spacing;
	pulse_spacing = (double *)mxGetData(prhs[3]);
	int *max_pulse_distance;
	max_pulse_distance = (int *)mxGetData(prhs[4]);
	int *cuda_device_number;
	cuda_device_number = (int *)mxGetData(prhs[5]);

	printf("Bin width\t%f�s\t%fns\t%f�s\t%i\n", *max_time * 1e6, *bin_width * 1e9, *pulse_spacing * 1e6, *max_pulse_distance);

	int max_bin = (int)round(*max_time / *bin_width);
	int bin_pulse_spacing = (int)round(*pulse_spacing / *bin_width);

	//Create our array to hold the denominator and numerator
	plhs[0] = mxCreateNumericMatrix(1, ((2 * (max_bin)+1)), mxINT32_CLASS, mxREAL);
	long int* numer = (long int*)mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
	long int* denom = (long int*)mxGetData(plhs[1]);
	//Initialise denom and numer to zero
	#pragma omp parallel for
	for (int i = 0; i < (max_bin * 2 + 1); i++) {
		numer[i] = 0;
	}
	denom[0] = 0;

	//Figure out how many blocks we need
	int blocks_req;
	if (total_num_files < file_block_size) {
		blocks_req = 1;
	}
	else if ((total_num_files%file_block_size) == 0) {
		blocks_req = total_num_files / file_block_size;
	}
	else {
		blocks_req = total_num_files / file_block_size + 1;
	}
	printf("Processing %i files in %i blocks\n", total_num_files, blocks_req);

	cudaError_t cudaStatus = cudaSetDevice(*cuda_device_number);
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Pointers for our various pinned memory for host-GPU DMA
	long int* pinned_photon_bins;
	long int* pinned_start_and_end_clocks;
	int* pinned_photon_bins_length;
	cudaMallocHost((long int**)&pinned_photon_bins, max_tags_length * max_channels * file_block_size * sizeof(long int));
	cudaMallocHost((long int**)&pinned_start_and_end_clocks, 2 * file_block_size * sizeof(long int));
	cudaMallocHost((int**)&pinned_photon_bins_length, max_channels * file_block_size * sizeof(int));

	//Load some stuff to the GPU we will use permenantly
	//Allocate memory on GPU for various things
	gpuData gpu_data;

	cudaStatus = cudaMalloc((void**)&(gpu_data.photon_bins_gpu), max_channels * max_tags_length * file_block_size * sizeof(long int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc photon_bins_gpu failed\n");
		mexPrintf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.offset_gpu), max_channels * file_block_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc offset_gpu failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.photon_bins_length_gpu), max_channels * file_block_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc photon_bins_length_gpu failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.coinc_gpu), ((2 * (max_bin) + 1) + (*max_pulse_distance * 2)) * file_block_size * sizeof(long int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc numer_gpu failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.start_and_end_clocks_gpu), 2 * file_block_size * sizeof(long int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc start_and_end_clocks_gpu failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.max_bin_gpu), sizeof(int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc max_bin_gpu failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.pulse_spacing_gpu), sizeof(int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc pulse_spacing_gpu failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&(gpu_data.max_pulse_distance_gpu), sizeof(int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMalloc max_pulse_distance_gpu failed!\n");
		goto Error;
	}

	//And set some values that are constant across all data
	cudaStatus = cudaMemcpy((gpu_data.max_bin_gpu), &max_bin, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy((gpu_data.pulse_spacing_gpu), &bin_pulse_spacing, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy((gpu_data.max_pulse_distance_gpu), max_pulse_distance, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMemcpy failed!\n");
		goto Error;
	}

	//Pointer to first photon bin for each channel
	int host_offest_array[max_channels * file_block_size];
	for (int i = 0; i < max_channels * file_block_size; i++) {
		host_offest_array[i] = i * max_tags_length;
	}
	cudaStatus = cudaMemcpy((gpu_data.offset_gpu), host_offest_array, max_channels * file_block_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMemcpy failed!\n");
		goto Error;
	}

	//Set numerator and denominator to 0
	cudaStatus = cudaMemset((gpu_data).coinc_gpu, 0, ((2 * (max_bin)+1) + (*max_pulse_distance * 2)) * file_block_size * sizeof(long int));
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMemset failed!\n");
		goto Error;
	}

	//Create some streams for us to use for GPU parallelism
	cudaStream_t streams[file_block_size];
	for (int i = 0; i < file_block_size; i++) {
		cudaStreamCreate(&streams[i]);
	}

	//Create some events to allow us to know if previous transfer has completed
	cudaEvent_t events[file_block_size];
	for (int i = 0; i < file_block_size; i++) {
		cudaEventCreate(&events[i]);
	}

	//Figure out how many blocks to chunk the processing up into
	//For the numerator
	int threads_per_cuda_block_numer = 128;
	int cuda_blocks_numer = 0;
	if (threads_per_cuda_block_numer >= (max_bin * 2 + 1) + (*max_pulse_distance * 2)) {
		cuda_blocks_numer = 1;
	}
	else if (((max_bin * 2 + 1) % threads_per_cuda_block_numer) == 0) {
		cuda_blocks_numer = ((max_bin * 2 + 1) + (*max_pulse_distance * 2)) / threads_per_cuda_block_numer;
	}
	else {
		cuda_blocks_numer = ((max_bin * 2 + 1) + (*max_pulse_distance * 2)) / threads_per_cuda_block_numer + 1;
	}

	//And the denominator
	int threads_per_cuda_block_denom = (*max_pulse_distance * 2 + 1);
	int cuda_blocks_denom = 1;

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(file_block_size);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, *bin_width);
		//printShotChannelBins(&(shot_block[0]), 1);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		/*cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			mexPrintf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}*/

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			mexPrintf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		//Asyncronously load data to GPU
		for (int shot_file_num = 0; shot_file_num < file_block_size; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				int num_channels = (shot_block)[shot_file_num].channel_list.size();
				if (num_channels >= 2) {

					std::vector<long int*> photon_bins;
					long int start_and_end_clocks[2];
					std::vector<int> photon_bins_length;
					photon_bins.resize(max_channels);
					photon_bins_length.resize(max_channels);

					start_and_end_clocks[0] = (shot_block)[shot_file_num].sorted_clock_bins[1][0];
					start_and_end_clocks[1] = (shot_block)[shot_file_num].sorted_clock_bins[0][0];
					for (int i = 0; i < num_channels; i++) {
						photon_bins[i] = &((shot_block)[shot_file_num].sorted_photon_bins[i][0]);
						photon_bins_length[i] = (shot_block)[shot_file_num].sorted_photon_tag_pointers[i];
					}

					//Synch to ensure previous asnyc memcopy has finished otherwise we'll start overwriting writing to data that may be DMA'd
					cudaEventSynchronize(events[shot_file_num]);

					//Write photon bins to memory
					int photon_offset = shot_file_num * max_channels * max_tags_length;
					for (int i = 0; i < photon_bins_length.size(); i++) {
						memcpy(pinned_photon_bins + photon_offset, (photon_bins)[i], (photon_bins_length)[i] * sizeof(long int));
						cudaStatus = cudaMemcpyAsync((gpu_data).photon_bins_gpu + photon_offset, pinned_photon_bins + photon_offset, (photon_bins_length)[i] * sizeof(long int), cudaMemcpyHostToDevice, streams[shot_file_num]);
						if (cudaStatus != cudaSuccess) {
							mexPrintf("cudaMemcpy photon_bins failed!\n");
							goto Error;
						}
						photon_offset += max_tags_length;
					}

					int clock_offset = shot_file_num * 2;
					//And other parameters
					memcpy(pinned_start_and_end_clocks + clock_offset, start_and_end_clocks, 2 * sizeof(long int));
					cudaStatus = cudaMemcpyAsync((gpu_data).start_and_end_clocks_gpu + clock_offset, pinned_start_and_end_clocks + clock_offset, 2 * sizeof(long int), cudaMemcpyHostToDevice, streams[shot_file_num]);
					if (cudaStatus != cudaSuccess) {
						mexPrintf("cudaMemcpy clock_offset failed!\n");
						goto Error;
					}

					int length_offset = shot_file_num * max_channels;
					//Can't copy vector to cuda easily
					for (int i = 0; i < photon_bins_length.size(); i++) {
						memcpy(pinned_photon_bins_length + i + length_offset, &((photon_bins_length)[i]), sizeof(int));
					}
					cudaStatus = cudaMemcpyAsync((gpu_data).photon_bins_length_gpu + length_offset, pinned_photon_bins_length + length_offset, max_channels * sizeof(int), cudaMemcpyHostToDevice, streams[shot_file_num]);
					if (cudaStatus != cudaSuccess) {
						mexPrintf("cudaMemcpy length_offset failed!\n");
						goto Error;
					}

					//Create an event to let us know all the async copies have occured
					cudaEventRecord(events[shot_file_num], streams[shot_file_num]);

					//Run kernels
					calculateCoincidencesGPU_g2 << <cuda_blocks_numer, threads_per_cuda_block_numer, 0, streams[shot_file_num] >> >((gpu_data).coinc_gpu, (gpu_data).photon_bins_gpu, (gpu_data).start_and_end_clocks_gpu, (gpu_data).max_bin_gpu, (gpu_data).pulse_spacing_gpu, (gpu_data).max_pulse_distance_gpu, (gpu_data).offset_gpu, (gpu_data).photon_bins_length_gpu, num_channels, shot_file_num);
				}
			}
		}
		mexPrintf("Finished block %i/%i\n", block_num + 1, blocks_req);
		mexEvalString("pause(.0001);");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//Free pinned memory
	cudaFreeHost(pinned_photon_bins);
	cudaFreeHost(pinned_photon_bins_length);
	cudaFreeHost(pinned_start_and_end_clocks);

	//This is to pull the streamed numerator off the GPU
	//Streamed numerator refers to the way the numerator is stored on the GPU where each GPU stream has a seperate numerator
	long int *streamed_coinc;
	streamed_coinc = (long int *)malloc(((2 * (max_bin)+1) + (*max_pulse_distance * 2)) * file_block_size * sizeof(long int));

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(streamed_coinc, (gpu_data).coinc_gpu, ((2 * (max_bin)+1) + (*max_pulse_distance * 2)) * file_block_size * sizeof(long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaMemcpy numerator failed!\n");
		free(streamed_coinc);
		goto Error;
	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < file_block_size; i++) {
		for (int j = 0; j < ((2 * (max_bin)+1) + (*max_pulse_distance * 2)); j++) {
			if (j < (2 * (max_bin)+1)) {
				numer[j] += streamed_coinc[j + i * ((2 * (max_bin)+1) + (*max_pulse_distance * 2))];
			}
			else {
				denom[0] += streamed_coinc[j + i * ((2 * (max_bin)+1) + (*max_pulse_distance * 2))];
			}
		}
	}
	free(streamed_coinc);

	//Free filenames we malloc'd earlier
	for (int i = 0; i < total_num_files; i++) {
		mxFree(filelist[i]);
	}

	//Release CUDA device
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		mexPrintf("cudaDeviceReset failed!\n");
	}

Error:
	cudaFree((gpu_data.coinc_gpu));
	cudaFree((gpu_data.offset_gpu));
	cudaFree((gpu_data.max_bin_gpu));
	cudaFree((gpu_data.pulse_spacing_gpu));
	cudaFree((gpu_data.max_pulse_distance_gpu));
	cudaFree((gpu_data.photon_bins_length_gpu));
	cudaFree(gpu_data.photon_bins_gpu);
	cudaFree(gpu_data.start_and_end_clocks_gpu);
	cudaFreeHost(pinned_photon_bins);
	cudaFreeHost(pinned_photon_bins_length);
	cudaFreeHost(pinned_start_and_end_clocks);
	cudaDeviceReset();
}