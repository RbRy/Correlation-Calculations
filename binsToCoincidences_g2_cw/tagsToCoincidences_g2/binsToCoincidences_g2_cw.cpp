// tagsToCoincidences_g2.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "mex.h"
#include <vector>
#include <omp.h>

void calculateCoincidences(std::vector<long long int*> *photon_bins, std::vector<int> *photon_bins_length, long int *numer, long int *denom, int *max_bin, int *pulse_spacing, int *max_pulse_distance, long long int *start_and_end_clocks) {
	//Loop over all channel pairs
	for (int channel_1 = 0; channel_1 < photon_bins->size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < photon_bins->size(); channel_2++) {
			int tau;
			int pulse_shift;
			//Loop over all tau steps for the numerator
			#pragma omp parallel for
			for (tau = -*max_bin; tau <= *max_bin; tau++){
				//Keep a running total of the coincidence counts
				long int running_tot = 0;
				int i = 0;
				int j = 0;
				//Calculate the various tau steps for the numerator
				//Loop until we hit the end of one of our vectors
				while ((i < (*photon_bins_length)[channel_1]) & (j < (*photon_bins_length)[channel_2])) {
					//Check if the bin shift will cause an undeflow and increment till it does not
					if ((tau >(*photon_bins)[channel_2][j]) & (tau > 0)) {
						j++;
					}
					//Ensure we don't look at channel 1 photons in a window a given time after it has started and before it has ended to prevent edge effects
					else if (((*photon_bins)[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0])) || ((*photon_bins)[channel_1][i] > (start_and_end_clocks[1] - (*max_bin + *max_pulse_distance * *pulse_spacing)))) {
						//printf("%i\t%i\t%i\n", (*photon_bins)[channel_1][i], (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0]), (start_and_end_clocks[1] - (*max_bin + *max_pulse_distance * *pulse_spacing)));
						i++;
					}
					else {
						//Abuse the fact that each vector is chronologically ordered to help find common elements quickly
						//If chan_1 > chan_2
						if ((*photon_bins)[channel_1][i] > ((*photon_bins)[channel_2][j] - tau)) {
							j++;
						}
						//If chan_2 > chan_1
						else if ((*photon_bins)[channel_1][i] < ((*photon_bins)[channel_2][j] - tau)) {
							i++;
						}
						//If there is a common elements increment coincidence counts
						else if ((*photon_bins)[channel_1][i] == ((*photon_bins)[channel_2][j] - tau)) {
							//See if there are duplicate elements that hold extra coincidences
							int duplicate_chan_1 = 1;
							int duplicate_chan_2 = 1;
							int dummy_i = 1;
							int dummy_j = 1;
							bool looking_for_duplciates = true;
							//First check for duplicates on channel 1
							while (looking_for_duplciates) {
								if ((*photon_bins)[channel_1][i] == (*photon_bins)[channel_1][i + dummy_i]) {
									duplicate_chan_1++;
									dummy_i++;
								}
								else {
									looking_for_duplciates = false;
								}
							}
							//Then on channel 2
							looking_for_duplciates = true;
							while (looking_for_duplciates) {
								if ((*photon_bins)[channel_2][j] == (*photon_bins)[channel_2][j + dummy_j]) {
									duplicate_chan_2++;
									dummy_j++;
								}
								else {
									looking_for_duplciates = false;
								}
							}
							//Increment the running tot by the number of combined coincidences due to duplicates
							running_tot += duplicate_chan_1 * duplicate_chan_2;
							//Shift i & j past the duplicates
							i += dummy_i;
							j += dummy_j;
						}
					}
				}
				numer[tau + *max_bin] += running_tot;
			}
			//Now do the denominator calculations
			long int running_denom = 0;
			for (pulse_shift = -*max_pulse_distance; pulse_shift <= *max_pulse_distance; pulse_shift++) {
				if (pulse_shift != 0) {
					int i = 0;
					int j = 0;
					//Loop until we hit the end of one of our vectors
					while ((i < (*photon_bins_length)[channel_1]) & (j < (*photon_bins_length)[channel_2])) {
						//Check if the bin shift will cause an undeflow and increment till it does not
						if ((pulse_shift*(*pulse_spacing) > (*photon_bins)[channel_2][j]) & (pulse_shift*(*pulse_spacing) > 0)) {
							j++;
						}
						else if (((*photon_bins)[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0])) || ((*photon_bins)[channel_1][i] > (start_and_end_clocks[1] - (*max_bin + *max_pulse_distance * *pulse_spacing)))) {
							//printf("%i\t%i\t%i\n", (*photon_bins)[channel_1][i], (*max_bin + *max_pulse_distance * *pulse_spacing + start_and_end_clocks[0]), (start_and_end_clocks[1] - (*max_bin + *max_pulse_distance * *pulse_spacing)));
							i++;
						}
						else {
							//Abuse the fact that each vector is chronologically ordered to help find common elements quickly
							if ((*photon_bins)[channel_1][i] > ((*photon_bins)[channel_2][j] - pulse_shift*(*pulse_spacing))) {
								j++;
							}
							else if ((*photon_bins)[channel_1][i] < ((*photon_bins)[channel_2][j] - pulse_shift*(*pulse_spacing))) {
								i++;
							}
							//If there is a common elements increment coincidence counts
							else if ((*photon_bins)[channel_1][i] == ((*photon_bins)[channel_2][j] - pulse_shift*(*pulse_spacing))) {
								//See if there are duplicate elements that hold extra coincidences
								int duplicate_chan_1 = 1;
								int duplicate_chan_2 = 1;
								int dummy_i = 1;
								int dummy_j = 1;
								bool looking_for_duplciates = true;
								//First check for duplicates on channel 1
								while (looking_for_duplciates) {
									if ((*photon_bins)[channel_1][i] == (*photon_bins)[channel_1][i + dummy_i]) {
										duplicate_chan_1++;
										dummy_i++;
									}
									else {
										looking_for_duplciates = false;
									}
								}
								//Then on channel 2
								looking_for_duplciates = true;
								while (looking_for_duplciates) {
									if ((*photon_bins)[channel_2][j] == (*photon_bins)[channel_2][j + dummy_j]) {
										duplicate_chan_2++;
										dummy_j++;
									}
									else {
										looking_for_duplciates = false;
									}
								}
								//Increment the running tot by the number of combined coincidences due to duplicates
								running_denom += duplicate_chan_1 * duplicate_chan_2;
								//Shift i & j past the duplicates
								i += dummy_i;
								j += dummy_j;
							}
						}
					}
				}
			}
			denom[0] += running_denom;
		}
	}
}

void mexFunction(int nlhs, mxArray* plhs[], int nrgs, const mxArray* prhs[]) {
	mxArray *cell_element_ptr;
	mwSize vector_length, num_channels;
	
	//Figure out how many channels we're getting passed data for
	num_channels = mxGetNumberOfElements(prhs[0]);
	
	//Create a vector of pointers to store our bins
	std::vector<long long int*> photon_bins(num_channels);
	
	//And a vector to hold the length of the vectors for each channel
	std::vector<int> photon_bins_length(num_channels);
	
	//Populate the vector
	for (int i = 0; i < num_channels; i++) {
		cell_element_ptr = mxGetCell(prhs[0], i);
		vector_length = mxGetNumberOfElements(cell_element_ptr);
		photon_bins[i] = (long long int *)mxGetData(cell_element_ptr);
		photon_bins_length[i] = mxGetNumberOfElements(cell_element_ptr);
	}
	
	//Get the max bin (time) we want to look at correlations
	int *max_bin;
	max_bin = (int *)mxGetData(prhs[1]);
	
	//Get the pulse spacing in bins (used for calculating the denominator)
	int *pulse_spacing;
	pulse_spacing = (int *)mxGetData(prhs[2]);
	
	//Find out how many pulses away we want to include in denominator calculation
	int *max_pulse_distance;
	max_pulse_distance = (int *)mxGetData(prhs[3]);

	long long int *start_and_end_clocks = (long long int *)mxGetData(prhs[4]);
	
	//Create our array to hold the denominator and numerator
	plhs[0] = mxCreateNumericMatrix(1, *max_bin * 2 + 1, mxINT32_CLASS, mxREAL);
	long int* numer = (long int*)mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
	long int* denom = (long int*)mxGetData(plhs[1]);
	//Initialise denom and numer to zero
	#pragma omp parallel for
	for (int i = 0; i < (*max_bin * 2 + 1); i++) {
		numer[i] = 0;
	}
	denom[0] = 0;
	//Check that we have at least 2 channels
	if (num_channels >= 2) {
		
		//Calculate coincidences
		calculateCoincidences(&photon_bins, &photon_bins_length, numer, denom, max_bin, pulse_spacing, max_pulse_distance, start_and_end_clocks);

	}
	else {
		mexPrintf("Only %i channels, did not perform computation\n", num_channels);
	}
}
