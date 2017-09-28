// tagsToCoincidences_g2.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "mex.h"
#include <vector>
#include <omp.h>

void calculateCoincidences(std::vector<long long int*> *photon_bins, std::vector<int> *photon_bins_length, std::vector<std::vector<long int>> *coinc, int *max_bin, int *pulse_spacing, int *max_pulse_distance) {
	//Loop over all channel pairs
	for (int channel_1 = 0; channel_1 < photon_bins->size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < photon_bins->size(); channel_2++) {
			int tau;
			int pulse_shift;
			//Loop over all tau steps
			#pragma omp parallel for
			for (tau = -*max_bin; tau <= *max_bin; tau++){
				for (pulse_shift = -*max_pulse_distance; pulse_shift <= *max_pulse_distance; pulse_shift++) {
					//Keep a running total of the coincidence counts
					long int running_tot = 0;
					int i = 0;
					int j = 0;
					//Loop until we hit the end of one of our vectors
					while ((i < (*photon_bins_length)[channel_1]) & (j < (*photon_bins_length)[channel_2])) {
						//Check if the bin shift will cause an undeflow and increment till it does not
						if ((tau + pulse_shift*(*pulse_spacing) > (*photon_bins)[channel_2][j]) & (tau + pulse_shift*(*pulse_spacing) > 0)) {
							j++;
						}
						else {
							//Abuse the fact that each vector is chronologically ordered to help find common elements quickly
							if ((*photon_bins)[channel_1][i] > ((*photon_bins)[channel_2][j] - tau - pulse_shift*(*pulse_spacing))) {
								j++;
							}
							else if ((*photon_bins)[channel_1][i] < ((*photon_bins)[channel_2][j] - tau - pulse_shift*(*pulse_spacing))) {
								i++;
							}
							//If there is a common elements increment coincidence counts
							else if ((*photon_bins)[channel_1][i] == ((*photon_bins)[channel_2][j] - tau - pulse_shift*(*pulse_spacing))) {
								running_tot++;
								i++;
								j++;
							}
						}
					}
					(*coinc)[pulse_shift + *max_pulse_distance][tau + *max_bin] += running_tot;
				}
			}
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
	
	//Create our array to hold the denominator and numerator
	plhs[0] = mxCreateNumericMatrix(1, *max_bin * 2 + 1, mxINT32_CLASS, mxREAL);
	long int* numer = (long int*)mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1, *max_bin * 2 + 1, mxINT32_CLASS, mxREAL);
	long int* denom = (long int*)mxGetData(plhs[1]);
	//Initialise denom and numer to zero
	#pragma omp parallel for
	for (int i = 0; i < (*max_bin * 2 + 1); i++) {
		numer[i] = 0;
		denom[i] = 0;
	}

	//Check that we have at least 2 channels
	if (num_channels >= 2) {
		//And a matrix that will hold the coincidence counts as we go
		std::vector<std::vector<long int>> coinc(max_pulse_distance * 2 + 1, std::vector<long int>(*max_bin * 2 + 1, 0));

		//Calculate coincidences
		calculateCoincidences(&photon_bins, &photon_bins_length, &coinc, max_bin, pulse_spacing, max_pulse_distance);

		//Extract numerator and denominator from the coincidence matrix
		#pragma omp parallel for
		for (int i = 0; i < (max_bin * 2 + 1); i++) {
			for (int j = -(max_pulse_distance); j <= (max_pulse_distance); j++) {
				if (j != 0) {
					denom[i] += coinc[j + (max_pulse_distance)][i];
				}
				else {
					numer[i] = coinc[j + (max_pulse_distance)][i];
				}
			}
		}
	}
	else {
		mexPrintf("Only %i channels, did not perform computation\n", num_channels);
	}
}
