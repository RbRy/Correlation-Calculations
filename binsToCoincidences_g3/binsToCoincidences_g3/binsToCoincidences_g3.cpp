// binsToCoincidences_g3.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "mex.h"
#include <vector>

void calculateCoincidences(std::vector<long long int*> *photon_bins, std::vector<int> *photon_bins_length, std::vector<std::vector<std::vector<long int>>> *coinc, int *max_bin, int *pulse_spacing, int *max_pulse_distance) {
	//Loop over all channel pairs
	for (int channel_1 = 0; channel_1 < photon_bins->size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < photon_bins->size(); channel_2++) {
			for (int channel_3 = channel_2 + 1; channel_3 < photon_bins->size(); channel_3++) {
				//The shifts within a pulse
				int tau_1;
				int tau_2;
				//For a given pulse this is the relative pulse you're looking at on channel 2
				int pulse_shift_2;
				//Loop over all tau steps
				#pragma omp parallel for
				for (tau_1 = -*max_bin; tau_1 <= *max_bin; tau_1++) {
					#pragma omp parallel for
					for (tau_2 = -*max_bin; tau_2 <= *max_bin; tau_2++) {
						for (pulse_shift_2 = -*max_pulse_distance; pulse_shift_2 <= *max_pulse_distance; pulse_shift_2++) {
							//For a given pulse this is the relative pulse you're looking at on channel 3. Should be different from the shift for channel 2 to ensure we're looking at uncorrelated stuff
							int pulse_shift_3;
							//Determine what the pulse shift for the third channel is
							if (pulse_shift_2 < 0) {
								pulse_shift_3 = pulse_shift_2 - 1;
							}
							else if (pulse_shift_2 > 0) {
								pulse_shift_3 = pulse_shift_2 + 1;
							}
							else {
								pulse_shift_3 = 0;
							}
							//Keep a running total of the coincidence counts
							long int running_tot = 0;
							int i = 0;
							int j = 0;
							int k = 0;
							//Loop until we hit the end of one of our vectors
							while ((i < (*photon_bins_length)[channel_1]) & (j < (*photon_bins_length)[channel_2]) & (k < (*photon_bins_length)[channel_3])) {
								//Check if the bin shift will cause an undeflow and increment till it does not
								if ((tau_1 + pulse_shift_2*(*pulse_spacing) > (*photon_bins)[channel_2][j]) & (tau_1 + pulse_shift_2*(*pulse_spacing) > 0)) {
									j++;
								}
								else if ((tau_2 + pulse_shift_3*(*pulse_spacing) > (*photon_bins)[channel_3][k]) & (tau_2 + pulse_shift_3*(*pulse_spacing)  > 0)) {
									k++;
								}
								else {
									//Abuse the fact that each vector is chronologically ordered to help find common elements quickly
									//Check if we have a coincidence
									if (((*photon_bins)[channel_1][i] == ((*photon_bins)[channel_2][j] - tau_1 - pulse_shift_2*(*pulse_spacing))) & ((*photon_bins)[channel_1][i] == ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing)))) {
										running_tot++;
										i++;
										j++;
										k++;
									}
									//Else try and figure out which vector is lagging and increment its pointer
									//First check if channel 1 is smaller (or equal) than channel 2
									// c1 <= c2
									else if ((*photon_bins)[channel_1][i] <= ((*photon_bins)[channel_2][j] - tau_1 - pulse_shift_2*(*pulse_spacing))) {
										//Then check if channel 1 is the smallest
										// c1 <= c2 & c1 < c3
										if ((*photon_bins)[channel_1][i] < ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing))) {
											i++;
										}
										//Channel 3 is smallest
										// c3 < c1 <= c2
										else if ((*photon_bins)[channel_1][i] > ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing))) {
											k++;
										}
										// Remember if c1 = c2 = c3 code doesn't reach here so we don't worry about the possibility that c1 = c2
										// c1 = c3 < c2
										else if ((*photon_bins)[channel_1][i] == ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing))) {
											i++;
											k++;
										}
									}
									// c2 < c1
									else {
										//Channel 2 is smallest
										// c2 < c1, c3
										if (((*photon_bins)[channel_2][j] - tau_1 - pulse_shift_2*(*pulse_spacing)) < ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing))) {
											j++;
										}
										//Channel 3 is smallest
										// c3 < c2 < c1
										else if (((*photon_bins)[channel_2][j] - tau_1 - pulse_shift_2*(*pulse_spacing)) > ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing))) {
											k++;
										}
										//Channel 2 and Channel 3 are equal
										// c2 = c3 < c1
										else if (((*photon_bins)[channel_2][j] - tau_1 - pulse_shift_2*(*pulse_spacing)) == ((*photon_bins)[channel_3][k] - tau_2 - pulse_shift_3*(*pulse_spacing))) {
											j++;
											k++;
										}
									}
								}
							}
							(*coinc)[pulse_shift_2 + *max_pulse_distance][tau_1 + *max_bin][tau_2 + *max_bin] += running_tot;
						}
					}
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
	plhs[0] = mxCreateNumericMatrix(*max_bin * 2 + 1, *max_bin * 2 + 1, mxINT32_CLASS, mxREAL);
	long int* numer = (long int*)mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(*max_bin * 2 + 1, *max_bin * 2 + 1, mxINT32_CLASS, mxREAL);
	long int* denom = (long int*)mxGetData(plhs[1]);
	//Initialise denom and numer to zero
	#pragma omp parallel for
	for (int i = 0; i < (*max_bin * 2 + 1); i++) {
		for (int j = 0; j < (*max_bin * 2 + 1); j++) {
			numer[i * (*max_bin * 2 + 1) + j] = 0;
			denom[i * (*max_bin * 2 + 1) + j] = 0;
		}
	}
	//Check that the bin cell array has at least three channels
	if (num_channels >= 3) {
		//And a rank three tensor that will hold the coincidence counts as we go
		std::vector<std::vector<std::vector<long int>>> coinc(*max_pulse_distance * 2 + 1, std::vector<std::vector<long int>>(*max_bin * 2 + 1, std::vector<long int>(*max_bin * 2 + 1, 0)));

		//Calculate the coincidences
		calculateCoincidences(&photon_bins, &photon_bins_length, &coinc, max_bin, pulse_spacing, max_pulse_distance);

		//Extract numerator and denominator from the coincidence matrix
		#pragma omp parallel for
		for (int i = 0; i < (*max_bin * 2 + 1); i++) {
			for (int j = 0; j < (*max_bin * 2 + 1); j++) {
				for (int k = -(*max_pulse_distance); k <= (*max_pulse_distance); k++) {
					if (k != 0) {
						//Reversed indices here due to some crazy convention I used back when i first started coding this
						denom[i * (*max_bin * 2 + 1) + j] += coinc[k + (*max_pulse_distance)][j][i];
					}
					else {
						numer[i * (*max_bin * 2 + 1) + j] = coinc[k + (*max_pulse_distance)][j][i];
					}
				}
			}
		}
	}
	else {
		mexPrintf("Only %i channels, did not perform computation\n", num_channels);
	}
}
