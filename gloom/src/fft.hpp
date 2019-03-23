#pragma once

#include <vector>

// Hamming window.
constexpr double alpha = 0.54;
constexpr double beta  = 0.46;

constexpr int fft_size = 4096;
constexpr int num_bins = fft_size / 2;
constexpr int step_size = fft_size / 2; // 50% window overlap

double *hamming(int fft_size);
double bin_to_frequency(double bin, double sample_rate, double frame_size);
double fft_bass_amplitude(short *samples, size_t nsamples);
