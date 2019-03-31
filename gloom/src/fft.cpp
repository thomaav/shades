#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include <fftw3.h>

#include "fft.hpp"

double *hamming(int fft_size)
{
    double *window = new double[fft_size];

    for (int i = 0; i < fft_size; ++i) {
        window[i] = alpha - beta * cos(2.0 * M_PI * double(i) / double(fft_size - 1));
    }

    return window;
}

double bin_to_frequency(double bin, double sample_rate, double frame_size)
{
        return double(bin) * sample_rate / double(frame_size);
}

double fft_bass_amplitude(short *samples, size_t nsamples)
{
    double bass_amplitude = 0;
    int bass_bins = 3;

    double *in;
    fftw_complex *out;
    double bins[num_bins];
    fftw_plan plan;
    double *hamming_window = hamming(fft_size);

    in = fftw_alloc_real(fft_size);
    out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * fft_size);
    plan = fftw_plan_dft_r2c_1d(fft_size, in, out, FFTW_ESTIMATE);

    for (int x = 0; x < nsamples / step_size; ++x) {
        // Process all samples of one FFT period, i.e. period of
        // fft_size amount of samples, windows will overlap 50% by
        // using step_size as fft_size / 2.
        for (int i = 0, j = x * step_size; j < x * step_size + fft_size; ++i, ++j)
            in[i] = samples[j] * hamming_window[i];

        fftw_execute(plan);

        // Process values into values relevant to create spectrogram,
        // only the first fft_size / 2 bins are useful.
        for (int i = 0; i < num_bins; ++i) {
            bins[i] = 10.0 * log10(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
            // bins[i] = fmax(0, bins[i]);
        }

        for (size_t i = 0; i < bass_bins; ++i) {
            bass_amplitude += bins[i];
        }
    }

    free(hamming_window);
    fftw_free(in); fftw_free(out);
    fftw_destroy_plan(plan);

    return bass_amplitude / (bass_bins * nsamples / step_size);
}
