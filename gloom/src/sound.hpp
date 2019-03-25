#pragma once

#include <future>

#define HEADSET "MOMENTUM M2 IEBT"
#define SPEAKER "Built-in Audio Analog Stereo"
#define DEFAULT NULL

extern bool enable_audio;
extern float bass_amplitude;

void playWAV(const char *fp, std::future<bool> &&stop);
