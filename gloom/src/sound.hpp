#pragma once

#include <future>

extern float bass_amplitude;

void playWAV(const char *fp, std::future<bool> &&stop);
