#pragma once

#include <future>

void playWAV(const char *fp, std::future<bool> &&stop);
