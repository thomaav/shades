#include "sound.hpp"
#include "fft.hpp"

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alut.h>
#include <iostream>
#include <chrono>
#include <thread>

float bass_amplitude = 60.0f;

#include <string.h>
void print_devices()
{
    ALCchar *devices = (ALCchar *) alcGetString(NULL, ALC_ALL_DEVICES_SPECIFIER);
    ALCchar *next = devices + 1;
    ALCchar *device = devices;
    size_t len = 0;

    std::cout << "Devices:" << std::endl;
    while (device && *device != '\0' && next && *next != '\0') {
        std::cout << "  " << device << std::endl;
        len = strlen(device);
        device += (len + 1);
        next += (len + 2);
    }
}

// http://www.david-amador.com/2011/06/playing-sound-using-openal/
void playWAV(const char *fp, std::future<bool> &&stop)
{
    // We need to initialize an OpenAL context to use to play
    // sound. This is very similar to what we do with OpenGL.
    ALCcontext *context;
    ALCdevice *device;

    if (!(device = alcOpenDevice("MOMENTUM M2 IEBT"))) {
        std::cout << "Could not open default audio device" << std::endl;
        exit(1);
    }

    if (!(context = alcCreateContext(device, NULL))) {
        std::cout << "Could not create OpenAL context" << std::endl;
        exit(1);
    }

    alcMakeContextCurrent(context);

    // We need to buffer data from a WAV file to an OpenAL buffer. I
    // think some of this stuff is pretty deprecated, but I just
    // wanted to play some sound.
    ALuint       *alBuffer;
    ALenum        alFormatBuffer;
    ALsizei       alFreqBuffer;
    long          alBufferLen;
    ALboolean     alLoop;
    unsigned int  alSource;
    unsigned int  alSampleSet;

    alutLoadWAVFile((ALbyte *) fp,
                    &alFormatBuffer,
                    (void **) &alBuffer,
                    (ALsizei *) &alBufferLen,
                    &alFreqBuffer,
                    &alLoop);

    alGenSources(1, &alSource);
    alGenBuffers(1, &alSampleSet);
    alBufferData(alSampleSet, alFormatBuffer, alBuffer, alBufferLen, alFreqBuffer);
    alSourcei(alSource, AL_BUFFER, alSampleSet);

    // alBuffer contains STEREO16, so convert that to left and right
    // channels.
    short *left_channel_data = new short[2*alBufferLen/2];
    short *right_channel_data = new short[2*alBufferLen/2];
    long i_right = 0; long i_left = 0;

    for (long i = 0; i < alBufferLen/2; ++i) {
        if (i % 2 == 0)
            left_channel_data[i_left++] = ((short *) alBuffer)[i];
        else
            right_channel_data[i_right++] = ((short *) alBuffer)[i];
    }

    // Play some sound.
    alSourcePlay(alSource);

    // Use where we are currently positioned within the music to
    // calculate the FFT.
    ALint offset;
    while (true) {
        alGetSourcei(alSource, AL_SAMPLE_OFFSET, &offset);
        bass_amplitude = fft_bass_amplitude(left_channel_data + offset, 4410);
    }

    // Wait until the main thread tells us to stop playing.
    stop.get();

    // Clean up.
    delete[] left_channel_data;
    delete[] right_channel_data;
    alutUnloadWAV(alFormatBuffer, alBuffer, alBufferLen, alFreqBuffer);

    alDeleteSources(1, &alSource);
    alDeleteBuffers(1, &alSampleSet);
    alcDestroyContext(context);
    alcCloseDevice(device);
}
