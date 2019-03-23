#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alut.h>
#include <iostream>

#include "sound.hpp"

// http://www.david-amador.com/2011/06/playing-sound-using-openal/
void playWAV(const char *fp)
{
    // We need to initialize an OpenAL context to use to play
    // sound. This is very similar to what we do with OpenGL.
    ALCcontext *context;
    ALCdevice *device;

    if (!(device = alcOpenDevice(NULL))) {
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

    alutUnloadWAV(alFormatBuffer, alBuffer, alBufferLen, alFreqBuffer);

    // Play some sound.
    alSourcePlay(alSource);

    // Check if we are still playing.
    ALenum state;
    while (true) {
        alGetSourcei(alSource, AL_SOURCE_STATE, &state);
        if (state != AL_PLAYING) break;
    }

    // Clean up.
    alDeleteSources(1, &alSource);
    alDeleteBuffers(1, &alSampleSet);
    alcDestroyContext(context);
    alcCloseDevice(device);
}
