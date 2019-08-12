#include <stdio.h>

#include "shader.h"

GLuint compileShaderFromFile(std::string file, GLenum type) {
    FILE *fp = fopen(file.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buffer = new char[length];
    size_t err;

    err = fread(buffer, sizeof(char), length, fp);
    if ((long)err < length) {
        perror("read shader");
    }

    GLuint shader = compileShader(buffer, length, type);
    delete[] buffer;

    return shader;
}

GLuint compileShader(char *shaderCode, size_t codeLength, GLenum type) {
    GLuint shader;
    shader = glCreateShader(type);

    GLint shaderCodeLength = codeLength;
    glShaderSource(shader, 1, (const GLchar **)&shaderCode, &shaderCodeLength);

    glCompileShader(shader);

    GLint mStatus;

    glGetShaderiv(shader, GL_COMPILE_STATUS, &mStatus);

    if (mStatus == GL_FALSE) {
        GLint len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);

        if (len > 0) {
            GLchar *str = new GLchar[len];
            glGetShaderInfoLog(shader, len, NULL, str);
            fprintf(stderr, "Failed to compile shader: %s\n", str);
            delete[] str;
        }
        else {
            fprintf(stderr, "Failed to compile shader\n");
        }
    }

    return shader;
}

bool linkShaderProgram(GLuint program) {
    GLint link_status;

    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, (int *)&link_status);
    if (link_status == GL_FALSE) {
        GLint log_length;

        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

        char *log = (char*)calloc(log_length+1, 1);
        glGetProgramInfoLog(program, log_length, &log_length, log);

        fprintf(stderr, "shader: %.*s\n", (int)log_length, log);
    }

    if (link_status == GL_FALSE) {
        glDeleteProgram(program);
        program = 0;
    }

    return !!program;
}

