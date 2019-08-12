#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <glad/glad.h>

GLuint compileShaderFromFile(std::string file, GLenum type);
GLuint compileShader(char *shaderCode, size_t codeLength, GLenum type);
bool linkShaderProgram(GLuint program);

#endif // SHADER_H
