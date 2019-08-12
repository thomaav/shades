#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.h"

static constexpr int windowWidth = 1600;
static constexpr int windowHeight = 900;
static constexpr char windowTitle[] = "shades";
static constexpr GLboolean windowResizable = GL_FALSE;
static constexpr int MSAA = 4;

static void glfwErrorCallback(int error, const char *description)
{
    fprintf(stderr, "GLFW returned an error:\n\t%s (%i)\n", description, error);
}

int main(int, char **)
{
    if (!glfwInit()) {
        fprintf(stderr, "Could not start GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwSetErrorCallback(glfwErrorCallback);

    glfwWindowHint(GLFW_RESIZABLE, windowResizable);
    glfwWindowHint(GLFW_SAMPLES, MSAA);

    GLFWwindow* window =
        glfwCreateWindow(windowWidth, windowHeight, windowTitle, nullptr, nullptr);

    if (!window) {
        fprintf(stderr, "Could not open GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    gladLoadGL();

    printf("%s: %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER));
    printf("GLFW\t %s\n", glfwGetVersionString());
    printf("OpenGL\t %s\n", glGetString(GL_VERSION));
    printf("GLSL\t %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_MULTISAMPLE);
    // glEnable(GL_CULL_FACE);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    GLuint shader = glCreateProgram();
    GLuint vShader = compileShaderFromFile("shaders/shades.vert", GL_VERTEX_SHADER);
	GLuint fShader = compileShaderFromFile("shaders/shades.frag", GL_FRAGMENT_SHADER);

    glAttachShader(shader, vShader);
	glAttachShader(shader, fShader);
	if (!linkShaderProgram(shader)) {
		return -1;
	}

    glUseProgram(shader);

    float quad[] = {
        -1.0f, +1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
        +1.0f, +1.0f, 0.0f,

        +1.0f, +1.0f, 0.0f,
        +1.0f, -1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
    };

    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    int noise_width, noise_height, noise_n_channels;
    unsigned char *noise_texture_data =
        stbi_load("textures/rgbanoise.png", &noise_width, &noise_height, &noise_n_channels, 0);

    GLuint noise_texture;
    glGenTextures(1, &noise_texture);
    glBindTexture(GL_TEXTURE_2D, noise_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, noise_width, noise_height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, noise_texture_data);

    stbi_image_free(noise_texture_data);

    // Set up texture for planet.
    int planet_width, planet_height, planet_n_channels;
    unsigned char *planet_texture_data =
        stbi_load("textures/planet.jpg", &planet_width, &planet_height, &planet_n_channels, 0);

    GLuint planet_texture;
    glGenTextures(1, &planet_texture);
    glBindTexture(GL_TEXTURE_2D, planet_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, planet_width, planet_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, planet_texture_data);

    stbi_image_free(planet_texture_data);

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniform2f(0, windowWidth, windowHeight);
        glUniform1f(1, (float) glfwGetTime());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, noise_texture);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, planet_texture);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }

        glfwSwapBuffers(window);
    }

    glfwTerminate();

    return EXIT_SUCCESS;
}
