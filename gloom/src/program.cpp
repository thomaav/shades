#include "program.hpp"
#include "gloom/gloom.hpp"
#include "gloom/shader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"


void runProgram(GLFWwindow* window)
{
    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Configure miscellaneous OpenGL settings
    glEnable(GL_MULTISAMPLE);
    // glEnable(GL_CULL_FACE);

    // Set default colour after clearing the colour buffer
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Initialize some shaders.
    Gloom::Shader *shader = new Gloom::Shader();
    shader->makeBasicShader("../gloom/shaders/simple.vert", "../gloom/shaders/simple.frag");
    shader->activate();

    // Set up the scene for the fragment shader here (essentially just
    // a quad that covers the entire screen).
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

    // Set up noise texture, exactly
    // like. https://learnopengl.com/Getting-started/Textures.
    int noise_width, noise_height, noise_n_channels;
    unsigned char *noise_texture_data = stbi_load("../gloom/textures/rgbanoise.png",
                                                  &noise_width, &noise_height, &noise_n_channels, 0);

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
    unsigned char *planet_texture_data = stbi_load("../gloom/textures/planet.jpg",
                                                   &planet_width, &planet_height, &planet_n_channels, 0);

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

    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {
        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Pass required uniforms.
        glUniform2f(0, windowWidth, windowHeight);
        glUniform1f(1, (float) glfwGetTime());

        // Make sure textures are activated.
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, noise_texture);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, planet_texture);

        // Draw calls.
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Handle other events
        glfwPollEvents();
        handleKeyboardInput(window);

        // Flip buffers
        glfwSwapBuffers(window);
    }
}


void handleKeyboardInput(GLFWwindow* window)
{
    // Use escape key for terminating the GLFW window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}
