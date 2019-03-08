// Local headers
#include "program.hpp"
#include "gloom/gloom.hpp"
#include "gloom/shader.hpp"


void runProgram(GLFWwindow* window)
{
    // Enable depth (Z) buffer (accept "closest" fragment)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Configure miscellaneous OpenGL settings
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

    // Rendering Loop
    while (!glfwWindowShouldClose(window))
    {
        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Pass required uniforms.
        glUniform2f(0, windowWidth, windowHeight);
        glUniform1f(1, (float) glfwGetTime());

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
