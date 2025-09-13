import math
import time
import pygame
from pydub import AudioSegment
import sys
import numpy as np
from pygame.locals import *
from OpenGL.GL import *

# --- CONFIGURATION ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
aspect_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
AUDIO_FILE = "file_example_WAV_1MG.wav"
#TODO: make audiofile an output file from elevenlabs


# --- SHADERS --- dunno what this does
particle_vertex = """
#version 330 core
uniform float uNoiseOffset;
uniform float uAspectRatio;
uniform float uAmplitude;
uniform int uSeed;
uniform int uDotCount;
uniform float uDotRadius;
uniform float uDotRadiusPX;
uniform float uDotSpacing;
uniform float uDotOffset;
uniform float uSphereRadius;
uniform float uFeather;
uniform float uNoiseFrequency;
uniform float uNoiseAmplitude;
layout(location = 0) in vec2 inPosition;
out vec2 fragUV;
out float fragDotRadiusPX;

// --- START: Complete Perlin Noise Implementation ---
const float GAIN = 0.5;
const float LACUNARITY = 1.5;
const float FRACTAL_BOUNDING = 1.0 / 1.75;
const ivec3 PRIMES = ivec3(501125321, 1136930381, 1720413743);
const float GRADIENTS_3D[] = float[](
   0., 1., 1., 0.,  0.,-1., 1., 0.,  0., 1.,-1., 0.,  0.,-1.,-1., 0.,
   1., 0., 1., 0., -1., 0., 1., 0.,  1., 0.,-1., 0., -1., 0.,-1., 0.,
   1., 1., 0., 0., -1., 1., 0., 0.,  1.,-1., 0., 0., -1.,-1., 0., 0.,
   0., 1., 1., 0.,  0.,-1., 1., 0.,  0., 1.,-1., 0.,  0.,-1.,-1., 0.,
   1., 0., 1., 0., -1., 0., 1., 0.,  1., 0.,-1., 0., -1., 0.,-1., 0.,
   1., 1., 0., 0., -1., 1., 0., 0.,  1.,-1., 0., 0., -1.,-1., 0., 0.,
   0., 1., 1., 0.,  0.,-1., 1., 0.,  0., 1.,-1., 0.,  0.,-1.,-1., 0.,
   1., 0., 1., 0., -1., 0., 1., 0.,  1., 0.,-1., 0., -1., 0.,-1., 0.,
   1., 1., 0., 0., -1., 1., 0., 0.,  1.,-1., 0., 0., -1.,-1., 0., 0.,
   0., 1., 1., 0.,  0.,-1., 1., 0.,  0., 1.,-1., 0.,  0.,-1.,-1., 0.,
   1., 0., 1., 0., -1., 0., 1., 0.,  1., 0.,-1., 0., -1., 0.,-1., 0.,
   1., 1., 0., 0., -1., 1., 0., 0.,  1.,-1., 0., 0., -1.,-1., 0., 0.,
   0., 1., 1., 0.,  0.,-1., 1., 0.,  0., 1.,-1., 0.,  0.,-1.,-1., 0.,
   1., 0., 1., 0., -1., 0., 1., 0.,  1., 0.,-1., 0., -1., 0.,-1., 0.,
   1., 1., 0., 0., -1., 1., 0., 0.,  1.,-1., 0., 0., -1.,-1., 0., 0.,
   1., 1., 0., 0.,  0.,-1., 1., 0., -1., 1., 0., 0.,  0.,-1.,-1., 0.
);
float smootherStep(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
int hash(int seed, ivec3 primed) { return (seed ^ primed.x ^ primed.y ^ primed.z) * 0x27d4eb2d; }
float gradCoord(int seed, ivec3 primed, vec3 d) {
   int hash_val = hash(seed, primed);
   hash_val ^= hash_val >> 15;
   hash_val &= 63 << 2;
   return d.x * GRADIENTS_3D[hash_val] + d.y * GRADIENTS_3D[hash_val | 1] + d.z * GRADIENTS_3D[hash_val | 2];
}
float perlinSingle(int seed, vec3 coord) {
   ivec3 coord0 = ivec3(floor(coord));
   vec3 d0 = coord - vec3(coord0);
   vec3 d1 = d0 - 1.0;
   vec3 s = vec3(smootherStep(d0.x), smootherStep(d0.y), smootherStep(d0.z));
   coord0 *= PRIMES;
   ivec3 coord1 = coord0 + PRIMES;
   float xf00 = mix(gradCoord(seed, coord0, d0), gradCoord(seed, ivec3(coord1.x, coord0.yz), vec3(d1.x, d0.yz)), s.x);
   float xf10 = mix(gradCoord(seed, ivec3(coord0.x, coord1.y, coord0.z), vec3(d0.x, d1.y, d0.z)), gradCoord(seed, ivec3(coord1.xy, coord0.z), vec3(d1.xy, d0.z)), s.x);
   float xf01 = mix(gradCoord(seed, ivec3(coord0.xy, coord1.z), vec3(d0.xy, d1.z)), gradCoord(seed, ivec3(coord1.x, coord0.y, coord1.z), vec3(d1.x, d0.y, d1.z)), s.x);
   float xf11 = mix(gradCoord(seed, ivec3(coord0.x, coord1.yz), vec3(d0.x, d1.yz)), gradCoord(seed, coord1, d1), s.x);
   float yf0 = mix(xf00, xf10, s.y);
   float yf1 = mix(xf01, xf11, s.y);
   return mix(yf0, yf1, s.z) * 0.964921414852142333984375f;
}
float fractalNoise(vec3 coord) {
   return perlinSingle(uSeed, coord) * FRACTAL_BOUNDING
       + perlinSingle(uSeed + 1, coord * LACUNARITY) * FRACTAL_BOUNDING * GAIN
       + perlinSingle(uSeed + 2, coord * LACUNARITY * LACUNARITY) * FRACTAL_BOUNDING * GAIN * GAIN;
}
// --- END: Complete Perlin Noise Implementation ---

void main() {
   vec2 dotPos = vec2(float(gl_InstanceID % uDotCount), float(gl_InstanceID / uDotCount));
   float noise = fractalNoise(vec3(dotPos * uNoiseFrequency, uNoiseOffset)) * uNoiseAmplitude;
   vec3 dotCenter = vec3(dotPos * uDotSpacing + uDotOffset + noise, (noise + 0.5 * uNoiseAmplitude) * uAmplitude * 0.4);
   float distanceFromCenter = length(dotCenter);
   if (distanceFromCenter > 0.0) {
        dotCenter /= distanceFromCenter;
   }
   distanceFromCenter = min(uSphereRadius, distanceFromCenter);
   dotCenter *= distanceFromCenter;
   float featherRadius = uSphereRadius - uFeather;
   float featherStrength = 1.0 - clamp((distanceFromCenter - featherRadius) / uFeather, 0.0, 1.0);
   if (distanceFromCenter > 0.0) {
        dotCenter *= featherStrength * (uSphereRadius / distanceFromCenter - 1.0) + 1.0;
   }
   dotCenter.y *= -1.0;
   vec2 finalPos = dotCenter.xy + inPosition * uDotRadius * (1.0 + 1.0 / uDotRadiusPX);
   finalPos.x /= uAspectRatio;
   gl_Position = vec4(finalPos, 0.0, 1.0);
   fragUV = inPosition;
   fragDotRadiusPX = uDotRadiusPX + 1.0;
}
"""
particle_fragment = """
#version 330 core
in vec2 fragUV;
in float fragDotRadiusPX;
out float outColor;
void main() {
   float t = clamp((1.0 - length(fragUV)) * fragDotRadiusPX, 0.0, 1.0);
   outColor = t;
}
"""
blur_vertex = """
#version 330 core
uniform float uBlurRadius;
uniform vec2 uBlurDirection;
layout(location = 0) in vec2 inPosition;
out vec2 fragUV;
flat out vec2 fragBlurDirection;
flat out int fragSupport;
flat out vec3 fragGaussCoefficients;

float calculateGaussianTotal(int support, vec3 fragGaussCoefficientsIn) {
   float total = fragGaussCoefficientsIn.x;
   vec3 tempCoefficients = fragGaussCoefficientsIn;
   for (int i = 1; i < support; i++) {
       tempCoefficients.xy *= tempCoefficients.yz;
       total += 2.0 * tempCoefficients.x;
   }
   return total;
}

void main() {
   fragSupport = int(ceil(1.5 * uBlurRadius)) * 2;
   fragGaussCoefficients = vec3(1.0 / (sqrt(2.0 * 3.14159265) * uBlurRadius), exp(-0.5 / (uBlurRadius * uBlurRadius)), 0.0);
   fragGaussCoefficients.z = fragGaussCoefficients.y * fragGaussCoefficients.y;
   fragGaussCoefficients.x /= calculateGaussianTotal(fragSupport, fragGaussCoefficients);
   gl_Position = vec4(inPosition, 0.0, 1.0);
   fragUV = (inPosition + 1.0) / 2.0;
   fragBlurDirection = uBlurDirection;
}
"""
blur_fragment = """
#version 330 core
uniform sampler2D uInputTexture;
in vec2 fragUV;
flat in vec2 fragBlurDirection;
flat in int fragSupport;
flat in vec3 fragGaussCoefficients;
out float outColor;

void main() {
   vec3 gaussCoefficients = fragGaussCoefficients;
   outColor = gaussCoefficients.x * texture(uInputTexture, fragUV).r;
   for (int i = 1; i < fragSupport; i += 2) {
       gaussCoefficients.xy *= gaussCoefficients.yz;
       float coefficientSum = gaussCoefficients.x;
       gaussCoefficients.xy *= gaussCoefficients.yz;
       coefficientSum += gaussCoefficients.x;
       float pixelRatio = gaussCoefficients.x / coefficientSum;
       vec2 offset = (float(i) + pixelRatio) * fragBlurDirection;
       outColor += coefficientSum * (texture(uInputTexture, fragUV + offset).r + texture(uInputTexture, fragUV - offset).r);
   }
}
"""
finalize_vertex = """
#version 330 core
uniform vec3 uOutputColor;
layout(location = 0) in vec2 inPosition;
out vec2 fragUV;
out vec3 fragOutputColor;
void main() {
   gl_Position = vec4(inPosition, 0.0, 1.0);
   fragUV = (inPosition + 1.0) / 2.0;
   fragOutputColor = uOutputColor;
}
"""
finalize_fragment = """
#version 330 core
uniform sampler2D uBlurredTexture;
uniform sampler2D uOriginalTexture;
in vec2 fragUV;
in vec3 fragOutputColor;
out vec4 outColor;
void main() {
   float value = max(texture(uBlurredTexture, fragUV).r, texture(uOriginalTexture, fragUV).r);
   outColor = vec4(fragOutputColor * value, 1.0);
}
"""


# --- HELPER FUNCTIONS ---
def create_shader_program(vertex, fragment):
    program = glCreateProgram()
    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, vertex)
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        print("Vertex Shader Error:")
        print(glGetShaderInfoLog(vs).decode())
        return None
    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, fragment)
    glCompileShader(fs)
    if not glGetShaderiv(fs, GL_COMPILE_STATUS):
        print("Fragment Shader Error:")
        print(glGetShaderInfoLog(fs).decode())
        return None
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    return program


def create_fbo_with_texture(width, height):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Framebuffer is not complete!")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, texture


def audio_analyze(audio_path):
    print("Analyzing audio...")
    song = AudioSegment.from_wav(audio_path)
    magnitude_lists = []
    x_y_z_form = []
    chunk_size = 50
    # Reduce the number of samples
    for i in range(0, len(song), chunk_size):
        samples = song[i:i + chunk_size].get_array_of_samples()
        if len(samples) > 0:
            # rms loudness measuring ts
            rms_val = np.sqrt(np.mean(np.square(samples, dtype=np.float64)))
            x_y_z_form.append(rms_val)
    print("Analysis complete.")
    return np.array(x_y_z_form, dtype=np.float32)


def linear_interpolation(remap, old_min, old_max, new_min, new_max):
    if old_max - old_min == 0:
        return new_min
    return (remap - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


# --- MAIN
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL, vsync=1)
    pygame.display.set_caption("J.A.R.V.I.S.")
    CLOCK = pygame.time.Clock()
    
    # --- COMPILE SHADERS ---
    particle_program = create_shader_program(particle_vertex, particle_fragment)
    blur_program = create_shader_program(blur_vertex, blur_fragment)
    finalize_program = create_shader_program(finalize_vertex, finalize_fragment)
    if not all([particle_program, blur_program, finalize_program]):
        print("Shader compilation failed. Exiting.")
        return
    
    # --- CREATE OPENGL OBJECTS ---
    particle_fbo, particle_texture = create_fbo_with_texture(SCREEN_WIDTH, SCREEN_HEIGHT)
    blur_x_fbo, blur_x_texture = create_fbo_with_texture(SCREEN_WIDTH, SCREEN_HEIGHT)
    blur_y_fbo, blur_y_texture = create_fbo_with_texture(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    quad_vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
    quad_vao = glGenVertexArrays(1)
    glBindVertexArray(quad_vao)
    quad_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glBindVertexArray(0)
    
    # --- GET UNIFORM LOCATIONS ---
    p_uniforms = {name: glGetUniformLocation(particle_program, name) for name in
                  ["uAspectRatio", "uNoiseOffset", "uAmplitude", "uSeed", "uDotCount", "uDotRadius", "uDotRadiusPX",
                   "uDotSpacing",
                   "uDotOffset", "uSphereRadius", "uFeather", "uNoiseFrequency", "uNoiseAmplitude"]}
    b_uniforms = {name: glGetUniformLocation(blur_program, name) for name in
                  ["uBlurRadius", "uBlurDirection", "uInputTexture"]}
    f_uniforms = {name: glGetUniformLocation(finalize_program, name) for name in
                  ["uOutputColor", "uBlurredTexture", "uOriginalTexture"]}
    
    # --- LOAD DATA & INITIALIZE VARIABLES ---
    audio_data = audio_analyze(AUDIO_FILE)
    max_amplitude = np.max(audio_data) if len(audio_data) > 0 else 1.0
    
    list_index = 0
    last_update_time = pygame.time.get_ticks()
    update_interval = 50  # updates per second
    
    # --- MAIN LOOP ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- UPDATE LOGIC
        current_time = pygame.time.get_ticks()
        if current_time - last_update_time >= update_interval:
            if list_index < len(audio_data):
                # Remap audio data to useful ranges
                normalized_amp = audio_data[list_index] / max_amplitude
                sphere_radius = linear_interpolation(normalized_amp, 0, 1, 0.7, 0.8)
                feather = linear_interpolation(normalized_amp, 0, 1, 0.05, 0.2)
                amplitude = linear_interpolation(normalized_amp, 0, 1, 0.5, 1.5)
                
                glBindVertexArray(quad_vao)
                
                # PASS 1\
                glBindFramebuffer(GL_FRAMEBUFFER, particle_fbo)
                glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
                glClear(GL_COLOR_BUFFER_BIT)
                glUseProgram(particle_program)
                glUniform1f(p_uniforms["uAspectRatio"], aspect_ratio)
                glUniform1f(p_uniforms["uNoiseOffset"], list_index * 0.02)
                glUniform1f(p_uniforms["uAmplitude"], amplitude)
                glUniform1i(p_uniforms["uSeed"], 12345)
                glUniform1i(p_uniforms["uDotCount"], 322)
                glUniform1f(p_uniforms["uDotRadius"], 0.9 / 322)
                glUniform1f(p_uniforms["uDotRadiusPX"], (0.9 / 322 * 0.5 * min(SCREEN_WIDTH, SCREEN_HEIGHT)))
                glUniform1f(p_uniforms["uDotSpacing"], 0.9 / 321)
                glUniform1f(p_uniforms["uDotOffset"], -0.45)
                glUniform1f(p_uniforms["uSphereRadius"], sphere_radius)
                glUniform1f(p_uniforms["uFeather"], feather)
                glUniform1f(p_uniforms["uNoiseFrequency"], 4 / 322)
                glUniform1f(p_uniforms["uNoiseAmplitude"], 0.32 * 0.9)
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 322 * 322)
                
                # PASS 2 & 3
                blur_radius = linear_interpolation(normalized_amp, 0, 1, 0.0, 5.0)
                # Horizontal Blur
                glBindFramebuffer(GL_FRAMEBUFFER, blur_x_fbo)
                glClear(GL_COLOR_BUFFER_BIT)
                glUseProgram(blur_program)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, particle_texture)
                glUniform1f(b_uniforms["uBlurRadius"], blur_radius)
                glUniform2f(b_uniforms["uBlurDirection"], 1.0 / SCREEN_WIDTH, 0.0)
                glUniform1i(b_uniforms["uInputTexture"], 0)
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
                # Vertical Blur
                glBindFramebuffer(GL_FRAMEBUFFER, blur_y_fbo)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindTexture(GL_TEXTURE_2D, blur_x_texture)
                glUniform2f(b_uniforms["uBlurDirection"], 0.0, 1.0 / SCREEN_HEIGHT)
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
                
                list_index += 1
            last_update_time = current_time
        
        # --- RENDER LOGIC
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(finalize_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, blur_y_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, particle_texture)
        glUniform3f(f_uniforms["uOutputColor"], 0.0, 0.6, 1.0)
        glUniform1i(f_uniforms["uBlurredTexture"], 0)
        glUniform1i(f_uniforms["uOriginalTexture"], 1)
        
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        pygame.display.flip()
        CLOCK.tick(60)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()