#version 430 core

out vec4 color;

uniform layout(location=0) vec2 window_size;
uniform layout(location=1) float time;

const int MARCHSTEPS = 250;
const float MIN_DIST = 0.0f;
const float MAX_DIST = 100.0f;
const float EPSILON = 0.005;
const float AA = 2.0f;

float sd_plane(vec3 point)
{
    return point.y;
}

float sd_sphere(vec3 point, float r)
{
    return length(point) - r;
}

float sd_box(vec3 point, vec3 b, float r)
{
    vec3 d = abs(point) - b;
    return length(max(d, 0.0f)) - r;
}

float sd_intersect(float sda, float sdb)
{
    return max(sda, sdb);
}

float sd_union(float sd_a, float sd_b)
{
    return min(sd_a, sd_b);
}

float sd_smooth_union(float sda, float sdb, float k)
{
    float h = clamp(0.5f + 0.5f*(sdb-sda)/k, 0.0f, 1.0f);
    return mix(sdb, sda, h) - k*h*(1.0f-h);
}

float sd_difference(float sda, float sdb)
{
    return max(sda, -sdb);
}

vec3 sd_translate(vec3 point, vec3 translation)
{
    return point - translation;
}

vec3 sd_rotate(vec3 point, float theta)
{
    return (mat4(
        vec4(cos(theta), 0.0f, sin(theta), 0.0f),
        vec4(0.0f, 1.0f, 0.0f, 0.0f),
        vec4(-sin(theta), 0.0f, cos(theta), 0.0f),
        vec4(0.0f, 0.0f, 0.0f, 1.0f)
    ) * vec4(point, 1.0f)).xyz;
}

float sd_scene(vec3 point)
{
    float plane = sd_plane(sd_translate(point, vec3(0.0f, -0.5f, 0.0f)));
    float box = sd_box(point, vec3(0.5f, 0.5f, 0.5f), 0.01);
    float sphere = sd_sphere(sd_translate(point, vec3(0.0f, 0.5f, 0.0f)), 0.45f);
    float sphere2 = sd_sphere(sd_translate(point, vec3(0.0f, 1.1f, 0.0f)), 0.45f);

    float scene = sd_smooth_union(box, sphere, 0.05f);
    scene = sd_smooth_union(sphere2, scene, 0.05f);
    return sd_union(plane, scene);
}

float ray_march(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene(eye + depth * dir);

        if (dist < EPSILON)
            return depth;

        depth += dist;
        if (depth >= end)
            return end;
    }

    return end;
}

vec3 ray_direction(float fov, vec2 size, vec2 fc)
{
    vec2 xy = fc - size / 2.0f;
    float z = size.y / tan(radians(fov) / 2.0f);
    return normalize(vec3(xy, -z));
}

vec3 estimate_normal(vec3 p)
{
    const float EPSILON = 0.001;
    return normalize(vec3(
          sd_scene(vec3(p.x + EPSILON, p.y, p.z)) - sd_scene(vec3(p.x - EPSILON, p.y, p.z)),
          sd_scene(vec3(p.x, p.y + EPSILON, p.z)) - sd_scene(vec3(p.x, p.y - EPSILON, p.z)),
          sd_scene(vec3(p.x, p.y, p.z + EPSILON)) - sd_scene(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

float penumbra_shadow(vec3 ro, vec3 rd, float mint, float maxt, float k)
{
    float EPSILON = 0.001;
    float res = 1.0f;

    for (float t = mint; t < maxt;) {
        float h = sd_scene(ro + rd*t);
        if (h < EPSILON)
            return 0.0f;
        res = min(res, k*h/t);
        t += h;
    }

    return res;
}

// https://en.wikipedia.org/wiki/Phong_reflection_model#Description
vec3 phong_light_contrib(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                         vec3 light_pos, vec3 light_intensity)
{
    vec3 N = estimate_normal(p);
    vec3 L = normalize(light_pos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));

    float dot_LN = dot(L, N);
    float dot_RV = dot(R, V);

    if (dot_LN < 0.0f)
        return vec3(0.0f, 0.0f, 0.0f);

    if (dot_RV < 0.0f)
        return light_intensity * (k_d * dot_LN);

    float shadow = penumbra_shadow(p, normalize(light_pos - p), 0.01f, 100.0f, 32.0f);
    return light_intensity * (k_d * dot_LN * shadow + k_s * pow(dot_RV, alpha));
}

vec3 phong_illumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha,
                        vec3 p, vec3 eye)
{
    vec3 ambient_light = 0.3 * vec3(1.0f, 1.0f, 1.0f);
    vec3 tmp_color = k_a * ambient_light;

    vec3 light_pos = vec3(10.0f, 8.0f, -10.0f);
    vec3 light_intensity = vec3(0.7f, 0.7f, 0.7f);
    tmp_color += phong_light_contrib(k_d, k_s, alpha, p, eye,
                                     light_pos, light_intensity);

    return tmp_color;
}

mat4 camera(vec3 eye, vec3 center, vec3 up)
{
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0f),
        vec4(u, 0.0f),
        vec4(-f, 0.0f),
        vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

void main()
{
    vec3 dir = ray_direction(45.0f, window_size, gl_FragCoord.xy);
    vec3 eye = vec3(2.0f, 3.0f, 10.0f);
    mat4 camera_mat = camera(eye, vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    dir = (camera_mat * vec4(dir, 1.0f)).xyz;
    float dist = ray_march(eye, dir, MIN_DIST, MAX_DIST);

    if (dist > MAX_DIST - EPSILON) {
        color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    vec3 p = eye + dist * dir;
    vec3 k_a = vec3(0.5f, 0.5f, 0.5f);
    vec3 k_d = vec3(0.8f, 0.8f, 0.8f);
    vec3 k_s = vec3(0.4f, 0.4f, 0.4f);
    float shininess = 32.0f;

    vec4 tot = vec4(phong_illumination(k_a, k_d, k_s, shininess, p, eye), 1.0f);
    color = tot;
}
