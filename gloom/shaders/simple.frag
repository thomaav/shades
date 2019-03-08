#version 430 core

out vec4 color;

uniform layout(location=0) vec2 window_size;
uniform layout(location=1) float time;

const int MARCHSTEPS = 250;
const float MIN_DIST = 0.0f;
const float MAX_DIST = 100.0f;
const float EPSILON = 0.005;
const float AA = 2.0f;
const float PI = 3.1415926535897932384626433832795;

float sd_plane(vec3 point)
{
    float dist = length(point - 0.0f);
    return point.y + 0.7*sin(3*dist + 3*time)/(3*dist);
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

float sd_displace(vec3 point)
{
    return sin(45.0f*point.x) * sin(45.0f*point.y) * sin(45.0f*point.z);
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
    float box = sd_box(point, vec3(0.5f, 0.0f, 0.5f), 0.01);
    float sphere = sd_sphere(sd_translate(point, vec3(0.0f, 0.5f, 0.0f)), 0.45f);
    float sphere2 = sd_sphere(sd_translate(point, vec3(0.0f, 1.1f, 0.0f)), 0.45f);
    sphere2 = sphere2 + 0.05*sd_displace(point);

    float scene = sd_smooth_union(box, sphere, 0.05f);
    scene = sd_smooth_union(sphere2, scene, 0.05f);
    return sd_smooth_union(plane, scene, 0.05f);
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

// http://delivery.acm.org/10.1145/1190000/1185834/p153-evans.pdf?ip=129.241.110.156&id=1185834&acc=ACTIVE%20SERVICE&key=CDADA77FFDD8BE08%2E5386D6A7D247483C%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1552050912_e7cfd8c9dad8342ae20b52e0aeabbaf2
// http://www.iquilezles.org/www/material/nvscene2008/rwwtt.pdf
float ambient_occlusion(vec3 pos, vec3 normal)
{
    float occ = 0.0f;
    float sca = 1.0f;

    for (int i = 0; i < 5; ++i) {
        float h = 0.001 + 0.15*float(i)/4.0f;
        float d = sd_scene(pos + h*normal);
        occ += (h-d)*sca;
        sca *= 0.95;
    }

    return clamp(1.0f - 1.5f*occ, 0.0f, 1.0f);
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

vec3 phong_light_contrib(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                         vec3 light_pos, vec3 light_intensity)
{
    vec3 N = estimate_normal(p);
    vec3 L = normalize(light_pos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));

    float dot_LN = dot(L, N);
    float dot_RV = dot(R, V);

    vec3 diffuse_light_intensity = light_intensity;
    if (p.y < 0.0f)
        diffuse_light_intensity = vec3(0.0f, 0.0f, 1.0f);

    if (dot_LN < 0.0f)
        return vec3(0.0f, 0.0f, 0.0f);

    if (dot_RV < 0.0f)
        return diffuse_light_intensity * (k_d * dot_LN);

    float shadow = penumbra_shadow(p, L, 0.01f, 100.0f, 32.0f);
    return diffuse_light_intensity * k_d * dot_LN * shadow
        + light_intensity * k_s * pow(dot_RV, alpha);
}

vec3 phong_illumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha,
                        vec3 p, vec3 eye)
{
    float occlusion = ambient_occlusion(p, estimate_normal(p));
    occlusion = 1.0f;
    vec3 ambient_light = 0.3 * vec3(1.0f, 1.0f, 1.0f);
    vec3 tmp_color = k_a * ambient_light * occlusion;

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
    vec3 eye = vec3(3.0f, 3.0f, 10.0f);
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
    float shininess = 8.0f;

    vec4 tot = vec4(phong_illumination(k_a, k_d, k_s, shininess, p, eye), 1.0f);
    color = tot;
    // color = vec4(dist/50, 0.0f, 0.0f, 1.0f);
}
