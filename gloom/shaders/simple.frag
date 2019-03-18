#version 430 core

out vec4 color;

uniform layout(location=0) vec2 window_size;
uniform layout(location=1) float time;

const int MARCHSTEPS = 250;
const float MIN_DIST = 0.0f;
const float MAX_DIST = 100.0f;
const float EPSILON = 0.0005;
const float AA = 2.0f;
const float PI = 3.1415926535897932384626433832795;

const vec3 wave_color = vec3(0.35f, 0.74f, 0.85f);
const vec3 ball_color = vec3(1.0f, 1.0f, 1.0f);

#define SPHERE_RADIUS 0.45f
#define sd_sphere_call (sd_sphere(sd_translate(point, vec3(0.0f, -0.5f, 0.0f)), SPHERE_RADIUS))

float sd_plane(vec3 point)
{
    return point.y;
    float dist = length(point);

    if (dist > 7.5f)
        return point.y;

    return point.y + 0.03*(dist)*cos(dist*6.0f-time*6.0f);
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
    float sphere = sd_sphere_call;

    return sd_union(plane, sphere);
}

float sd_scene_ball(vec3 point)
{
    return sd_sphere_call;
}

float sd_scene_plane(vec3 point)
{
    return sd_plane(sd_translate(point, vec3(0.0f, -0.5f, 0.0f)));
}

float trace_ball(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_ball(eye + depth * dir);

        if (dist < EPSILON)
            return depth;

        depth += dist;
        if (depth >= end)
            return end;
    }

    return end;
}

float trace_plane(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_plane(eye + depth * dir);

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

vec3 estimate_sphere_normal(vec3 p, float r)
{
    const float EPSILON = 0.001;
    p = sd_translate(p, vec3(0.0f, -0.5f, 0.0f));
    return normalize(vec3(
        sd_sphere(vec3(p.x + EPSILON, p.y, p.z), r) - sd_sphere(vec3(p.x - EPSILON, p.y, p.z), r),
        sd_sphere(vec3(p.x, p.y + EPSILON, p.z), r) - sd_sphere(vec3(p.x, p.y - EPSILON, p.z), r),
        sd_sphere(vec3(p.x, p.y, p.z + EPSILON), r) - sd_sphere(vec3(p.x, p.y, p.z - EPSILON), r)
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
                         vec3 light_pos, vec3 light_intensity, vec3 normal)
{
    vec3 N = normal;
    vec3 L = normalize(light_pos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));

    float dot_LN = dot(L, N);
    float dot_RV = dot(R, V);

    vec3 diffuse_light_intensity = light_intensity;
    light_intensity = vec3(1.0f, 1.0f, 1.0f);

    if (dot_LN < 0.0f)
        return vec3(0.0f, 0.0f, 0.0f);

    if (dot_RV < 0.0f)
        return diffuse_light_intensity * (k_d * dot_LN);

    float shadow = 1.0f;
    if (length(p) < 5.0f)
        shadow = penumbra_shadow(p, L, 0.01f, 100.0f, 32.0f);

    shadow = 1.0f;
    return diffuse_light_intensity * k_d * dot_LN * shadow
        + light_intensity * k_s * pow(dot_RV, alpha);
}

vec3 phong_illumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha,
                        vec3 p, vec3 eye, vec3 light_intensity, vec3 normal)
{
    float occlusion = ambient_occlusion(p, estimate_normal(p));
    vec3 ambient_light = 0.3 * vec3(1.0f, 1.0f, 1.0f);
    vec3 tmp_color = k_a * ambient_light * occlusion;
    vec3 light_pos = vec3(10.0f, 8.0f, -10.0f);

    tmp_color += phong_light_contrib(k_d, k_s, alpha, p, eye,
                                     light_pos, light_intensity, normal);

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

vec4 shade_scene()
{
    vec3 k_a = vec3(0.5f, 0.5f, 0.5f);
    vec3 k_d = vec3(0.8f, 0.8f, 0.8f);
    vec3 k_s = vec3(0.4f, 0.4f, 0.4f);
    float shininess = 16.0f;

    vec3 dir = ray_direction(45.0f, window_size, gl_FragCoord.xy);
    vec3 eye = vec3(3.0f, 4.0f, 10.0f);
    mat4 camera_mat = camera(eye, vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    dir = (camera_mat * vec4(dir, 1.0f)).xyz;

    float dist_plane = trace_plane(eye, dir, MIN_DIST, MAX_DIST);
    float dist_ball = trace_ball(eye, dir, MIN_DIST, MAX_DIST);

    if (min(dist_plane, dist_ball) > MAX_DIST - EPSILON) {
        return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    vec3 p;
    if (dist_plane < dist_ball) {
        p = eye + dist_plane*dir;
        vec3 p_normal = estimate_normal(p);
        vec3 sphere_normal = estimate_sphere_normal(p, SPHERE_RADIUS);
        float ndotdir = dot(dir, p_normal);
        float fresnel = pow(1.0f-abs(dot(dir, p_normal)), 2.0f);
        vec3 ref_dir = reflect(dir, p_normal);
        float ref_dist_ball = trace_ball(p, ref_dir, MIN_DIST, MAX_DIST);

        // Standard color of water.
        vec4 col = vec4(phong_illumination(k_a, k_d, k_s, shininess, p, eye, wave_color, p_normal), 1.0f);

        // Refraction.
        vec3 refracted_dir = normalize(dir + (-cos(1.33*acos(-ndotdir))-ndotdir)*p_normal);
        float refracted_dist_ball = trace_ball(p, refracted_dir, MIN_DIST, MAX_DIST);
        if (refracted_dist_ball < MAX_DIST - EPSILON) {
            vec3 refracted_ball_p = p + refracted_dist_ball*refracted_dir;
            vec4 refraction_color =
                vec4(phong_illumination(k_a, k_d, k_s, shininess, refracted_ball_p, p, ball_color, sphere_normal), 1.0f);
            col = mix(col, refraction_color, exp(-refracted_dist_ball));
        }

        // Reflection.
        // if (ref_dist_ball < MAX_DIST - EPSILON) {
        //     vec3 ball_p = p + ref_dist_ball*ref_dir;
        //     vec4 reflection = vec4(phong_illumination(k_a, k_d, k_s, shininess, ball_p, p, ball_color), 1.0f, p_normal);
        //     col = mix(col, reflection, fresnel);
        // }

        return col;
    } else {
        p = eye + dist_ball*dir;
        return vec4(phong_illumination(k_a, k_d, k_s, shininess, p, eye, ball_color, estimate_normal(p)), 1.0f);
    }
}

void main()
{
    color = shade_scene();
}
