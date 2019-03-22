#version 430 core

out vec4 color;

uniform layout(location=0) vec2 window_size;
uniform layout(location=1) float time;

uniform sampler2D noise_texture;

const int MARCHSTEPS = 400;
const float MIN_DIST = 0.0f;
const float MAX_DIST = 100.0f;
const float EPSILON = 0.0005;
const float AA = 2.0f;
const float PI = 3.1415926535897932384626433832795;

const vec3 underwater_color = vec3(0.0f, 0.0f, 0.10f);
const vec3 sunny_sky_color = vec3(0.31f, 0.62f, 0.86f);
const vec3 rainy_sky_color = vec3(0.22f, 0.26f, 0.29f);
const vec3 sphere_color = vec3(1.0f, 1.0f, 1.0f);

const vec3 scene_eye = vec3(3.0f, 1.0f, 10.0f);

#define SPHERE_RADIUS 0.45f
#define SPHERE_TRANSLATION (sd_translate(p, vec3(0.0f, sin(time) * -1.0f, 0.0f)))
#define sd_sphere_call (sd_sphere(SPHERE_TRANSLATION, SPHERE_RADIUS))

#define REFLECTION
#define REFRACTION
#define SHADOW
// #define AO
#define RAIN
#define RAIN_SPLASH
#define WATER_FOAM
#define CLOUDS

float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233)))*48301.231*sin(time));
}

vec2 noise(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);

    // Cubic Hermine Curve (smoothstep).
    f = f*f*(3.0-2.0*f);

    vec2 uv = (i.xy + vec2(37.0, 17.0)*i.z);
    vec4 rg = textureLod(noise_texture, (uv+f.xy+0.5)/256.0, 0.0);

    return mix(rg.yw, rg.xz, f.z);
}

float fbm(vec2 p)
{
    float v = 0.0, f = 1.0, a = 0.5;

    for (int i = 0; i < 5; ++i) {
        v += noise(vec3(p, 1.0f)*f).y*a;

        f *= 2.0;
        a *= 0.5;
    }

    return v;
}

float sd_waves(vec3 p)
{
    float height = p.y;
    p *= 0.2*vec3(1, 1, 1);

    const int octaves = 5;
    float f = 0.0;

    p += time*vec3(0, 0.1, 0.1);
    for (int i = 0; i < octaves; ++i) {
        p = (p.yzx + p.zyx*vec3(1, -1, 1)) / sqrt(2.0);
        f = f*2.0 + abs(noise(p).x - 0.5)*2.0;
        p *= 2.0;
    }

    f /= exp2(float(octaves));
    float turbulence = (0.5 - f)*1.0;

    return height - turbulence;
}

float sd_waves_foam(vec3 p)
{
    return 0.0f;
}

float sd_plane(vec3 p)
{
    float dist = length(p);

    if (dist > 7.5f)
        return p.y;

    return p.y + 0.03*(dist)*cos(dist*5.0f-time*5.0f);
}

float sd_sphere(vec3 p, float r)
{
    return length(p) - r;
}

float sd_box(vec3 p, vec3 b, float r)
{
    vec3 d = abs(p) - b;
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

float sd_displace(vec3 p)
{
    return sin(45.0f*p.x) * sin(45.0f*p.y) * sin(45.0f*p.z);
}

vec3 sd_translate(vec3 p, vec3 translation)
{
    return p - translation;
}

vec3 sd_rotate(vec3 p, float theta)
{
    return (mat4(
        vec4(cos(theta), 0.0f, sin(theta), 0.0f),
        vec4(0.0f, 1.0f, 0.0f, 0.0f),
        vec4(-sin(theta), 0.0f, cos(theta), 0.0f),
        vec4(0.0f, 0.0f, 0.0f, 1.0f)
    ) * vec4(p, 1.0f)).xyz;
}

float sd_scene(vec3 p)
{
    float plane = sd_waves(sd_translate(p, vec3(0.0f, -0.5f, 0.0f)));
    float sphere = sd_sphere_call;

    return sd_union(plane, sphere);
}

float sd_scene_sphere(vec3 p)
{
    return sd_sphere_call;
}

float sd_scene_plane(vec3 p)
{
    return sd_waves(sd_translate(p, vec3(0.0f, -0.5f, 0.0f)));
}

float trace_sphere(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_sphere(eye + depth * dir);

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

vec3 est_normal(vec3 p)
{
    const float EPSILON = 0.01*length(p);
    return normalize(vec3(
          sd_scene(vec3(p.x + EPSILON, p.y, p.z)) - sd_scene(vec3(p.x - EPSILON, p.y, p.z)),
          sd_scene(vec3(p.x, p.y + EPSILON, p.z)) - sd_scene(vec3(p.x, p.y - EPSILON, p.z)),
          sd_scene(vec3(p.x, p.y, p.z + EPSILON)) - sd_scene(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

vec3 est_sphere_normal(vec3 p, float r)
{
    const float EPSILON = 0.001;
    p = SPHERE_TRANSLATION;
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
    float EPSILON = 0.0001;
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
    #ifdef SHADOW
    if (length(p) < 5.0f && eye == scene_eye)
        shadow = penumbra_shadow(p, L, 0.01f, 100.0f, 16.0f);
    #endif

    return diffuse_light_intensity * k_d * dot_LN * shadow
        + light_intensity * k_s * pow(dot_RV, alpha);
}

vec3 phong_illumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha,
                        vec3 p, vec3 eye, vec3 light_intensity, vec3 normal)
{
    float occlusion = 1.0f;
    #ifdef AO
    occlusion = ambient_occlusion(p, normal);
    #endif
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

    vec3 ray_dir = ray_direction(45.0f, window_size, gl_FragCoord.xy);
    vec3 eye = scene_eye;
    mat4 camera_mat = camera(eye, vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    ray_dir = (camera_mat * vec4(ray_dir, 1.0f)).xyz;

    float dist_plane = trace_plane(eye, ray_dir, MIN_DIST, MAX_DIST);
    float dist_sphere = trace_sphere(eye, ray_dir, MIN_DIST, MAX_DIST);

    vec4 col = vec4(rainy_sky_color, 1.0f);

    #ifdef RAIN
    // https://www.shadertoy.com/view/XdSGDc
    vec2 q = gl_FragCoord.xy / window_size;
    float f = 5.0f;
    vec2 st = f * (q*vec2(1.5f, 0.05f) + vec2(-time*0.1 + q.y*0.5, time*0.12));
    f = (texture(noise_texture, st*0.5, -99.0).x + texture(noise_texture, st*0.284, -99.0).y);
    f = clamp(pow(abs(f)*0.5f, 29.0f)*140.0f, 0.00, q.y*0.4f + 0.5f);
    vec3 brightness = vec3(0.25f);
    vec4 rain = vec4(brightness*f, 1.0f);
    #endif

    if (min(dist_plane, dist_sphere) > MAX_DIST - EPSILON) {
        #ifdef CLOUDS
        vec2 uv = q*2.0 - 1.0;

        float p = fbm(uv - vec2(time / 10, 0.0));
        p = 1.0 - abs(p*2.0 - 1.0);
        vec3 cloud = pow(vec3(p), vec3(0.3)) - (uv.y + 3.0)*0.2;
        col = mix(col, vec4(cloud, 1.0f), 0.5f);
        #endif

        #ifdef RAIN
        col += rain;
        #endif

        return col;
    }

    if (dist_plane < dist_sphere) {
        vec3 p_plane = eye + dist_plane*ray_dir;
        vec3 n_plane = est_normal(p_plane);

        // Standard color of water.
        col = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                      p_plane, eye, underwater_color,
                                      n_plane), 1.0f);

        // Refraction.
        #ifdef REFRACTION
        float raydotn = dot(ray_dir, n_plane);
        vec3 refr_ray = normalize(ray_dir + (-cos(1.10*acos(-raydotn))-raydotn)*n_plane);
        float refr_dist_sphere = trace_sphere(p_plane, refr_ray, MIN_DIST, MAX_DIST);
        if (refr_dist_sphere < MAX_DIST - EPSILON) {
            vec3 refr_p_sphere = p_plane + refr_dist_sphere*refr_ray;
            vec3 sphere_normal = est_sphere_normal(refr_p_sphere, SPHERE_RADIUS);
            vec4 refr_color = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                                      refr_p_sphere, p_plane, sphere_color,
                                                      sphere_normal), 1.0f);
            col = mix(col, refr_color, exp(-refr_dist_sphere));
        }
        #endif

        // Reflection.
        #ifdef REFLECTION
        float fresnel = pow(1.0f-abs(dot(ray_dir, n_plane)), 2.0f);
        vec3 refl_dir = reflect(ray_dir, n_plane);
        float refl_dist_sphere = trace_sphere(p_plane, refl_dir, MIN_DIST, MAX_DIST);
        vec4 reflection = vec4(rainy_sky_color, 1.0f);
        if (refl_dist_sphere < MAX_DIST - EPSILON) {
            vec3 refl_p_sphere = p_plane + refl_dist_sphere*refl_dir;
            vec3 refl_sphere_n = est_sphere_normal(refl_p_sphere, SPHERE_RADIUS);
            reflection = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                                 refl_p_sphere, p_plane, sphere_color,
                                                 refl_sphere_n), 1.0f);
        }
        col = mix(col, reflection, fresnel);

        #ifdef RAIN_SPLASH
        vec2 q = (gl_FragCoord.xy / window_size)*500.0f;
        vec2 i = floor(q);
        vec2 f = fract(q);

        float splash = random(i);
        if (splash >= 0.999f)
            col = mix(col, vec4(1.0f), 0.5f);
        #endif

        #endif
    } else {
        vec3 p_sphere = eye + dist_sphere*ray_dir;
        col = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                      p_sphere, eye, sphere_color,
                                      est_sphere_normal(p_sphere, SPHERE_RADIUS)), 1.0f);
    }

    #ifdef RAIN
    col += rain;
    #endif

    #ifdef WATER_FOAM
    ;
    #endif

    return col;
}

void main()
{
    color = shade_scene();
}
