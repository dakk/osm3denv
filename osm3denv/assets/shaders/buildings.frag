#version 150

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 fog_colour;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in float v_fog_factor;

out vec4 frag_color;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p, int octaves) {
    float amp = 0.5;
    float sum = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

// Standard brick: ~25 cm wide, 7 cm tall, with half-offset alternating rows.
vec3 brick_wall(vec2 uv) {
    vec2 brick = vec2(0.25, 0.07);
    float row_idx = floor(uv.y / brick.y);
    float row_offset = mod(row_idx, 2.0) * brick.x * 0.5;
    vec2 local = vec2(uv.x - row_offset, uv.y);
    vec2 cell = floor(local / brick);
    vec2 f = fract(local / brick);

    float mortar_frac = 0.10;  // mortar occupies ~10% of cell on each side
    float m = 1.0 -
        smoothstep(0.0, mortar_frac, f.x) *
        smoothstep(0.0, mortar_frac, 1.0 - f.x) *
        smoothstep(0.0, mortar_frac, f.y) *
        smoothstep(0.0, mortar_frac, 1.0 - f.y);

    // Per-brick colour lottery.
    float seed = hash21(cell + row_offset);
    vec3 brick_cols[6] = vec3[](
        vec3(0.62, 0.24, 0.15),
        vec3(0.55, 0.22, 0.15),
        vec3(0.58, 0.28, 0.18),
        vec3(0.50, 0.20, 0.13),
        vec3(0.68, 0.30, 0.20),
        vec3(0.48, 0.22, 0.17)
    );
    int pick = int(mod(floor(seed * 6.0), 6.0));
    vec3 bc = brick_cols[pick];
    // Surface variation + vignette within the brick.
    bc *= 0.82 + 0.30 * fbm(uv * 40.0, 3);
    float edge = 1.0 - 0.35 * max(max(abs(f.x - 0.5) * 2.0, abs(f.y - 0.5) * 2.0), 0.0);
    bc *= edge;

    vec3 mortar_col = vec3(0.42, 0.40, 0.37) * (0.9 + 0.2 * hash21(cell * 3.7));
    return mix(bc, mortar_col, m);
}

// Overlay windows on the wall colour. Window "glass" is dark blue-grey.
vec3 windows(vec2 uv, vec3 wall_col) {
    vec2 spacing = vec2(1.6, 3.0);   // 1 window per ~1.6 m x 3 m (floor) cell
    vec2 size = vec2(0.9, 1.4);      // 0.9 m x 1.4 m window opening
    vec2 cell = floor(uv / spacing);
    vec2 within = fract(uv / spacing) * spacing - (spacing - size) * 0.5;
    // Skip windows on the ground floor (no window if cell.y == 0 — the row nearest the base).
    float alive = step(0.5, cell.y);  // no windows on row 0; all upper rows have windows
    bvec2 inside = bvec2(within.x > 0.0 && within.x < size.x,
                          within.y > 0.0 && within.y < size.y);
    if (alive > 0.5 && inside.x && inside.y) {
        // Glass pane with inner highlight variation per window.
        float ws = hash21(cell);
        vec3 glass = mix(vec3(0.16, 0.20, 0.28), vec3(0.30, 0.38, 0.48), ws);
        // Frame: thin border just inside the opening.
        vec2 d_edge = min(within, size - within);
        float frame = 1.0 - smoothstep(0.0, 0.05, min(d_edge.x, d_edge.y));
        vec3 frame_col = vec3(0.15, 0.12, 0.10);
        return mix(glass, frame_col, frame);
    }
    return wall_col;
}

vec3 roof_pattern(vec2 uv) {
    // Terracotta tile rows: thin horizontal ridges + per-tile colour variation.
    float row = uv.y / 0.30;  // 30 cm per tile row
    float band = abs(fract(row) - 0.5) * 2.0;  // 0 at band centre, 1 at edges
    float shade = mix(0.55, 1.0, 1.0 - band);
    float col_jit = hash21(vec2(floor(uv.x / 0.22), floor(row)));
    vec3 tile_base = mix(vec3(0.58, 0.30, 0.20), vec3(0.76, 0.42, 0.25), col_jit);
    vec3 groove = vec3(0.35, 0.25, 0.18);
    vec3 c = mix(groove, tile_base, shade);
    c *= 0.90 + 0.15 * fbm(uv * 8.0, 3);
    return c;
}

void main() {
    vec3 N = normalize(v_world_normal);
    float up = N.y;
    vec3 base;
    if (up > 0.6) {
        base = roof_pattern(v_uv);
    } else if (up < -0.6) {
        base = vec3(0.28, 0.26, 0.24);  // floor underside
    } else {
        base = brick_wall(v_uv);
        base = windows(v_uv, base);
    }

    float diffuse = max(dot(N, -normalize(light_direction.xyz)), 0.0);
    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse);
    vec3 final = mix(fog_colour, lit, v_fog_factor);
    frag_color = vec4(final, 1.0);
}
