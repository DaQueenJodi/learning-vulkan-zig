#version 450


#define MAX_ITERATIONS 128.0

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform constants {
	float time;
	uint height;
	uint width;
} resolution;

vec3 hash13(float m) {
	float x = fract(sin(m) * 5625.246);
	float y = fract(sin(m + x) * 2216.486);
	float z = fract(sin(x + y) * 8276.352);
	return vec3(x, y, z);
}
float mandelBrot(vec2 uv) {
	float iter = 0.0;
	vec2 c = 4.0 * uv - vec2(0.7, 0.0);
	c = c / pow(resolution.time, 4.0) - vec2(0.65, 0.45);
	vec2 z = vec2(0.0);
	for (float i = 0.0; i < MAX_ITERATIONS; i++) {
		z = vec2(z.x  * z.x - z.y * z.y,
					   2.0 * z.x * z.y) + c;
		if (dot(z, z) > 4.0) return iter/MAX_ITERATIONS;
		iter++;
	}
	return 0.0;
}

void main() {
	vec2 res = vec2(float(resolution.width), float(resolution.height));
	vec2 uv = (gl_FragCoord.xy - 0.5 * res.xy) / res.y;
	vec3 col = vec3(0.0);
	col += hash13(mandelBrot(uv));
	outColor = vec4(col, 1.0);
}
