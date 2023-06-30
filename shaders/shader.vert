#version 450

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

layout(push_constant) uniform constants {
	int triangleIndex;
} index;

void main() {
	const mat4 mvp = ubo.proj * ubo.view * ubo.model;
	gl_Position = mvp * vec4(inPosition, 1.0);
	const vec3 purple = vec3(0.5, 0.0, 1.0);
	const vec3 aqua = vec3(0.0, 0.5, 1.0);
	const float z = inPosition.z;
	fragColor = mix(purple, aqua, z);
	fragTexCoord = inTexCoord;
}
