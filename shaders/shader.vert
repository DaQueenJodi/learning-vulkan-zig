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
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
	int idx = index.triangleIndex;
	if (idx == 0) { fragColor = vec3(1, 0, 0); }
	else if (idx == 1) { fragColor = vec3(0, 1, 0); }
	else if (idx == 2) { fragColor = vec3(0, 0, 1); }
	else if (idx == 3) { fragColor = vec3(1, 1, 0); }
	else if (idx == 4) { fragColor = vec3(1, 0, 1); }

	fragTexCoord = inTexCoord;
}
