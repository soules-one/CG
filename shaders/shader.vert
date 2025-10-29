#version 450

// NOTE: Attributes must match the declaration of VkVertexInputAttribute array
layout(location = 0) in vec3 v_position;
layout(location = 2) in vec3 v_color;

layout(location = 0) out vec3 f_color;
// NOTE: Must match declaration order of a C struct
layout (push_constant, std430) uniform ShaderConstants {
	mat4 projection;
	mat4 transform;
	vec3 color;
};

void main() {
	vec4 point = vec4(v_position, 1.0f);
	vec4 transformed = transform * point;
	vec4 projected = projection * transformed;
	gl_Position = projected;
    f_color = color * 0.7 + v_color * 0.3;
}
