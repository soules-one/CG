#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;

float ind_num = 0;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec3 color;
};

veekay::mat4 identity() {
	veekay::mat4 result{};

	result[0][0] = 1.0f;
	result[1][1] = 1.0f;
	result[2][2] = 1.0f;
	result[3][3] = 1.0f;
	
	return result;
}

veekay::mat4 projection(float fov, float aspect_ratio, float near, float far) {
	veekay::mat4 result{};

	const float radians = fov * M_PI / 180.0f;
	const float cot = 1.0f / tanf(radians / 2.0f);

	result[0][0] = cot / aspect_ratio;
	result[1][1] = cot;
	result[2][3] = 1.0f;

	result[2][2] = far / (far - near);
	result[3][2] = (-near * far) / (far - near);

	return result;
}

veekay::vec3 sum(veekay::vec3 t, veekay::vec3 other){
	veekay::vec3 result = other;
	result.x += t.x;
	result.y += t.y;
	result.z += t.z;
	return result;
}

veekay::mat4 translation(veekay::vec3 vector) {
	veekay::mat4 result = identity();

	result[3][0] = vector.x;
	result[3][1] = vector.y;
	result[3][2] = vector.z;

	return result;
}

veekay::mat4 rotation(veekay::vec3 axis, float angle) {
	veekay::mat4 result{};

	float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

	axis.x /= length;
	axis.y /= length;
	axis.z /= length;

	float sina = sinf(angle);
	float cosa = cosf(angle);
	float cosv = 1.0f - cosa;

	result[0][0] = (axis.x * axis.x * cosv) + cosa;
	result[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
	result[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

	result[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
	result[1][1] = (axis.y * axis.y * cosv) + cosa;
	result[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

	result[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
	result[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
	result[2][2] = (axis.z * axis.z * cosv) + cosa;

	result[3][3] = 1.0f;

	return result;
}

veekay::mat4 multiply(const veekay::mat4& a, const veekay::mat4& b) {
	veekay::mat4 result{};

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			for (int k = 0; k < 4; k++) {
				result[j][i] += a[j][k] * b[k][i];
			}
		}
	}

	return result;
}

veekay::mat4 orthographic(float left, float right, float bottom, float top, float near, float far) {
	veekay::mat4 m{};
	m[0][0] = 2.0f / (right - left);
	m[1][1] = 2.0f / (top - bottom);
	m[2][2] = 1.0f / (far - near);
	m[3][0] = - (right + left) / (right - left);
	m[3][1] = - (top + bottom) / (top - bottom);
	m[3][2] = - near / (far - near);
	m[3][3] = 1.0f;
	return m;
}

// NOTE: These variable will be available to shaders through push constant uniform
struct ShaderConstants {
	veekay::mat4 projection;
	veekay::mat4 transform;
	veekay::vec3 color;
};
VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

// NOTE: Declare buffers and other variables here
veekay::graphics::Buffer* vertex_buffer;
veekay::graphics::Buffer* index_buffer;

veekay::vec3 model_position = {0.0f, 0.0f, 1.0f};
bool anim_reverse = false;
bool anim_play = false;
bool anim_switch = true;
bool is_circle = true;
float pause_time = 0.0f;
float acc_time = 0.0f;
float cicle = 10.0f;
float model_rotation;
veekay::vec3 model_color = {1.0f, 1.0f, 1.0f };
veekay::vec3 model_curr_diff = {0.0f, 0.0f, 0.0f};
veekay::mat4 curr_proj;
bool model_spin = false;
bool orthograph = true;
float fig_w = 0.5f;
float fig_h = 1.0f;
float radius = 5.0f;

veekay::vec3 circleAnimation(float time){
	veekay::vec3 diff = {0.0f, 0.0f, 0.0f};
	float angle = 2.0f * M_PIf * time / cicle;
	float x = radius * sin(angle);
	float z = radius - radius * cos(angle) ;
	diff.x =  x;
	diff.z =  z;
	return diff;
}

veekay::vec3 elipseAnimation(float time){
	veekay::vec3 diff = {0.0f, 0.0f, 0.0f};
	float angle = 2.0f * M_PIf * time / cicle;
	float y = radius * 2.0f * sin(angle);
	float x = y;
	float z = 2.0f * radius -  radius * 2.0f * cos(angle);
	diff.y = y;
	diff.z = z;
	diff.x = x;
	return diff;
}


// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize() {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			 {
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal)
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color)
			},
		};

		// NOTE: Bring 
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// NOTE: Declare constant memory region visible to vertex and fragment shaders
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
			              VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	class Cylinder{
	private:
		const int segments = 50;
		const float radius = fig_w / 2;
		const float height = fig_h;
		veekay::vec3 calculateGradient(float x, int i){
			float t = (x + height/2.0f) / height;
			return {0.0f + t, 0.0f, 1.0f - t};
		};
		void generateSideSurface(std::vector<Vertex> & vertices, std::vector<uint32_t> & indices){
			for (int i = 0; i < segments / 2; ++ i){
				float angle = 2.0f * M_PIf * i / (segments / 2);
				float x = radius * cos(angle);
				float z = radius * sin(angle);
				Vertex bottom, top;
				bottom.position = {x, -height/2.0f, z};
				bottom.normal = {cos(angle), 0.0f, sin(angle)};
				bottom.color = calculateGradient(-height/2.0f, i);
				top.position = {x, height/2.0f, z};
				top.normal = {cos(angle), 0.0f, sin(angle)};
				top.color = calculateGradient(height/2.0f, i);
				vertices.push_back(bottom);
				vertices.push_back(top);
			}
			for (int i = 0; i < segments / 2; ++ i){
				uint32_t bleft = (i * 2) % segments;
				uint32_t tleft = (i * 2 + 1) % segments;
				uint32_t bright = ((i + 1) * 2) % segments;
				uint32_t tright = (((i + 1) * 2) + 1) % segments;
				indices.insert(indices.end(), {tleft, bright, tright, bright, tleft, bleft});
				indices.insert(indices.end(), {tleft, tright, bright, bright, bleft, tleft});
			}
		};
	public:
		void generateGeometry(std::vector<Vertex> & vertices, std::vector<uint32_t> & indices){
			vertices.clear();
			indices.clear();
			generateSideSurface(vertices, indices);
		};
		
	};
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	Cylinder geom;
	geom.generateGeometry(vertices, indices);
	ind_num = indices.size();

	vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(),
	                                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(),
	                                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// NOTE: Destroy resources here, do not cause leaks in your program!
	delete index_buffer;
	delete vertex_buffer;

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&model_position));
	ImGui::SliderFloat("Rotation", &model_rotation, 0.0f, 2.0f * M_PI);
	ImGui::Checkbox("Spin?", &model_spin);
	// TODO: Your GUI stuff here
	ImGui::Checkbox("Ortho?", &orthograph);
	ImGui::Checkbox("Simple animation", &anim_switch);
	ImGui::SliderFloat("Animation Param", &radius, 0.1f, 10.f);
	ImGui::Checkbox("Play", &anim_play);
	ImGui::Checkbox("Reversed", &anim_reverse);
	ImGui::End();

	// NOTE: Animation code and other runtime variable updates go here
	if (model_spin) {
		model_rotation = float(time);
	}
	if(anim_switch != is_circle){
		acc_time = 0.0f;
		pause_time = 0.0f;
		is_circle = anim_switch;
	}
	if (anim_play){
		float delta;
		if (pause_time == 0.0f) pause_time = float(time);
		if (!anim_reverse)
			{delta = float(time) - pause_time;}
		else{
			delta = - float(time) + pause_time;}
		acc_time  = fmodf(acc_time + delta, cicle);
		if (acc_time < 0) acc_time += cicle;
		model_curr_diff = is_circle? circleAnimation(acc_time) : elipseAnimation(acc_time);
		pause_time = float(time);
	}
	if(!anim_play){
		pause_time = 0.0f;
	}
	const float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
	if (orthograph) {
			curr_proj = orthographic(-fig_w * aspect* 0.5f, fig_w  * aspect* 0.5f, 
				-fig_h * 0.5f, fig_h * 0.5f,
			camera_near_plane, camera_far_plane);
		} else {
			curr_proj = veekay::mat4::projection(
				camera_fov,
				aspect,
				camera_near_plane, camera_far_plane);
		}

	model_rotation = fmodf(model_rotation, 2.0f * M_PI);
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// TODO: Vulkan rendering code here
	// NOTE: ShaderConstant updates, vkCmdXXX expected to be here
	{
		// NOTE: Use our new shiny graphics pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


		// NOTE: Use our quad vertex buffer
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer->buffer, &offset);

		// NOTE: Use our quad index buffer
		vkCmdBindIndexBuffer(cmd, index_buffer->buffer, offset, VK_INDEX_TYPE_UINT32);

		// NOTE: Variables like model_XXX were declared globally

		ShaderConstants constants{
			.projection = curr_proj,

			.transform = veekay::mat4::rotation({1.0f, 1.0f, 0.0f}, model_rotation) *
			             veekay::mat4::translation(sum(model_position, model_curr_diff)),

			.color = model_color,
		};

		// NOTE: Update constant memory with new shader constants
		vkCmdPushConstants(cmd, pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(ShaderConstants), &constants);

		// NOTE: Draw 6 indices (3 vertices * 2 triangles), 1 group, no offsets
		vkCmdDrawIndexed(cmd, static_cast<uint32_t>(ind_num), 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
