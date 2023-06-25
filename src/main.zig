const std = @import("std");
const helper = @import("helper.zig");
const ArrayList = std.ArrayList;
const c = @cImport({
	@cDefine("GLFW_INCLUDE_VULKAN", {});
	@cInclude("GLFW/glfw3.h");
});
const Allocator = std.mem.Allocator;
const MAX_FRAMES_IN_FLIGHT = 2;
const SCREEN_W = 800;
const SCREEN_H = 600;
const validationLayers = [_][*c]const u8{"VK_LAYER_KHRONOS_validation"};

const deviceExtensions = [_][*c]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};


fn vkDie(res: c.VkResult) !void {
	return switch (res) {
		c.VK_SUCCESS => {},
		c.VK_NOT_READY => error.VkNotReady,
		c.VK_TIMEOUT => error.VkTimeOut,
		c.VK_EVENT_SET => error.VkEventSet,
		c.VK_EVENT_RESET => error.VkEventReset,
		c.VK_ERROR_OUT_OF_HOST_MEMORY => error.VkOutOfHostMemory,
		c.VK_ERROR_OUT_OF_DEVICE_MEMORY => error.VkOutOfDeviceMemory,
		c.VK_ERROR_INITIALIZATION_FAILED => error.VkInitializationFailed,
		c.VK_ERROR_DEVICE_LOST => error.VkDeviceLost,
		c.VK_ERROR_MEMORY_MAP_FAILED => error.VkMemoryManFailed,
		c.VK_ERROR_LAYER_NOT_PRESENT => error.VkLayerNotPresent,
		c.VK_ERROR_EXTENSION_NOT_PRESENT => error.VkExtensionNotPresent,
		c.VK_ERROR_INCOMPATIBLE_DRIVER => error.VkIncompatibleDriver,
		c.VK_ERROR_TOO_MANY_OBJECTS => error.VkTooManyObjects,
		c.VK_ERROR_FORMAT_NOT_SUPPORTED => error.VkFormatNotSupported,
		c.VK_ERROR_FRAGMENTED_POOL => error.VkFragmentedPool,
		c.VK_ERROR_UNKNOWN => error.VKUnknownError,
		c.VK_ERROR_OUT_OF_POOL_MEMORY => error.VkOutOfPoolMemory,
		c.VK_ERROR_INVALID_EXTERNAL_HANDLE => error.VkInvalidExternalHandle,
		c.VK_ERROR_FRAGMENTATION => error.VkFragmentationError,
		c.VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS => error.VkInvalidOpaqueCaptureAddress,
		else => @panic("invalid VK_RESULT"),
	};
}

fn CreateDebugUtilsMessengerEXT(instance: c.VkInstance, createInfo: *const c.VkDebugUtilsMessengerCreateInfoEXT, allocator: ?*c.VkAllocationCallbacks, debugMessenger: *c.VkDebugUtilsMessengerEXT) c.VkResult {
	const func = @ptrCast(c.PFN_vkCreateDebugUtilsMessengerEXT, c.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
	if (func != null) {
		return (func.?)(instance, createInfo, allocator, debugMessenger);
	} else {
		return c.VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}
fn DestroyDebugUtilsMessengerEXT(
instance: c.VkInstance,
debugMessenger: c.VkDebugUtilsMessengerEXT,
allocator: ?*c.VkAllocationCallbacks,
) void {
	const func = @ptrCast(c.PFN_vkDestroyDebugUtilsMessengerEXT, c.vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
	if (func != null) {
	(func.?)(instance, debugMessenger, allocator);
	} else {
		@panic("failed to load DestroyDebugUtilsMessengerEXT");
	}
}

const QueueFamilyIndices = struct {
	graphicsFamily: ?u32 = null,
	presentFamily: ?u32 = null,
	const Self = @This();
	pub fn isComplete(self: *const Self) bool {
		return (self.graphicsFamily != null and self.presentFamily != null);
	}
};

const SwapChainSupportDetails = struct {
	capabilities: c.VkSurfaceCapabilitiesKHR,
	formats: []c.VkSurfaceFormatKHR,
	presentModes: []c.VkPresentModeKHR,
	allocator: Allocator,
	const Self = @This();
	pub fn init(allocator: Allocator, device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !Self {
		var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
		try vkDie(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities));
		var formatCount: u32 = undefined;
		try vkDie(c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, null));
		var formats = try allocator.alloc(c.VkSurfaceFormatKHR, formatCount);
		try vkDie(c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, formats.ptr));
		var presentModeCount: u32 = undefined;
		try vkDie(c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, null));
		var presentModes = try allocator.alloc(c.VkPresentModeKHR, presentModeCount);
		try vkDie(c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.ptr));

		return .{ .allocator = allocator, .capabilities = capabilities, .formats = formats, .presentModes = presentModes };
	}
	pub fn deinit(self: *Self) void {
		self.allocator.free(self.formats);
		self.allocator.free(self.presentModes);
	}
};

const HelloTriangleApplication = struct {
	window: *c.GLFWwindow,
	instance: c.VkInstance,
	debugMessenger: c.VkDebugUtilsMessengerEXT,
	physicalDevice: c.VkPhysicalDevice,
	device: c.VkDevice,
	graphicsQueue: c.VkQueue,
	presentQueue: c.VkQueue,
	surface: c.VkSurfaceKHR,
	swapChain: c.VkSwapchainKHR,
	swapChainImages: ArrayList(c.VkImage),
	swapChainImageViews: ArrayList(c.VkImageView),
	swapChainImageFormat: c.VkFormat,
	swapChainExtent: c.VkExtent2D,
	pipelineLayout: c.VkPipelineLayout,
	renderPass: c.VkRenderPass,
	graphicsPipeline: c.VkPipeline,
	swapChainFramebuffers: ArrayList(c.VkFramebuffer),
	commandPool: c.VkCommandPool,
	commandBuffers: [MAX_FRAMES_IN_FLIGHT]c.VkCommandBuffer,
	imageAvailableSemaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore,
	renderFinishedSemaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore,
	inFlightFences: [MAX_FRAMES_IN_FLIGHT]c.VkFence,
	currentFrame: u32,
	allocator: Allocator,
	const Self = @This();
	pub fn init(allocator: Allocator) Self {
		var self: Self = undefined;
		self.allocator = allocator;
		self.swapChainImages = ArrayList(c.VkImage).init(allocator);
		self.swapChainImageViews = ArrayList(c.VkImageView).init(allocator);
		self.swapChainFramebuffers = ArrayList(c.VkFramebuffer).init(allocator);
		self.currentFrame = 0;
		return self;
	}
	pub fn run(self: *Self) !void {
		try self.initWindow();
		try self.initVulkan();
		try self.mainLoop();
		try self.cleanup();
	}
	fn initWindow(self: *Self) !void {
		if (c.glfwInit() == c.GLFW_FALSE) return error.FailedInitGLFW;
		c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
		c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);
		self.window = c.glfwCreateWindow(SCREEN_W, SCREEN_H, "uwu", null, null) orelse return error.FailedCreateWindow;
	}
	fn initVulkan(self: *Self) !void {
		try self.createInstance();
		try self.setupDebugMessenger();
		try self.createSurface();
		try self.pickPhysicalDevice();
		try self.createLogicalDevice();
		try self.createSwapChain();
		try self.createImageViews();
		try self.createRenderPass();
		try self.createGraphicsPipeline();
		try self.createFramebuffers();
		try self.createCommandPool();
		try self.createCommandBuffers();
		try self.createSyncObjects();
	}
	fn createSyncObjects(self: *Self) !void {
		const semaphoreCreateInfo: c.VkSemaphoreCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = null,
			.flags = 0
		};
		const fenceCreateInfo: c.VkFenceCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = null,
			// so first vkWaitForFences doesnt block
			.flags = c.VK_FENCE_CREATE_SIGNALED_BIT
		};
		
		for (0..MAX_FRAMES_IN_FLIGHT) |i| {
			try vkDie(c.vkCreateSemaphore(self.device, &semaphoreCreateInfo, null, &self.imageAvailableSemaphores[i]));
			try vkDie(c.vkCreateSemaphore(self.device, &semaphoreCreateInfo, null, &self.renderFinishedSemaphores[i]));
			try vkDie(c.vkCreateFence(self.device, &fenceCreateInfo, null, &self.inFlightFences[i]));
		}
	}
	fn recordCommandBuffer(self: *Self, commandBuffer: c.VkCommandBuffer, imageIndex: usize) !void {
		
		const bufferBeginInfo: c.VkCommandBufferBeginInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pInheritanceInfo = null,
			.pNext = null,
			.flags = 0,
		};


		const renderPassBeginInfo: c.VkRenderPassBeginInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = self.renderPass,
			.framebuffer = self.swapChainFramebuffers.items[imageIndex],
			.renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapChainExtent },
			.clearValueCount = 1,
			.pClearValues = &c.VkClearValue{ .color = .{ .float32 = .{ 0, 0, 0, 1 } } },
			.pNext = null,
		};
		try vkDie(c.vkBeginCommandBuffer(commandBuffer, &bufferBeginInfo));
		c.vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, c.VK_SUBPASS_CONTENTS_INLINE);
		c.vkCmdBindPipeline(commandBuffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphicsPipeline);
		const viewport: c.VkViewport = .{
			.x = 0.0,
			.y = 0.0,
			.width = @floatFromInt(f32, self.swapChainExtent.width),
			.height = @floatFromInt(f32, self.swapChainExtent.height),
			.minDepth = 0.0,
			.maxDepth = 1.0
		};
		c.vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		c.vkCmdSetScissor(commandBuffer, 0, 1, &c.VkRect2D{ .offset = .{ .x = 0.0, .y = 0.0 }, .extent = self.swapChainExtent });

		c.vkCmdDraw(commandBuffer, 3, 1, 0, 0);

		c.vkCmdEndRenderPass(commandBuffer);
		try vkDie(c.vkEndCommandBuffer(commandBuffer));
	}
	fn createCommandBuffers(self: *Self) !void {
		const allocInfo: c.VkCommandBufferAllocateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = self.commandPool,
			.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = MAX_FRAMES_IN_FLIGHT,
			.pNext = null
		};
		try vkDie(c.vkAllocateCommandBuffers(self.device, &allocInfo, &self.commandBuffers));
	}
	fn createCommandPool(self: *Self) !void {
		const indices = try self.findQueueFamilies(self.physicalDevice);
		
		const commandPoolCreateInfo: c.VkCommandPoolCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = indices.graphicsFamily.?,
			.pNext = null,
		};

		try vkDie(c.vkCreateCommandPool(self.device, &commandPoolCreateInfo, null, &self.commandPool));
	}
	fn createFramebuffers(self: *Self) !void {
		try self.swapChainFramebuffers.resize(self.swapChainImageViews.items.len);
		for (0..self.swapChainImageViews.items.len) |i| {
			const framebufferCreateInfo: c.VkFramebufferCreateInfo = .{
				.sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = self.renderPass,
				.attachmentCount = 1,
				.pAttachments = &self.swapChainImageViews.items[i],
				.width = self.swapChainExtent.width,
				.height = self.swapChainExtent.height,
				.layers = 1,
				.pNext = null,
				.flags = 0
			};
			try vkDie(c.vkCreateFramebuffer(self.device, &framebufferCreateInfo, null, &self.swapChainFramebuffers.items[i]));
		}
	}
	fn createRenderPass(self: *Self) !void {
		const colorAttachmentDescription: c.VkAttachmentDescription = .{
			.format = self.swapChainImageFormat,
			.samples = c.VK_SAMPLE_COUNT_1_BIT,
			.loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.flags = 0
		};
		

		const colorAttachmentRef: c.VkAttachmentReference = .{
			.attachment = 0,
			.layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		const subpassDescription: c.VkSubpassDescription = .{
			.pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentRef,
			.inputAttachmentCount = 0,
			.pInputAttachments = null,
			.pDepthStencilAttachment = null,
			.pResolveAttachments = null,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = null,
			.flags = 0
		};
		
		const subpassDependency: c.VkSubpassDependency = .{
			.srcSubpass = c.VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = 0,
		};
		

		const renderPassCreateInfo: c.VkRenderPassCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &colorAttachmentDescription,
			.subpassCount = 1,
			.pSubpasses = &subpassDescription,
			.dependencyCount = 1,
			.pDependencies = &subpassDependency,
			.pNext = null,
			.flags = 0
		};

		try vkDie(c.vkCreateRenderPass(self.device, &renderPassCreateInfo, null, &self.renderPass));

	}
	fn createShaderModule(self: *Self, code: []u8) !c.VkShaderModule {

		const shaderModuleCreateInfo: c.VkShaderModuleCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.len,
			.pCode = @ptrCast(*u32, @alignCast(@alignOf(*u32), code.ptr)),
			.pNext = null,
			.flags = 0
		};
		var shaderModule: c.VkShaderModule = undefined;
		try vkDie(c.vkCreateShaderModule(self.device, &shaderModuleCreateInfo, null, &shaderModule));
		return shaderModule;
	}
	fn createGraphicsPipeline(self: *Self) !void {
		const vertShaderCode = try helper.readFile(self.allocator, "shaders/shader.vert.spv");
		defer self.allocator.free(vertShaderCode);
		
		const fragShaderCode = try helper.readFile(self.allocator, "shaders/shader.frag.spv");
		defer self.allocator.free(fragShaderCode);
		
		const vertShaderModule = try self.createShaderModule(vertShaderCode);
		defer c.vkDestroyShaderModule(self.device, vertShaderModule, null);
		const fragShaderModule = try self.createShaderModule(fragShaderCode);
		defer c.vkDestroyShaderModule(self.device, fragShaderModule, null);

		const vertShaderCreateInfo: c.VkPipelineShaderStageCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = c.VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertShaderModule,
			.pName = "main",
			.pSpecializationInfo = null,
			.pNext = null,
			.flags = 0
		};

		const fragShaderCreateInfo: c.VkPipelineShaderStageCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragShaderModule,
			.pName = "main",
			.pSpecializationInfo = null,
			.pNext = null,
			.flags = 0
		};
		
		

		const shaderStages = [_]c.VkPipelineShaderStageCreateInfo {
			vertShaderCreateInfo,
			fragShaderCreateInfo
		};
		

		const vertexInputCreateInfo: c.VkPipelineVertexInputStateCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 0,
			.pVertexBindingDescriptions = null,
			.vertexAttributeDescriptionCount = 0,
			.pVertexAttributeDescriptions = null,
			.pNext = null,
			.flags = 0
		};
		

		const inputAssemblyCreateInfo: c.VkPipelineInputAssemblyStateCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = c.VK_FALSE,
			.pNext = null,
			.flags = 0
		};
		

		const viewport: c.VkViewport = .{
			.x = 0.0,
			.y = 0.0,
			.width = @floatFromInt(f32, self.swapChainExtent.width),
			.height = @floatFromInt(f32, self.swapChainExtent.height),
			.minDepth = 0.0,
			.maxDepth = 1.0
		};
		

		const scissor: c.VkRect2D = .{
			.offset = .{ .x = 0, .y = 0 },
			.extent = self.swapChainExtent
		};
		

		const viewPortCreateInfo: c.VkPipelineViewportStateCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
			.pNext = null,
			.flags = 0
		};
		

		const rasterizationCreateInfo: c.VkPipelineRasterizationStateCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = c.VK_FALSE,
			.rasterizerDiscardEnable = c.VK_FALSE,
			.polygonMode = c.VK_POLYGON_MODE_FILL,
			.lineWidth = 1.0,
			.cullMode = c.VK_CULL_MODE_BACK_BIT,
			.frontFace = c.VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = c.VK_FALSE,
			.depthBiasConstantFactor = 0.0,
			.depthBiasClamp = 0.0,
			.depthBiasSlopeFactor = 0.0,
			.pNext = null,
			.flags = 0
		};
		


		const multiSamplingCreateInfo: c.VkPipelineMultisampleStateCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.sampleShadingEnable = c.VK_FALSE,
			.rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
			.minSampleShading = 1.0,
			.pSampleMask = null,
			.alphaToCoverageEnable = c.VK_FALSE,
			.alphaToOneEnable = c.VK_FALSE,
			.pNext = null,
			.flags = 0
		};
		
		const colorBlendAttachment: c.VkPipelineColorBlendAttachmentState = .{
			.colorWriteMask =
			c.VK_COLOR_COMPONENT_R_BIT |
			c.VK_COLOR_COMPONENT_G_BIT |
			c.VK_COLOR_COMPONENT_B_BIT |
			c.VK_COLOR_COMPONENT_A_BIT,
			.blendEnable = c.VK_FALSE,
			.srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE,
			.dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO,
			.colorBlendOp = c.VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
			.alphaBlendOp = c.VK_BLEND_OP_ADD

		};
		

		const colorBlendCreateInfo: c.VkPipelineColorBlendStateCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = c.VK_FALSE,
			.logicOp = c.VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
			.blendConstants = std.mem.zeroes([4]f32),
			.pNext = null,
			.flags = 0
		};
		

		const pipelineLayoutCreateInfo: c.VkPipelineLayoutCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 0,
			.pSetLayouts = null,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = null,
			.pNext = null,
			.flags = 0
		};

		try vkDie(c.vkCreatePipelineLayout(self.device, &pipelineLayoutCreateInfo, null, &self.pipelineLayout));

		const dynamicStates = [_]c.VkDynamicState { 
			c.VK_DYNAMIC_STATE_VIEWPORT,
			c.VK_DYNAMIC_STATE_SCISSOR
		};

		const dynamicCreateInfo: c.VkPipelineDynamicStateCreateInfo  = .{
			.sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = @intCast(u32, dynamicStates.len),
			.pDynamicStates = &dynamicStates,
			.pNext = null,
			.flags = 0
		};

		const pipelineCreateInfo: c.VkGraphicsPipelineCreateInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = &shaderStages,
			.pVertexInputState = &vertexInputCreateInfo,
			.pInputAssemblyState = &inputAssemblyCreateInfo,
			.pViewportState = &viewPortCreateInfo,
			.pRasterizationState = &rasterizationCreateInfo,
			.pMultisampleState = &multiSamplingCreateInfo,
			.pDepthStencilState = null,
			.pColorBlendState = &colorBlendCreateInfo,
			.pDynamicState = &dynamicCreateInfo,
			.pTessellationState = null,
			.layout = self.pipelineLayout,
			.renderPass = self.renderPass,
			.subpass = 0,
			.basePipelineHandle = null,
			.basePipelineIndex = -1,
			.pNext = null,
			.flags = 0
		};

		try vkDie(c.vkCreateGraphicsPipelines(self.device, null, 1, &pipelineCreateInfo, null, &self.graphicsPipeline));

	}
	fn createImageViews(self: *Self) !void {
		try self.swapChainImageViews.resize(self.swapChainImages.items.len);
		for (self.swapChainImages.items, 0..) |image, n| {
			const imageViewCreateInfo: c.VkImageViewCreateInfo = .{
				.sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = image,
				.viewType = c.VK_IMAGE_VIEW_TYPE_2D,
				.format = self.swapChainImageFormat,
				.components = .{
					.r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = c.VK_COMPONENT_SWIZZLE_IDENTITY },
				.subresourceRange = .{
					.aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				},
				.pNext = null,
				.flags = 0 
			};
			try vkDie(c.vkCreateImageView(self.device, &imageViewCreateInfo, null, &self.swapChainImageViews.items[n]));
		}
	}
	fn createSwapChain(self: *Self) !void {
		var supportDetails = try SwapChainSupportDetails.init(self.allocator, self.physicalDevice, self.surface);
		defer supportDetails.deinit();
		const format = chooseSwapSurfaceFormat(supportDetails.formats);
		const presentMode = chooseSwapPresentMode(supportDetails.presentModes);

		const extent = self.chooseSwapExtent(supportDetails.capabilities);

		var imageCount: u32 = supportDetails.capabilities.minImageCount + 1;
		if (supportDetails.capabilities.maxImageCount > 0 and imageCount > supportDetails.capabilities.maxImageCount) imageCount = supportDetails.capabilities.maxImageCount;

		const indices = try self.findQueueFamilies(self.physicalDevice);
		const queuesAreSame = indices.graphicsFamily == indices.presentFamily;

		const swapchainCreateInfo: c.VkSwapchainCreateInfoKHR = .{ .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = self.surface, .minImageCount = imageCount, .imageFormat = format.format, .imageColorSpace = format.colorSpace, .imageExtent = extent, .imageArrayLayers = 1, .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = if (queuesAreSame) c.VK_SHARING_MODE_EXCLUSIVE else c.VK_SHARING_MODE_CONCURRENT, .queueFamilyIndexCount = if (queuesAreSame) 0 else 2, .pQueueFamilyIndices = if (queuesAreSame) null else &[_]u32{ indices.graphicsFamily.?, indices.presentFamily.? }, .preTransform = supportDetails.capabilities.currentTransform, .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = presentMode, .clipped = c.VK_TRUE, .oldSwapchain = null, .pNext = null, .flags = 0 };
		try vkDie(c.vkCreateSwapchainKHR(self.device, &swapchainCreateInfo, null, &self.swapChain));

		var neededImageCount: u32 = undefined;
		try vkDie(c.vkGetSwapchainImagesKHR(self.device, self.swapChain, &neededImageCount, null));
		try self.swapChainImages.resize(neededImageCount);
		try vkDie(c.vkGetSwapchainImagesKHR(self.device, self.swapChain, &neededImageCount, self.swapChainImages.items.ptr));
		self.swapChainImageFormat = format.format;
		self.swapChainExtent = extent;
	}
	fn chooseSwapSurfaceFormat(availableFormats: []c.VkSurfaceFormatKHR) c.VkSurfaceFormatKHR {
		for (availableFormats) |form| {
			if (form.format == c.VK_FORMAT_B8G8R8A8_SRGB and form.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return form;
		}
		return availableFormats[0];
	}
	fn chooseSwapPresentMode(availablePresentModes: []c.VkPresentModeKHR) c.VkPresentModeKHR {
		for (availablePresentModes) |mode| {
			if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) return mode;
		}
		return c.VK_PRESENT_MODE_FIFO_KHR;
	}
	fn chooseSwapExtent(self: *Self, capabilities: c.VkSurfaceCapabilitiesKHR) c.VkExtent2D {
		if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
			return capabilities.currentExtent;
		} else {
			var w: i32 = undefined;
			var h: i32 = undefined;
			c.glfwGetFramebufferSize(self.window, &w, &h);
			var extent: c.VkExtent2D = .{ .width = @intCast(u32, w), .height = @intCast(u32, h) };

			extent.width = std.math.clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			extent.height = std.math.clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			return extent;
		}
	}
	fn createSurface(self: *Self) !void {
		try vkDie(c.glfwCreateWindowSurface(self.instance, self.window, null, &self.surface));
	}
	fn createLogicalDevice(self: *Self) !void {
		const indices = try self.findQueueFamilies(self.physicalDevice);
		const queueFamilies = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };
		var queueCreateInfos = ArrayList(c.VkDeviceQueueCreateInfo).init(self.allocator);
		defer queueCreateInfos.deinit();
		for (queueFamilies, 0..) |fam, n| {
			var wasFound = false;
			for (queueFamilies[0..n]) |famPrior| {
				if (fam == famPrior) {
					wasFound = true;
					break;
				}
			}
			if (!wasFound) {
				try queueCreateInfos.append(.{ .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = fam, .queueCount = 1, .pQueuePriorities = &@floatCast(f32, 1.0), .pNext = null, .flags = 0 });
			}
		}

		const deviceFeatures = std.mem.zeroes(c.VkPhysicalDeviceFeatures);
		const deviceCreateInfo: c.VkDeviceCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pQueueCreateInfos = queueCreateInfos.items.ptr, .queueCreateInfoCount = @intCast(u32, queueCreateInfos.items.len), .pEnabledFeatures = &deviceFeatures, .enabledExtensionCount = deviceExtensions.len, .ppEnabledExtensionNames = &deviceExtensions, .enabledLayerCount = validationLayers.len, .ppEnabledLayerNames = &validationLayers, .pNext = null, .flags = 0 };
		try vkDie(c.vkCreateDevice(self.physicalDevice, &deviceCreateInfo, null, &self.device));
		c.vkGetDeviceQueue(self.device, indices.graphicsFamily.?, 0, &self.graphicsQueue);
		c.vkGetDeviceQueue(self.device, indices.presentFamily.?, 0, &self.presentQueue);
	}
	fn findQueueFamilies(self: *Self, device: c.VkPhysicalDevice) !QueueFamilyIndices {
		var queueFamilyCount: u32 = 0;
		c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);
		const queueFamilies = try self.allocator.alloc(c.VkQueueFamilyProperties, queueFamilyCount);
		defer self.allocator.free(queueFamilies);
		c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);

		var indices: QueueFamilyIndices = .{};

		for (queueFamilies, 0..) |fam, n| {
			if (indices.isComplete()) break;
			if ((fam.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) > 0) {
				indices.graphicsFamily = @intCast(u32, n);
			}
			var presentSupport = c.VK_FALSE;
			try vkDie(c.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(u32, n), self.surface, &presentSupport));
			if (presentSupport == c.VK_TRUE) indices.presentFamily = @intCast(u32, n);
		}
		return indices;
	}
	fn checkDeviceExtensionSupport(self: *Self, device: c.VkPhysicalDevice) !bool {
		var extensionCount: u32 = 0;
		try vkDie(c.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, null));
		var availableExtensions = try self.allocator.alloc(c.VkExtensionProperties, extensionCount);
		defer self.allocator.free(availableExtensions);
		try vkDie(c.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, availableExtensions.ptr));
		for (deviceExtensions) |targetExt| {
			var found = false;
			for (availableExtensions) |realExt| {
				if (std.cstr.cmp(@ptrCast([*:0]const u8, &realExt.extensionName), targetExt) == 0) {
					found = true;
					break;
				}
			}
			if (!found) return false;
		}
		return true;
	}
	fn isDeviceSuitable(self: *Self, device: c.VkPhysicalDevice) !bool {
		const indices = try self.findQueueFamilies(device);
		const extensionsSupported = try self.checkDeviceExtensionSupport(device);
		var swapChainSupported = false;
		if (extensionsSupported) {
			var details = try SwapChainSupportDetails.init(self.allocator, device, self.surface);
			defer details.deinit();
			swapChainSupported = (details.formats.len > 0 and details.presentModes.len > 0);
		}
		return indices.isComplete() and extensionsSupported and swapChainSupported;
	}
	fn pickPhysicalDevice(self: *Self) !void {
		var deviceCount: u32 = 0;
		try vkDie(c.vkEnumeratePhysicalDevices(self.instance, &deviceCount, null));
		if (deviceCount == 0) return error.NoGPUWithVulkanSupport;
		var devices = try self.allocator.alloc(c.VkPhysicalDevice, deviceCount);
		defer self.allocator.free(devices);
		try vkDie(c.vkEnumeratePhysicalDevices(self.instance, &deviceCount, devices.ptr));
		var device: ?c.VkPhysicalDevice = null;
		for (devices) |d| {
			if (try self.isDeviceSuitable(d)) {
				device = d;
				break;
			}
		}
		self.physicalDevice = device orelse return error.FailedToFindSuitableGPU;
	}
	fn setupDebugMessenger(self: *Self) !void {
		const createInfo: c.VkDebugUtilsMessengerCreateInfoEXT = .{ .sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, .messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, .messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, .pfnUserCallback = debugCallback, .pUserData = null, .pNext = null, .flags = 0 };
		try vkDie(CreateDebugUtilsMessengerEXT(self.instance, &createInfo, null, &self.debugMessenger));
	}
	fn checkValidationLayerSupport(self: *Self) !bool {
		var count: u32 = undefined;
		try vkDie(c.vkEnumerateInstanceLayerProperties(&count, null));
		var layers = try self.allocator.alloc(c.VkLayerProperties, count);
		defer self.allocator.free(layers);
		try vkDie(c.vkEnumerateInstanceLayerProperties(&count, layers.ptr));
		for (validationLayers) |neededName| {
			var layerFound = false;
			for (layers) |layer| {
				if (std.cstr.cmp(neededName, @ptrCast([*:0]const u8, &layer.layerName)) == 0) {
					layerFound = true;
					break;
				}
			}
			if (!layerFound) return false;
		}
		return true;
	}
	fn getRequiredExtensions(self: *Self) !ArrayList([*c]const u8) {
		var glfwExtensionCount: u32 = 0;
		const glfwExtensions = c.glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		var extensions = ArrayList([*c]const u8).init(self.allocator);
		try extensions.appendSlice(glfwExtensions[0..glfwExtensionCount]);
		try extensions.append(c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		return extensions;
	}
	fn createInstance(self: *Self) !void {
		if (try self.checkValidationLayerSupport() == false) return error.ValidationLayersNotFound;

		const extensions = try self.getRequiredExtensions();
		defer extensions.deinit();
		const appInfo: c.VkApplicationInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = "Hello Triangle",
			.applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "No Engine",
			.engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = c.VK_API_VERSION_1_0,
			.pNext = null,
		};
		const createInfo: c.VkInstanceCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &appInfo, .enabledExtensionCount = @intCast(u32, extensions.items.len), .ppEnabledExtensionNames = extensions.items.ptr, .enabledLayerCount = validationLayers.len, .ppEnabledLayerNames = &validationLayers, .pNext = null, .flags = 0 };
		try vkDie(c.vkCreateInstance(&createInfo, null, &self.instance));
	}
	fn drawFrame(self: *Self) !void {
		try vkDie(c.vkWaitForFences(self.device, 1, &self.inFlightFences[self.currentFrame], c.VK_TRUE, std.math.maxInt(u64)));
		try vkDie(c.vkResetFences(self.device, 1, &self.inFlightFences[self.currentFrame]));

		var imageIndex: u32 = undefined;
		try vkDie(c.vkAcquireNextImageKHR(self.device, self.swapChain, std.math.maxInt(u64), self.imageAvailableSemaphores[self.currentFrame], null, &imageIndex));

		try vkDie(c.vkResetCommandBuffer(self.commandBuffers[self.currentFrame], 0));

		try self.recordCommandBuffer(self.commandBuffers[self.currentFrame], imageIndex);

		const submitInfo: c.VkSubmitInfo = .{
			.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &self.imageAvailableSemaphores[self.currentFrame],
			.pWaitDstStageMask = &@intCast(c.VkPipelineStageFlags, c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
			.commandBufferCount = 1,
			.pCommandBuffers = &self.commandBuffers[self.currentFrame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &self.renderFinishedSemaphores[self.currentFrame],
			.pNext = null,
		};

		try vkDie(c.vkQueueSubmit(self.graphicsQueue, 1, &submitInfo, self.inFlightFences[self.currentFrame]));

		const presentInfo: c.VkPresentInfoKHR = .{
			.sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &self.renderFinishedSemaphores[self.currentFrame],
			.swapchainCount = 1,
			.pSwapchains = &self.swapChain,
			.pImageIndices = &imageIndex,
			.pResults = null,
			.pNext = null
		};

		try vkDie(c.vkQueuePresentKHR(self.presentQueue, &presentInfo));


		self.currentFrame = (self.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}
	fn mainLoop(self: *Self) !void {
		while (c.glfwWindowShouldClose(self.window) == c.GLFW_FALSE) {
			c.glfwPollEvents();
			try self.drawFrame();
		}

		try vkDie(c.vkDeviceWaitIdle(self.device));
	}
	fn cleanup(self: *Self) !void {
		for (0..MAX_FRAMES_IN_FLIGHT) |i| {
			c.vkDestroySemaphore(self.device,  self.imageAvailableSemaphores[i], null);
			c.vkDestroySemaphore(self.device,  self.renderFinishedSemaphores[i], null);
			c.vkDestroyFence(self.device, self.inFlightFences[i], null);
		}
		c.vkDestroyCommandPool(self.device, self.commandPool, null);
		for (self.swapChainFramebuffers.items) |framebuffer| {
			c.vkDestroyFramebuffer(self.device, framebuffer, null);
		}
		self.swapChainFramebuffers.deinit();
		c.vkDestroyPipeline(self.device, self.graphicsPipeline, null);
		c.vkDestroyPipelineLayout(self.device, self.pipelineLayout, null);
		c.vkDestroyRenderPass(self.device, self.renderPass, null);
		for (self.swapChainImageViews.items) |imageView| {
			c.vkDestroyImageView(self.device, imageView, null);
		}
		self.swapChainImageViews.deinit();
		self.swapChainImages.deinit();
		c.vkDestroySwapchainKHR(self.device, self.swapChain, null);
		c.vkDestroyDevice(self.device, null);
		DestroyDebugUtilsMessengerEXT(self.instance, self.debugMessenger, null);
		c.vkDestroySurfaceKHR(self.instance, self.surface, null);
		c.vkDestroyInstance(self.instance, null);
		c.glfwDestroyWindow(self.window);
		c.glfwTerminate();
	}
	fn debugCallback(severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT, _: c.VkDebugUtilsMessageTypeFlagsEXT, callbackData: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT, _: ?*anyopaque) callconv(.C) c.VkBool32 {
		const Severity = enum { info, warning, fatal };
		const sev: Severity = switch (severity) {
			c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT, c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT => .info,
			c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT => .warning,
			c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT => .fatal,
			else => @panic("invalid severity"),
		};
		const tty = std.io.tty;
		const stdout = std.io.getStdOut();
		const ttyConfig = tty.detectConfig(stdout);
		ttyConfig.setColor(stdout, switch (sev) {
			.info => .blue,
			.warning => .yellow,
			.fatal => .red,
		}) catch @panic("failed to set color");
		std.debug.print("{s}\n", .{callbackData.*.pMessage});
		ttyConfig.setColor(stdout, .reset) catch @panic("failed to set color");
		return c.VK_FALSE;
	}
};

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer {
		const status = gpa.deinit();
		if (status == .leak) @panic("we leaked, boys :(\n");
	}
	const allocator = gpa.allocator();
	var app = HelloTriangleApplication.init(allocator);
	try app.run();
}
