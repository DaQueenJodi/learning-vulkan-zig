const std = @import("std");
const zm = @import("zmath");
const helper = @import("helper.zig");
const ArrayList = std.ArrayList;
const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("GLFW/glfw3.h");
    @cInclude("stb_image.h");
});
const Allocator = std.mem.Allocator;
const MAX_FRAMES_IN_FLIGHT = 2;
const SCREEN_W = 800;
const SCREEN_H = 600;
const validationLayers = [_][*c]const u8{
    "VK_LAYER_KHRONOS_validation"
};

const deviceExtensions = [_][*c]const u8{
    c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

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
        else => error.VkUnknownError
    };
}

fn CreateDebugUtilsMessengerEXT(instance: c.VkInstance, createInfo: *const c.VkDebugUtilsMessengerCreateInfoEXT, allocator: ?*c.VkAllocationCallbacks, debugMessenger: *c.VkDebugUtilsMessengerEXT) c.VkResult {
    const func: c.PFN_vkCreateDebugUtilsMessengerEXT = @ptrCast(c.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
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
    const func: c.PFN_vkDestroyDebugUtilsMessengerEXT = @ptrCast(c.vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
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
const F32x2 = @Vector(2, f32);
const F32x3 = @Vector(3, f32);

const Vertex = struct {
    pos: F32x3,
    color: F32x3,
    texCoord: F32x2,
    const Self = @This();
    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Self),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX
        };
    }
    pub fn getAttributeDescription() [3]c.VkVertexInputAttributeDescription {
        return .{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(Self, "pos")
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R32G32B32_SFLOAT,
                .offset = @offsetOf(Self, "color")
            },
            .{
                .binding = 0,
                .location = 2,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(Self, "texCoord")
            }
        };
    }
};

const verticies = [_]Vertex{
    .{ .pos = .{  1.0,  1.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0 }, .texCoord = .{ 1.0, 0.0 } },
    .{ .pos = .{ -1.0,  1.0, 0.0 }, .color = .{ 0.0, 1.0, 0.0 }, .texCoord = .{ 0.0, 0.0 } },
    .{ .pos = .{  1.0, -1.0, 0.0 }, .color = .{ 0.0, 0.0, 1.0 }, .texCoord = .{ 0.0, 1.0 } },
    .{ .pos = .{ -1.0, -1.0, 0.0 }, .color = .{ 0.0, 0.0, 1.0 }, .texCoord = .{ 0.0, 1.0 } },
    .{ .pos = .{  0.0,  0.0, 1.0 }, .color = .{ 0.0, 0.0, 1.0 }, .texCoord = .{ 0.0, 0.0 } },
};

const indices = [_]u16{
    0, 1, 3, // bottom left triangle
    0, 2, 3, // bottom right triangle
    0, 4, 1, // left triangle
    1, 4, 3, // bottom triangle
    2, 4, 3, // right triangle
    0, 4, 2  // top triangle
};

const UniformBufferObject = struct {
    model: zm.Mat,
    view: zm.Mat,
    proj: zm.Mat,
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
    swapChainImages: []c.VkImage,
    swapChainImageViews: []c.VkImageView,
    swapChainImageFormat: c.VkFormat,
    swapChainExtent: c.VkExtent2D,
    descriptorSetLayout: c.VkDescriptorSetLayout,
    pipelineLayout: c.VkPipelineLayout,
    renderPass: c.VkRenderPass,
    graphicsPipeline: c.VkPipeline,
    swapChainFramebuffers: []c.VkFramebuffer,
    commandPool: c.VkCommandPool,
    commandBuffers: [MAX_FRAMES_IN_FLIGHT]c.VkCommandBuffer,
    imageAvailableSemaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore,
    renderFinishedSemaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore,
    inFlightFences: [MAX_FRAMES_IN_FLIGHT]c.VkFence,
    currentFrame: u32,
    frameResized: bool,
    vertexBuffer: c.VkBuffer,
    vertexBufferMemory: c.VkDeviceMemory,
    indexBuffer: c.VkBuffer,
    indexBufferMemory: c.VkDeviceMemory,
    uniformBuffers: [MAX_FRAMES_IN_FLIGHT]c.VkBuffer,
    uniformBuffersMemory: [MAX_FRAMES_IN_FLIGHT]c.VkDeviceMemory,
    uniformBuffersMapped: [MAX_FRAMES_IN_FLIGHT]*UniformBufferObject,
    descriptorPool: c.VkDescriptorPool,
    descriptorSets: [MAX_FRAMES_IN_FLIGHT]c.VkDescriptorSet,
    textureImage: c.VkImage,
    textureImageMemory: c.VkDeviceMemory,
    textureImageView: c.VkImageView,
    textureSampler: c.VkSampler,
    depthImage: c.VkImage,
    depthImageMemory: c.VkDeviceMemory,
    depthImageView: c.VkImageView,
    allocator: Allocator,
    const Self = @This();
    pub fn init(allocator: Allocator) Self {
        var self: Self = undefined;
        self.allocator = allocator;
        self.currentFrame = 0;
        self.frameResized = false;
        return self;
    }
    pub fn run(self: *Self) !void {
        try self.initWindow();
        try self.initVulkan();
        try self.mainLoop();
        try self.cleanup();
    }
    fn framebufferResizeCallback(window: ?*c.GLFWwindow, _: i32, _: i32) callconv(.C) void {
        const self: *Self = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(window)));
        self.frameResized = true;
    }
    fn initWindow(self: *Self) !void {
    if (c.glfwInit() == c.GLFW_FALSE) return error.FailedInitGLFW;
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);
        self.window = c.glfwCreateWindow(SCREEN_W, SCREEN_H, "uwu", null, null) orelse return error.FailedCreateWindow;
        c.glfwSetWindowUserPointer(self.window, self);
        _ = c.glfwSetFramebufferSizeCallback(self.window, framebufferResizeCallback);
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
        try self.createDescriptorSetLayout();
        try self.createGraphicsPipeline();
        try self.createCommandPool();
        try self.createDepthResources();
        try self.createFramebuffers();
        try self.createTextureImage();
        try self.createTextureImageView();
        try self.createTextureSampler();
        try self.createCommandBuffers();
        try self.createVertexBuffer();
        try self.createIndexBuffer();
        try self.createUniformBuffers();
        try self.createDescriptorPool();
        try self.createDescriptorSets();
        try self.createSyncObjects();
    }
    fn findSupportedFormat(self: *Self, candidates: []c.VkFormat, tiling: c.VkImageTiling, features: c.VkFormatFeatureFlags) !c.VkFormat {
        for (candidates) |format| {
            var props: c.VkFormatProperties = undefined;
            c.vkGetPhysicalDeviceFormatProperties(self.physicalDevice, format, &props);

            if (tiling == c.VK_IMAGE_TILING_LINEAR and (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == c.VK_IMAGE_TILING_OPTIMAL and (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        return error.FailedToFindSuitableFormat;
    }
    fn hasStencilComponent(format: c.VkFormat) bool {
        return switch (format) {
            c.VK_FORMAT_D32_SFLOAT_S8_UINT,
            c.VK_FORMAT_D24_UNORM_S8_UINT => true,
            else => false
        };
    }
    fn findDepthFormat(self: *Self) !c.VkFormat {
        var formats = [_]c.VkFormat{c.VK_FORMAT_D32_SFLOAT, c.VK_FORMAT_D32_SFLOAT_S8_UINT, c.VK_FORMAT_D24_UNORM_S8_UINT};
        return self.findSupportedFormat(
            formats[0..],
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );

    }
    fn createDepthResources(self: *Self) !void {
        const format = try self.findDepthFormat();
        try self.createImage(
            self.swapChainExtent.width,
            self.swapChainExtent.height,
            format,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &self.depthImage,
            &self.depthImageMemory
        );

        self.depthImageView = try self.createImageView(self.depthImage, format, c.VK_IMAGE_ASPECT_DEPTH_BIT);

    }
    fn createTextureSampler(self: *Self) !void {
        var properties: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(self.physicalDevice, &properties);
        const maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        const samplerCreateInfo: c.VkSamplerCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .magFilter = c.VK_FILTER_LINEAR,
            .minFilter = c.VK_FILTER_LINEAR,
            .addressModeU = c.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = c.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = c.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .anisotropyEnable = c.VK_TRUE,
            .maxAnisotropy = maxAnisotropy,
            .borderColor = c.VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = c.VK_FALSE,
            .compareEnable = c.VK_FALSE,
            .compareOp = c.VK_COMPARE_OP_ALWAYS,
            .mipmapMode = c.VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .mipLodBias = 0.0,
            .minLod = 0.0,
            .maxLod = 0.0,
        };

        try vkDie(c.vkCreateSampler(self.device, &samplerCreateInfo, null, &self.textureSampler));
    }
    fn createTextureImageView(self: *Self) !void {
        self.textureImageView = try self.createImageView(self.textureImage, c.VK_FORMAT_R8G8B8A8_SRGB, c.VK_IMAGE_ASPECT_COLOR_BIT);
    }
    fn createImageView(self: *Self, image: c.VkImage, format: c.VkFormat, aspectFlags: c.VkImageAspectFlags) !c.VkImageView {
        const viewCreateInfo: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = .{ .r = c.VK_COMPONENT_SWIZZLE_IDENTITY, .g = c.VK_COMPONENT_SWIZZLE_IDENTITY, .b = c.VK_COMPONENT_SWIZZLE_IDENTITY, .a = c.VK_COMPONENT_SWIZZLE_IDENTITY },
            .subresourceRange = .{ .aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
        };
        var imageView: c.VkImageView = undefined;
        try vkDie(c.vkCreateImageView(self.device, &viewCreateInfo, null, &imageView));
        return imageView;
    }
    fn transitionImageLayout(self: *Self, image: c.VkImage, format: c.VkFormat, oldLayout: c.VkImageLayout, newLayout: c.VkImageLayout) !void {
        _ = format;
        const commandBuffer = try self.beginSingleTimeCommands();

        var barrier: c.VkImageMemoryBarrier = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = null,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            // defined later
            .srcAccessMask = undefined,
            .dstAccessMask = undefined,
        };

        var srcStage: c.VkPipelineStageFlags = undefined;
        var dstStage: c.VkPipelineStageFlags = undefined;

        if (oldLayout == c.VK_IMAGE_LAYOUT_UNDEFINED and newLayout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;

            srcStage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == c.VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT;

            srcStage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = c.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            return error.UnsupportedLayoutTransition;
        }

        c.vkCmdPipelineBarrier(
            commandBuffer, 
            srcStage, dstStage,
            0,
            0, null,
            0, null,
            1, &barrier
        );

        try self.endSingleTimeCommands(commandBuffer);
    }
    fn copyBufferToImage(self: *Self, buffer: c.VkBuffer, image: c.VkImage, width: u32, height: u32) !void {
        const commandBuffer = try self.beginSingleTimeCommands();

        const region: c.VkBufferImageCopy = .{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .imageOffset = .{ .x = 0, .y = 0, .z = 0 },
            .imageExtent = .{ .width = width, .height = height, .depth = 1 },
        };

        c.vkCmdCopyBufferToImage(commandBuffer, buffer, image, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        try self.endSingleTimeCommands(commandBuffer);
    }
    fn createTextureImage(self: *Self) !void {
        var w: i32 = undefined;
        var h: i32 = undefined;
        var cs: i32 = undefined;

        const pixels = c.stbi_load("textures/charlotte.png", &w, &h, &cs, c.STBI_rgb_alpha).?;
        defer c.stbi_image_free(pixels);

        const size: u64 = @intCast(w * h * 4);

        var stagingBuffer: c.VkBuffer = undefined;
        var stagingBufferMemory: c.VkDeviceMemory = undefined;
        try self.createBuffer(size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory);

        var data: ?*anyopaque = undefined;
        try vkDie(c.vkMapMemory(self.device, stagingBufferMemory, 0, size, 0, &data));
        for (0..size) |i| @as([*]u8, @ptrCast(data))[i] = pixels[i];

        c.vkUnmapMemory(self.device, stagingBufferMemory);

        try self.createImage(
            @intCast(w),
            @intCast(h),
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | c.VK_IMAGE_USAGE_SAMPLED_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &self.textureImage,
            &self.textureImageMemory
        );

        try self.transitionImageLayout(
            self.textureImage,
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_LAYOUT_UNDEFINED,
            c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );
        try self.copyBufferToImage(stagingBuffer, self.textureImage, @intCast(w), @intCast(h));

        try self.transitionImageLayout(
            self.textureImage,
            c.VK_FORMAT_R8G8B8A8_SRGB,
            c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            c.VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL
        );

        c.vkDestroyBuffer(self.device, stagingBuffer, null);
        c.vkFreeMemory(self.device, stagingBufferMemory, null);
    }
    fn createImage(
        self: *Self,
        width: u32,
        height: u32,
        format: c.VkFormat,
        tiling: c.VkImageTiling,
        usage: c.VkImageUsageFlags,
        properties: c.VkMemoryPropertyFlags,
        image: *c.VkImage,
        imageMemory: *c.VkDeviceMemory
    ) !void {
        _ = properties;


        const imageCreateInfo: c.VkImageCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = null,
            .extent = .{ .width = width, .height = height, .depth = 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .format = format,
            .tiling = tiling,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .imageType = c.VK_IMAGE_TYPE_2D,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .flags = 0
        };
        var imageMut = image;
        try vkDie(c.vkCreateImage(self.device, &imageCreateInfo, null, imageMut));
        var memoryRequirements: c.VkMemoryRequirements = undefined;
        c.vkGetImageMemoryRequirements(self.device, image.*, &memoryRequirements);

        const allocInfo: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = try self.findMemoryType(memoryRequirements.memoryTypeBits, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        };
        try vkDie(c.vkAllocateMemory(self.device, &allocInfo, null, imageMemory));

        try vkDie(c.vkBindImageMemory(self.device, image.*, imageMemory.*, 0));
    }

    fn createDescriptorSets(self: *Self) !void {
        var layouts: [MAX_FRAMES_IN_FLIGHT]c.VkDescriptorSetLayout = undefined;
        @memset(layouts[0..], self.descriptorSetLayout);

        const allocInfo: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.descriptorPool,
            .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
            .pSetLayouts = &layouts
        };

        try vkDie(c.vkAllocateDescriptorSets(self.device, &allocInfo, &self.descriptorSets));

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            const bufferInfo: c.VkDescriptorBufferInfo = .{
                .buffer = self.uniformBuffers[i],
                .offset = 0,
                .range = @sizeOf(UniformBufferObject)
            };

            const imageInfo: c.VkDescriptorImageInfo = .{
                .imageLayout = c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .imageView = self.textureImageView,
                .sampler = self.textureSampler
            };
            const descriptorWrites = [_]c.VkWriteDescriptorSet{
                .{
                    .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = null,
                    .dstSet = self.descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .pBufferInfo = &bufferInfo,
                    .pImageInfo = undefined,
                    .pTexelBufferView = undefined
                },
                .{
                    .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = null,
                    .dstSet = self.descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .pImageInfo = &imageInfo,
                    .pBufferInfo = undefined,
                    .pTexelBufferView = undefined
                }
            };

            c.vkUpdateDescriptorSets(self.device, descriptorWrites.len, &descriptorWrites, 0, null);
        }
    }
    fn createDescriptorPool(self: *Self) !void {
        const poolSizes = [_]c.VkDescriptorPoolSize{
            .{ .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            .{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
        };

        const poolCreateInfo: c.VkDescriptorPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .poolSizeCount = 2,
            .pPoolSizes = &poolSizes,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .flags = 0
        };

        try vkDie(c.vkCreateDescriptorPool(self.device, &poolCreateInfo, null, &self.descriptorPool));
    }
    fn createUniformBuffers(self: *Self) !void {
        const size = @sizeOf(UniformBufferObject);


        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        try self.createBuffer(
            size,
            c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            &self.uniformBuffers[i],
        &self.uniformBuffersMemory[i]
            );

            try vkDie(c.vkMapMemory(self.device, self.uniformBuffersMemory[i], 0, size, 0, @ptrCast(&self.uniformBuffersMapped[i])));
        }
    }

    fn createDescriptorSetLayout(self: *Self) !void {

        const samplerLayoutBinding: c.VkDescriptorSetLayoutBinding = .{
            .binding = 1,
            .descriptorCount = 1,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImmutableSamplers = null,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };


        const uboLayoutBinding: c.VkDescriptorSetLayoutBinding = .{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = null
        };

        const layoutBindings = [_]c.VkDescriptorSetLayoutBinding{ uboLayoutBinding, samplerLayoutBinding };

        const layoutCreateInfo: c.VkDescriptorSetLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null,
            .bindingCount = 2,
            .pBindings = &layoutBindings,
            .flags = 0
        };

        try vkDie(c.vkCreateDescriptorSetLayout(self.device, &layoutCreateInfo, null, &self.descriptorSetLayout));

    }
    fn copyBuffer(self: *Self, srcBuffer: c.VkBuffer, dstBuffer: c.VkBuffer, size: c.VkDeviceSize) !void {
        const commandBuffer = try self.beginSingleTimeCommands();
        const copyRegion: c.VkBufferCopy = .{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size
        };
        c.vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        try self.endSingleTimeCommands(commandBuffer);
    }
    fn findMemoryType(self: *Self, typeFilter: u32, properties: c.VkMemoryPropertyFlags) !u32 {

        var memoryProperties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(self.physicalDevice, &memoryProperties);
        for (0..memoryProperties.memoryTypeCount) |i| {
            // check if the type and properties are correct
            if (((typeFilter & (@as(usize, 1) << @as(u6, @intCast(i)))) > 0) and ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) return @intCast(i);
        }
        return error.FailedToFindSuitableMemoryType;

    }
    fn createIndexBuffer(self: *Self) !void {

        var	stagingBuffer: c.VkBuffer = undefined;

        var stagingBufferMemory: c.VkDeviceMemory = undefined;

        const size = @sizeOf(@TypeOf(indices));

        try self.createBuffer(size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &stagingBuffer, &stagingBufferMemory);

        var data: [*]u16 = undefined;
        try vkDie(c.vkMapMemory(self.device, stagingBufferMemory, 0, size, 0, @ptrCast(&data)));
        @memcpy(data, &indices);
        c.vkUnmapMemory(self.device, stagingBufferMemory);

        try self.createBuffer(size, c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &self.indexBuffer, &self.indexBufferMemory);
        try self.copyBuffer(stagingBuffer, self.indexBuffer, size);

        c.vkDestroyBuffer(self.device, stagingBuffer, null);
    c.vkFreeMemory(self.device, stagingBufferMemory, null);

    }
    fn createVertexBuffer(self: *Self) !void {
        var	stagingBuffer: c.VkBuffer = undefined;
        var stagingBufferMemory: c.VkDeviceMemory = undefined;
        const size = @sizeOf(@TypeOf(verticies));
        try self.createBuffer(size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &stagingBuffer, &stagingBufferMemory);

        var data: [*]Vertex = undefined;
        try vkDie(c.vkMapMemory(self.device, stagingBufferMemory, 0, size, 0, @ptrCast(&data)));
        @memcpy(data, &verticies);
        c.vkUnmapMemory(self.device, stagingBufferMemory);
        try self.createBuffer(size, c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &self.vertexBuffer, &self.vertexBufferMemory);
        try self.copyBuffer(stagingBuffer, self.vertexBuffer, size);
        c.vkDestroyBuffer(self.device, stagingBuffer, null);
    c.vkFreeMemory(self.device, stagingBufferMemory, null);
    }
    fn createBuffer(self: *Self, size: c.VkDeviceSize, usage: c.VkBufferUsageFlags, properties: c.VkMemoryPropertyFlags, buffer: *c.VkBuffer, bufferMemory: *c.VkDeviceMemory) !void {
        const bufferCreateInfo: c.VkBufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .pNext = null,
            .flags = 0
        };
        try vkDie(c.vkCreateBuffer(self.device, &bufferCreateInfo, null, buffer));

        var memoryRequirements: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.device, buffer.*, &memoryRequirements);

        const allocInfo: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = try self.findMemoryType(memoryRequirements.memoryTypeBits, properties),
            .pNext = null
        };

        // make this parameter non-constant
        var bufferMemoryPtr = bufferMemory;

        try vkDie(c.vkAllocateMemory(self.device, &allocInfo, null, bufferMemoryPtr));
        try vkDie(c.vkBindBufferMemory(self.device, buffer.*, bufferMemory.*, 0));
    }
    fn cleanupSwapChain(self: *Self) void {
        self.allocator.free(self.swapChainImages);
        for (self.swapChainFramebuffers) |framebuffer| {
            c.vkDestroyFramebuffer(self.device, framebuffer, null);
        }
        self.allocator.free(self.swapChainFramebuffers);
        for (self.swapChainImageViews) |imageView| {
            c.vkDestroyImageView(self.device, imageView, null);
        }
        self.allocator.free(self.swapChainImageViews);
        c.vkDestroySwapchainKHR(self.device, self.swapChain, null);


        c.vkDestroyImageView(self.device, self.depthImageView, null);
        c.vkDestroyImage(self.device, self.depthImage, null);
        c.vkFreeMemory(self.device, self.depthImageMemory, null);
    }
    fn recreateSwapChain(self: *Self) !void {

        std.log.info("recreating swapchain!", .{});

        try vkDie(c.vkDeviceWaitIdle(self.device));

        self.cleanupSwapChain();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createDepthResources();
        try self.createFramebuffers();
    }
    fn createSyncObjects(self: *Self) !void {
        const semaphoreCreateInfo: c.VkSemaphoreCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = null, .flags = 0 };
        const fenceCreateInfo: c.VkFenceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            // so first vkWaitForFences doesnt block
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            try vkDie(c.vkCreateSemaphore(self.device, &semaphoreCreateInfo, null, &self.imageAvailableSemaphores[i]));
            try vkDie(c.vkCreateSemaphore(self.device, &semaphoreCreateInfo, null, &self.renderFinishedSemaphores[i]));
            try vkDie(c.vkCreateFence(self.device, &fenceCreateInfo, null, &self.inFlightFences[i]));
        }
    }
    fn beginSingleTimeCommands(self: *Self) !c.VkCommandBuffer {
        const allocInfo: c.VkCommandBufferAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandPool = self.commandPool,
            .commandBufferCount = 1
        };
        var commandBuffer: c.VkCommandBuffer = undefined;
        try vkDie(c.vkAllocateCommandBuffers(self.device, &allocInfo, &commandBuffer));
        const beginInfo: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null
        };
        try vkDie(c.vkBeginCommandBuffer(commandBuffer, &beginInfo));
        return commandBuffer;
    }
    fn endSingleTimeCommands(self: *Self, commandBuffer: c.VkCommandBuffer) !void {
        try vkDie(c.vkEndCommandBuffer(commandBuffer));
        const submitInfo: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null
        };

        try vkDie(c.vkQueueSubmit(self.graphicsQueue, 1, &submitInfo, null));
        try vkDie(c.vkQueueWaitIdle(self.graphicsQueue));

        c.vkFreeCommandBuffers(self.device, self.commandPool, 1, &commandBuffer);
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
            .framebuffer = self.swapChainFramebuffers[imageIndex],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapChainExtent },
            .clearValueCount = 2,
            .pClearValues = &[_]c.VkClearValue{
                .{ .color = .{ .float32 = .{ 0, 0, 0, 1 } } },
                .{ .depthStencil = .{ .depth = 1.0, .stencil = 0 } }
            },
            .pNext = null,
        };
        try vkDie(c.vkBeginCommandBuffer(commandBuffer, &bufferBeginInfo));
        c.vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(commandBuffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphicsPipeline);
        c.vkCmdBindVertexBuffers(commandBuffer, 0, 1, &self.vertexBuffer, &@as(u64, 0));
        c.vkCmdBindIndexBuffer(commandBuffer, self.indexBuffer, 0, c.VK_INDEX_TYPE_UINT16);
        const viewport: c.VkViewport = .{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(self.swapChainExtent.width),
            .height = @floatFromInt(self.swapChainExtent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0 
        };
        c.vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        c.vkCmdSetScissor(commandBuffer, 0, 1, &c.VkRect2D{ .offset = .{ .x = 0.0, .y = 0.0 }, .extent = self.swapChainExtent });

        c.vkCmdBindDescriptorSets(commandBuffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout, 0, 1, &self.descriptorSets[self.currentFrame], 0, null);
        for (0..3) |i| {
            c.vkCmdPushConstants(commandBuffer, self.pipelineLayout, c.VK_SHADER_STAGE_VERTEX_BIT, 0, @sizeOf(i32), &@as(i32, @intCast(i)));
            c.vkCmdDrawIndexed(commandBuffer, 3, 1, @intCast(3*i), 0, 0);
        }
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
        const queueIndices = try self.findQueueFamilies(self.physicalDevice);


        const commandPoolCreateInfo: c.VkCommandPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueIndices.graphicsFamily.?,
            .pNext = null,
        };

        try vkDie(c.vkCreateCommandPool(self.device, &commandPoolCreateInfo, null, &self.commandPool));
    }
    fn createFramebuffers(self: *Self) !void {
        for (0..self.swapChainImageViews.len) |i| {
            const framebufferCreateInfo: c.VkFramebufferCreateInfo = .{
                .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = self.renderPass,
                .attachmentCount = 2,
                .pAttachments = &[_]c.VkImageView{self.swapChainImageViews[i], self.depthImageView},
                .width = self.swapChainExtent.width,
                .height = self.swapChainExtent.height,
                .layers = 1,
                .pNext = null,
                .flags = 0 
            };
            try vkDie(c.vkCreateFramebuffer(self.device, &framebufferCreateInfo, null, &self.swapChainFramebuffers[i]));
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

        const colorAttachmentRef: c.VkAttachmentReference = .{ .attachment = 0, .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };


        const depthAttachmentDescription: c.VkAttachmentDescription = .{
            .format = try self.findDepthFormat(),
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .flags = 0
        };

        const depthAttachmentRef: c.VkAttachmentReference = .{ .attachment = 1, .layout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

        const subpassDescription: c.VkSubpassDescription = .{
            .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
            .inputAttachmentCount = 0,
            .pInputAttachments = null,
            .pDepthStencilAttachment = &depthAttachmentRef,
            .pResolveAttachments = null,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = null,
            .flags = 0
        };

        const subpassDependency: c.VkSubpassDependency = .{
            .srcSubpass = c.VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | c.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = 0,
            .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | c.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = 0,
        };

        const renderPassCreateInfo: c.VkRenderPassCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 2,
            .pAttachments = &[_]c.VkAttachmentDescription{colorAttachmentDescription, depthAttachmentDescription},
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
            .pCode = @ptrCast(@alignCast(code.ptr)),
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

        const shaderStages = [_]c.VkPipelineShaderStageCreateInfo{ vertShaderCreateInfo, fragShaderCreateInfo };

        const vertexBindingDescription = Vertex.getBindingDescription();
        const vertexAttributeDescriptions = Vertex.getAttributeDescription();
        const vertexInputCreateInfo: c.VkPipelineVertexInputStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBindingDescription,
            .vertexAttributeDescriptionCount = vertexAttributeDescriptions.len,
            .pVertexAttributeDescriptions = &vertexAttributeDescriptions,
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
            .width = @floatFromInt(self.swapChainExtent.width),
            .height = @floatFromInt(self.swapChainExtent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0 
        };

        const scissor: c.VkRect2D = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapChainExtent };

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
            .cullMode = c.VK_CULL_MODE_NONE,
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

        const colorBlendAttachment: c.VkPipelineColorBlendAttachmentState = .{ .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT |
            c.VK_COLOR_COMPONENT_G_BIT |
            c.VK_COLOR_COMPONENT_B_BIT |
            c.VK_COLOR_COMPONENT_A_BIT, .blendEnable = c.VK_FALSE, .srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO, .colorBlendOp = c.VK_BLEND_OP_ADD, .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO, .alphaBlendOp = c.VK_BLEND_OP_ADD };

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

        const pushConstants: c.VkPushConstantRange = .{
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = @sizeOf(i32)
        };

        const pipelineLayoutCreateInfo: c.VkPipelineLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &self.descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstants,
            .pNext = null,
            .flags = 0
        };

        try vkDie(c.vkCreatePipelineLayout(self.device, &pipelineLayoutCreateInfo, null, &self.pipelineLayout));

        const dynamicStates = [_]c.VkDynamicState{ c.VK_DYNAMIC_STATE_VIEWPORT, c.VK_DYNAMIC_STATE_SCISSOR };

        const dynamicCreateInfo: c.VkPipelineDynamicStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = @intCast(dynamicStates.len),
            .pDynamicStates = &dynamicStates,
            .pNext = null,
            .flags = 0 
        };

        const depthStencilCreateInfo: c.VkPipelineDepthStencilStateCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .depthTestEnable = c.VK_TRUE,
            .depthWriteEnable = c.VK_TRUE,
            .depthCompareOp = c.VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = c.VK_FALSE,
            .stencilTestEnable = c.VK_FALSE,
            .minDepthBounds = 0.0,
            .maxDepthBounds = 1.0,
            .front = std.mem.zeroes(c.VkStencilOpState),
            .back = std.mem.zeroes(c.VkStencilOpState),
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
            .pDepthStencilState = &depthStencilCreateInfo,
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
        for (self.swapChainImages, 0..) |image, i| {
            self.swapChainImageViews[i] = try self.createImageView(image, self.swapChainImageFormat, c.VK_IMAGE_ASPECT_COLOR_BIT);
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

        const queueIndices = try self.findQueueFamilies(self.physicalDevice);

        const queuesAreSame = queueIndices.graphicsFamily == queueIndices.presentFamily;

        const swapchainCreateInfo: c.VkSwapchainCreateInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = self.surface,
            .minImageCount = imageCount,
            .imageFormat = format.format,
            .imageColorSpace = format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = if (queuesAreSame) c.VK_SHARING_MODE_EXCLUSIVE else c.VK_SHARING_MODE_CONCURRENT,
            .queueFamilyIndexCount = if (queuesAreSame) 0 else 2,
            .pQueueFamilyIndices = if (queuesAreSame) null else &[_]u32{ queueIndices.graphicsFamily.?,
                queueIndices.presentFamily.? },
            .preTransform = supportDetails.capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
            .pNext = null,
            .flags = 0 
        };
        try vkDie(c.vkCreateSwapchainKHR(self.device, &swapchainCreateInfo, null, &self.swapChain));

        var neededImageCount: u32 = undefined;
        try vkDie(c.vkGetSwapchainImagesKHR(self.device, self.swapChain, &neededImageCount, null));

        // allocate all swapchain image related guys at the same time for convenience
        self.swapChainImages = try self.allocator.alloc(c.VkImage, neededImageCount);
        self.swapChainImageViews = try self.allocator.alloc(c.VkImageView, neededImageCount);
        self.swapChainFramebuffers = try self.allocator.alloc(c.VkFramebuffer, neededImageCount);

        try vkDie(c.vkGetSwapchainImagesKHR(self.device, self.swapChain, &neededImageCount, self.swapChainImages.ptr));
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
            var extent: c.VkExtent2D = .{ .width = @intCast(w), .height = @intCast(h) };

            extent.width = std.math.clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            extent.height = std.math.clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return extent;
        }
    }
    fn createSurface(self: *Self) !void {
        try vkDie(c.glfwCreateWindowSurface(self.instance, self.window, null, &self.surface));
    }
    fn createLogicalDevice(self: *Self) !void {
        const queueIndices = try self.findQueueFamilies(self.physicalDevice);

        const queueFamilies = [_]u32{ queueIndices.graphicsFamily.?, queueIndices.presentFamily.? };
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
                try queueCreateInfos.append(.{ .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = fam, .queueCount = 1, .pQueuePriorities = &@as(f32, 1.0), .pNext = null, .flags = 0 });
            }
        }

        var deviceFeatures = std.mem.zeroes(c.VkPhysicalDeviceFeatures);
        deviceFeatures.samplerAnisotropy = c.VK_TRUE;
        const deviceCreateInfo: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pQueueCreateInfos = queueCreateInfos.items.ptr,
            .queueCreateInfoCount = @intCast(queueCreateInfos.items.len),
            .pEnabledFeatures = &deviceFeatures,
            .enabledExtensionCount = deviceExtensions.len,
            .ppEnabledExtensionNames = &deviceExtensions,
            .enabledLayerCount = validationLayers.len,
            .ppEnabledLayerNames = &validationLayers,
            .pNext = null,
            .flags = 0
        };
        try vkDie(c.vkCreateDevice(self.physicalDevice, &deviceCreateInfo, null, &self.device));
        c.vkGetDeviceQueue(self.device, queueIndices.graphicsFamily.?, 0, &self.graphicsQueue);
        c.vkGetDeviceQueue(self.device, queueIndices.presentFamily.?, 0, &self.presentQueue);
    }
    fn findQueueFamilies(self: *Self, device: c.VkPhysicalDevice) !QueueFamilyIndices {
        var queueFamilyCount: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);
        const queueFamilies = try self.allocator.alloc(c.VkQueueFamilyProperties, queueFamilyCount);
        defer self.allocator.free(queueFamilies);
        c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);

        var queueIndices: QueueFamilyIndices = .{};

        for (queueFamilies, 0..) |fam, n| {
            if (queueIndices.isComplete()) break;
            if ((fam.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) > 0) {
                queueIndices.graphicsFamily = @intCast(n);
            }
            var presentSupport = c.VK_FALSE;
            try vkDie(c.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(n), self.surface, &presentSupport));
            if (presentSupport == c.VK_TRUE) queueIndices.presentFamily = @intCast( n);
        }
        return queueIndices;
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
                if (std.mem.orderZ(u8, @as([*:0]const u8, @ptrCast(&realExt.extensionName)), targetExt) == .eq) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }
    fn isDeviceSuitable(self: *Self, device: c.VkPhysicalDevice) !bool {
        const queueIndices = try self.findQueueFamilies(device);

        const extensionsSupported = try self.checkDeviceExtensionSupport(device);
        var swapChainSupported = false;
        if (extensionsSupported) {
            var details = try SwapChainSupportDetails.init(self.allocator, device, self.surface);
        defer details.deinit();
        swapChainSupported = (details.formats.len > 0 and details.presentModes.len > 0);
        }
        var supportedFeatures: c.VkPhysicalDeviceFeatures = undefined;
        c.vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
        return queueIndices.isComplete() and extensionsSupported and swapChainSupported and supportedFeatures.samplerAnisotropy == c.VK_TRUE;
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
                if (std.mem.orderZ(u8, neededName, @as([*:0]const u8, @ptrCast(&layer.layerName))) == .eq) {
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
        const createInfo: c.VkInstanceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = @intCast(extensions.items.len),
            .ppEnabledExtensionNames = extensions.items.ptr,
            .enabledLayerCount = validationLayers.len,
            .ppEnabledLayerNames = &validationLayers,
            .pNext = null,
            .flags = 0 
        };
        try vkDie(c.vkCreateInstance(&createInfo, null, &self.instance));
    }
    fn updateUniformBuffer(self: *Self, currentImage: u32) !void {
        const time: f32 = @floatCast(c.glfwGetTime());
        _ = time;
        const rotation = 1 * std.math.degreesToRadians(f32, 90.0);
        _ = rotation;
        
        const model = zm.identity();//zm.translation(10, 20, 30); //: zm.Mat = zm.matFromAxisAngle(zm.f32x4(0.0, 0.0, 1.0, 0.0), rotation);
        const view: zm.Mat = zm.lookAtLh(zm.f32x4(2.0, 2.0, 2.0, 0.0), zm.f32x4(0.0, 0.0, 0.0, 0.0), zm.f32x4(0.0, 0.0, 1.0, 0.0));
        const proj: zm.Mat = zm.perspectiveFovLh(std.math.degreesToRadians(f32, 45.0), @as(f32, @floatFromInt(self.swapChainExtent.width)) / @as(f32, @floatFromInt(self.swapChainExtent.height)), 0.1, 10.0);

        const ubo: UniformBufferObject = .{ .model = model, .view = view, .proj = proj };
        self.uniformBuffersMapped[currentImage].* = ubo;
    }
    fn drawFrame(self: *Self) !void {
        try vkDie(c.vkWaitForFences(self.device, 1, &self.inFlightFences[self.currentFrame], c.VK_TRUE, std.math.maxInt(u64)));

        var imageIndex: u32 = undefined;
        {
            const result = c.vkAcquireNextImageKHR(self.device, self.swapChain, std.math.maxInt(u64), self.imageAvailableSemaphores[self.currentFrame], null, &imageIndex);
            switch (result) {
                //c.VK_ERROR_OUT_OF_DATE_KHR, => { try self.recreateSwapChain(); return; },
                c.VK_SUCCESS, c.VK_SUBOPTIMAL_KHR => {},
                else => return vkDie(result)
            }
        }
        try vkDie(c.vkResetFences(self.device, 1, &self.inFlightFences[self.currentFrame]));

        try vkDie(c.vkResetCommandBuffer(self.commandBuffers[self.currentFrame], 0));

        try self.recordCommandBuffer(self.commandBuffers[self.currentFrame], imageIndex);

        try self.updateUniformBuffer(self.currentFrame);

        const submitInfo: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &self.imageAvailableSemaphores[self.currentFrame],
            .pWaitDstStageMask = &@as(u32, c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
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

        {
            const result = c.vkQueuePresentKHR(self.presentQueue, &presentInfo);
        if (self.frameResized) { self.frameResized = false; try self.recreateSwapChain(); return;}
            switch (result) {
                c.VK_ERROR_OUT_OF_DATE_KHR => { self.frameResized = false; try self.recreateSwapChain(); return; },
                c.VK_SUCCESS, c.VK_SUBOPTIMAL_KHR => {},
                else => return vkDie(result)
            }
        }
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
            c.vkDestroySemaphore(self.device, self.imageAvailableSemaphores[i], null);
            c.vkDestroySemaphore(self.device, self.renderFinishedSemaphores[i], null);
            c.vkDestroyFence(self.device, self.inFlightFences[i], null);
        }
        c.vkDestroyCommandPool(self.device, self.commandPool, null);
        self.cleanupSwapChain();

        c.vkDestroySampler(self.device, self.textureSampler, null);
        c.vkDestroyImageView(self.device, self.textureImageView, null);
        c.vkDestroyImage(self.device, self.textureImage, null);
        c.vkFreeMemory(self.device, self.textureImageMemory, null);


        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            c.vkDestroyBuffer(self.device, self.uniformBuffers[i], null);
            c.vkFreeMemory(self.device, self.uniformBuffersMemory[i], null);
        }
        c.vkDestroyDescriptorPool(self.device, self.descriptorPool, null);
        c.vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, null);

        c.vkDestroyBuffer(self.device, self.vertexBuffer, null);
        c.vkFreeMemory(self.device, self.vertexBufferMemory, null);
        c.vkDestroyBuffer(self.device, self.indexBuffer, null);
        c.vkFreeMemory(self.device, self.indexBufferMemory, null);

        c.vkDestroyPipeline(self.device, self.graphicsPipeline, null);
        c.vkDestroyPipelineLayout(self.device, self.pipelineLayout, null);
        c.vkDestroyRenderPass(self.device, self.renderPass, null);
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
