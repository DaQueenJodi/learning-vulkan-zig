const std = @import("std");
const ArrayList = std.ArrayList;
const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("GLFW/glfw3.h");
});

const Allocator = std.mem.Allocator;

const SCREEN_W = 800;
const SCREEN_H = 600;


const validationLayers = [_][*c]const u8{
    "VK_LAYER_KHRONOS_validation"
};

const deviceExtensions = [_][*c]const u8 {
	c.VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
        else => @panic("invalid VK_RESULT")
    };
}


fn CreateDebugUtilsMessengerEXT(
    instance: c.VkInstance,
    createInfo: *const c.VkDebugUtilsMessengerCreateInfoEXT,
    allocator: ?*c.VkAllocationCallbacks,
    debugMessenger: *c.VkDebugUtilsMessengerEXT
) c.VkResult  {
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
) void  {
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
		var presentModeCount: u32 = undefined;
		try vkDie(c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, null));
		var presentModes = try allocator.alloc(c.VkPresentModeKHR, presentModeCount);
		try vkDie(c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.ptr));

		return .{
			.allocator = allocator,
			.capabilities = capabilities,
			.formats = formats,
			.presentModes = presentModes
		};
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
    allocator: Allocator,
    const Self = @This();
    pub fn init(allocator: Allocator) Self {
        var self: Self = undefined;
        self.allocator = allocator;
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
					.pQueueFamilyIndices = if (queuesAreSame) null else &[_]u32{indices.graphicsFamily.?, indices.presentFamily.?},
					.preTransform = supportDetails.capabilities.currentTransform,
					.compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
					.presentMode = presentMode,
					.clipped = c.VK_TRUE,
					.oldSwapchain = null,
					.pNext = null,
					.flags = 0
				};
				try vkDie(c.vkCreateSwapchainKHR(self.device, &swapchainCreateInfo, null, &self.swapChain));
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
					var extent: c.VkExtent2D = .{
						.width = @intCast(u32, w),
						.height = @intCast(u32, h)
					};
					
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
					try queueCreateInfos.append(
							.{
							.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
							.queueFamilyIndex = fam,
							.queueCount = 1,
							.pQueuePriorities = &@floatCast(f32, 1.0),
							.pNext = null,
							.flags = 0
							}
							);
				}
			}


        const deviceFeatures = std.mem.zeroes(c.VkPhysicalDeviceFeatures);
        const deviceCreateInfo: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pQueueCreateInfos = queueCreateInfos.items.ptr,
            .queueCreateInfoCount = @intCast(u32, queueCreateInfos.items.len),
            .pEnabledFeatures = &deviceFeatures,
            .enabledExtensionCount = deviceExtensions.len,
            .ppEnabledExtensionNames = &deviceExtensions,
            .enabledLayerCount = validationLayers.len,
            .ppEnabledLayerNames = &validationLayers,
            .pNext = null,
            .flags = 0
        };
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
        const createInfo: c.VkDebugUtilsMessengerCreateInfoEXT = .{
            .sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity =
                c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT  |
                c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT  |
                c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType =
                c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT     |
                c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT  |
                c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData = null,
            .pNext = null,
            .flags = 0

        };
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
        const createInfo: c.VkInstanceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = @intCast(u32, extensions.items.len),
            .ppEnabledExtensionNames = extensions.items.ptr,
            .enabledLayerCount = validationLayers.len,
            .ppEnabledLayerNames = &validationLayers,
            .pNext = null,
            .flags = 0
        };
        try vkDie(c.vkCreateInstance(&createInfo, null, &self.instance));
    }
    fn mainLoop(self: *Self) !void {
        while (c.glfwWindowShouldClose(self.window) == c.GLFW_FALSE) {
            c.glfwPollEvents();
        }
    }
    fn cleanup(self: *Self) !void {
				c.vkDestroySwapchainKHR(self.device, self.swapChain, null);
        c.vkDestroyDevice(self.device, null);
        DestroyDebugUtilsMessengerEXT(self.instance, self.debugMessenger, null);
        c.vkDestroySurfaceKHR(self.instance, self.surface, null);
        c.vkDestroyInstance(self.instance, null);
        c.glfwDestroyWindow(self.window);
        c.glfwTerminate();
    }
    fn debugCallback(
        severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
        _: c.VkDebugUtilsMessageTypeFlagsEXT,
        callbackData: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
        _: ?*anyopaque
    ) callconv(.C) c.VkBool32 {
        const Severity = enum {
            info,
            warning,
            fatal
        };
        const sev: Severity = switch (severity) {
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT, c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT => .info,
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT => .warning,
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT => .fatal,
            else => @panic("invalid severity")
        };
        const tty = std.io.tty;
        const stdout = std.io.getStdOut();
        const ttyConfig = tty.detectConfig(stdout);
        ttyConfig.setColor(stdout, 
            switch (sev) {
                .info => .blue,
                .warning => .yellow,
                .fatal => .red
            }
        ) catch @panic("failed to set color");
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
