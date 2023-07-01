pub usingnamespace @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("GLFW/glfw3.h");

    @cInclude("stb_image.h");

    @cUndef("__SSE2__");
    @cUndef("__SSE__");
    @cDefine("CGLM_NO_ANONYMOUS_LITERALS", "");
    @cDefine("CGLM_ALL_UNALIGNED", "");
    @cDefine("CGLM_FORCE_DEPTH_ZERO_TO_ONE", "");
    @cDefine("CGLM_FORCE_RADIANS", "");
    @cInclude("cglm/cglm.h");


});

