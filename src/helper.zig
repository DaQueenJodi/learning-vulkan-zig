const std = @import("std");
const c = @import("c.zig");
const fs = std.fs;
const Allocator = std.mem.Allocator;

pub fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
	const file = try std.fs.cwd().openFile(path, .{});
	defer file.close();

	const stat = try file.stat();
	const size = stat.size;
	var buff = try allocator.alloc(u8, size);

	const readSize = try file.readAll(buff);
	std.debug.assert(readSize == size);
	return buff;
}


pub fn vec3_new(f1: f32, f2: f32, f3: f32) align(16) *c.vec3 {
    var arr = [3]f32{f1, f2, f3};
    return &arr;
}
