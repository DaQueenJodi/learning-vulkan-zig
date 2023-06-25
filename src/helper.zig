const std = @import("std");
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

