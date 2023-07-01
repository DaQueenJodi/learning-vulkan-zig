const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const mem = std.mem;
const Vertex = struct {
    x: f32,
    y: f32,
    z: f32,
};

const TextureCoord = struct {
    u: f32,
    v: f32,
};

const FaceElement = struct {
    index: usize,
    textureIndex: usize,
};

const LineFlavor = enum {
    vertex,
    textureCoord,
    vertexNormal,
    polygonFaceElement,
    comment,
    name,
    mtl,
    smoothShading,
    done
};

fn getLineFlavor(line: []const u8) !LineFlavor {

    if (line.len == 0) return .done;

    const startsWith = std.mem.startsWith;
    if (startsWith(u8, line, "vt")) return .textureCoord;
    if (startsWith(u8, line, "vn")) return .vertexNormal;
    if (startsWith(u8, line, "v")) return .vertex;
    if (startsWith(u8, line, "f")) return .polygonFaceElement;
    if (startsWith(u8, line, "#")) return .comment;
    if (startsWith(u8, line, "o")) return .name;
    if (startsWith(u8, line, "mtl")) return .mtl;
    if (startsWith(u8, line, "usemtl")) return .mtl;
    if (startsWith(u8, line, "s")) return .smoothShading;
    return error.FailedToGetLineFlavor;
}


fn parseVertex(line: []const u8) !Vertex {
    var points = mem.splitScalar(u8, line[2..], ' ');
    var buff: [4]?f32 = .{null, null, null, null};
    var buffCounter: usize = 0;
    while (points.next()) |point| : (buffCounter += 1) {
        buff[buffCounter] = try std.fmt.parseFloat(f32, point);
    }
    return .{ 
        .x = buff[0].?,
        .y = buff[1].?,
        .z = buff[2].?,
    };
}

fn parseVertexNormal(line: []const u8) !Vertex {
    // hack because im lazy
    return parseVertex(line[1..]);
}

test "parse vertex" {
    {
        const vertex = try parseVertex("v 0.0 1.0 -0.1230123");
        try std.testing.expectEqual(
            Vertex{.x = 0, .y = 1, .z = -0.1230123 },
            vertex
        );
    }
    {
        const vertex = try parseVertex("v 0 1 2");
        try std.testing.expectEqual(
            Vertex{.x = 0, .y = 1, .z = 2},
            vertex
        );
    }
}

fn parseTextureCoord(line: []const u8) !TextureCoord {
    var points = mem.splitScalar(u8, line[3..], ' ');
    var buff: [2]?f32 = .{null, null};
    var buffCounter: usize = 0;
    while (points.next()) |point| : (buffCounter += 1) {
        buff[buffCounter] = try std.fmt.parseFloat(f32, point);
    }
    return .{ .u = buff[0].?, .v = buff[1] orelse 0.0};
}

test "parse texture coordinate" {
    const coord = try parseTextureCoord("vt 0.198708 0.637534");
    try std.testing.expectEqual(
        TextureCoord{.u = 0.198708, .v = 0.637534},
        coord
    );
}

fn parseTriangle(line: []const u8) ![3]FaceElement {
    var points = mem.splitScalar(u8, line[2..], ' ');
    var elements: [3]FaceElement = undefined;
    var elementCounter: usize = 0;
    // ignore normal vertex since idk what that even is
    while (points.next()) |point| : (elementCounter += 1) {
        var indices = mem.splitScalar(u8, point, '/');
        elements[elementCounter] = .{
            .index = try std.fmt.parseInt(usize, indices.next().?, 10) - 1,
            .textureIndex = try std.fmt.parseInt(usize, indices.next().?, 10) - 1,
        };
    }
    return elements;
}


pub const ObjData = struct {
    verticies: ArrayList(Vertex),
    textureCoords: ArrayList(TextureCoord),
    triangles: ArrayList([3]FaceElement),
    const Self = @This();
    pub fn init(allocator: Allocator, path: []const u8) !Self {
        const data = try loadFile(allocator, path);
        defer allocator.free(data);
        var verticies = ArrayList(Vertex).init(allocator);
        var textureCoords = ArrayList(TextureCoord).init(allocator);
        var triangles = ArrayList([3]FaceElement).init(allocator);

        var lines = std.mem.splitScalar(u8, data, '\n');
        var lineCounter: usize = 0;
        while (lines.next()) |line| : (lineCounter += 1) {
            const flavor = try getLineFlavor(line);
            switch (flavor) {
                .vertex => try verticies.append(try parseVertex(line)),
                .vertexNormal => try verticies.append(try parseVertexNormal(line)),
                .textureCoord => try textureCoords.append(try parseTextureCoord(line)),
                .polygonFaceElement => try triangles.append(try parseTriangle(line)),
                .done => break,
                .comment => {},
                .name => {},
                .mtl => {},
                .smoothShading => {},
            }
        }
        return .{
            .verticies = verticies,
            .textureCoords = textureCoords,
            .triangles = triangles
        };
    }
    pub fn deinit(self: *Self) void {
        self.verticies.deinit();
        self.textureCoords.deinit();
        self.triangles.deinit();
    }
};


fn loadFile(allocator: Allocator, path: []const u8) ![]const u8 {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    var buffer = try allocator.alloc(u8, stat.size);
    _ = try file.readAll(buffer);
    return buffer;
}
