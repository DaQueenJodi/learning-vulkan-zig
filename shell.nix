{ pkgs ? import <nixpkgs> {} }:
let
	myZig = 
    (pkgs.stdenv.mkDerivation rec {
      name = "zig";
      src = pkgs.fetchurl {
        url = "https://ziglang.org/builds/zig-linux-x86_64-0.11.0-dev.3737+9eb008717.tar.xz";
        sha256 = "5ddVAvpM56H7k2GbJbxTKUBUN1E47Nc9XdokKuKuEE0=";
      };
      installPhase = ''
        mkdir -p $out/bin
        mv * $out/bin
      '';
    });
in 
	pkgs.mkShell {
		buildInputs = with pkgs; [
			myZig
			vulkan-extension-layer
			vulkan-tools
			vulkan-tools-lunarg
			glfw3
			pkgconfig
			vulkan-headers
			vulkan-loader
			shaderc
			spirv-tools
		];
		VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
	}
