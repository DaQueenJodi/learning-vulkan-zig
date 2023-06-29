{ pkgs ? import <nixpkgs> {} }:
let
	myZig = 
    (pkgs.stdenv.mkDerivation rec {
      name = "zig";
      src = pkgs.fetchurl {
        url = "https://ziglang.org/builds/zig-linux-x86_64-0.11.0-dev.3859+88284c124.tar.xz";
        sha256 = "6Uj9l5/97bxArsOHzdvJsL1Xjf0sb3e6R4N5uaKWSV0=";
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
			cglm
		];
		VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
	}
