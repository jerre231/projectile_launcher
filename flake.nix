{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [];
        };
        
        pythonVersion = pkgs.python313;
        
        pythonPackages = python-packages: with python-packages; [
          numpy
          pandas
          matplotlib
	  pygame
        ];
        
        pythonEnv = pythonVersion.withPackages pythonPackages;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
    	    pkgs.SDL2  # Required for Pygame
    	    pkgs.SDL2_image
    	    pkgs.SDL2_mixer
    	    pkgs.SDL2_ttf
          ];
          
          shellHook = ''
            echo "Python `${pythonVersion}/bin/python --version` environment initialized"
          '';
        };
      }
    );
}
