{
  description = "Navigate visual pipeline and model preparation";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/e72e4f299401a3689d4b3d5fc6496b11db7064eb";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ];
      eachSystem = nixpkgs.lib.genAttrs systems;
    in {
      packages = eachSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };
          source = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              !(builtins.elem (builtins.baseNameOf path)
                [ ".git" "target" ".venv" "__pycache__" ])
              && pkgs.lib.cleanSourceFilter path type;
          };
          pipeline = pkgs.rustPlatform.buildRustPackage {
            pname = "navigate-visual-pipeline";
            version = "0.1.0";
            src = source;
            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [ "-p" "navigate-visual-pipeline" ];
            cargoTestFlags = [ "-p" "navigate-visual-pipeline" "-p" "navigate-segment" ];
            meta.mainProgram = "navigate-visual-pipeline";
          };
          model = pipeline.overrideAttrs (_: {
            pname = "navigate-model";
            cargoBuildFlags = [ "-p" "navigate-model" ];
            cargoTestFlags = [ "-p" "navigate-model" ];
            meta.mainProgram = "navigate-model";
          });
        in {
          default = pipeline;
          inherit pipeline model;
          runtime = pkgs.onnxruntime;
        });
      checks = eachSystem (system:
        builtins.removeAttrs self.packages.${system} [ "default" "runtime" ]);
      apps = eachSystem (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.pipeline}/bin/navigate-visual-pipeline";
        };
      });
    };
}
