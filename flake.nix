{
  description = "Navigate native visual workflow";

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
                [ ".git" "target" ".venv" "__pycache__" "webapp" "node_modules" ])
              && pkgs.lib.cleanSourceFilter path type;
          };
          workflow = pkgs.rustPlatform.buildRustPackage {
            pname = "navigate-workflow";
            version = "0.1.0";
            src = source;
            cargoLock = {
              lockFile = ./Cargo.lock;
              outputHashes = {
                "rustnn-0.5.11" = "sha256-boZ7PupZWix9mJwDjUONg+ou1cr0x7M90isKKMFU+/w=";
                "webnn-graph-0.3.0" = "sha256-KvxoUJDA00r2qX3CLLHA+he8apmOJvs5tfLeN62mypI=";
                "webnn-onnx-utils-0.1.1" = "sha256-3Z6+Jnk+64ATWd0m92oJ8eudIWwwJ+/oxW5Ky+2jmsI=";
              };
            };
            cargoBuildFlags = [ "-p" "navigate-workflow" ];
            cargoTestFlags = [ "-p" "navigate-workflow" "-p" "navigate-visual-workflow" "-p" "navigate-visual-render" "-p" "navigate-visual-fusion" ];
            meta.mainProgram = "navigate-workflow";
          };
          rustnn = workflow.overrideAttrs (old: {
            pname = "navigate-workflow-rustnn";
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.protobuf ];
            cargoBuildFlags = old.cargoBuildFlags ++ [ "--features" "navigate-workflow/rustnn-coreml" ];
            cargoTestFlags = old.cargoTestFlags ++ [ "--features" "navigate-workflow/rustnn-coreml" ];
          });
          model = workflow.overrideAttrs (old: {
            pname = "navigate-model";
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.protobuf ];
            cargoBuildFlags = [ "-p" "navigate-model" ];
            cargoTestFlags = [ "-p" "navigate-model" ];
            meta.mainProgram = "navigate-model";
          });
        in {
          default = workflow;
          inherit workflow model;
          runtime = pkgs.onnxruntime;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin { inherit rustnn; });
      checks = eachSystem (system:
        builtins.removeAttrs self.packages.${system} [ "default" "runtime" ]);
      apps = eachSystem (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.workflow}/bin/navigate-workflow";
        };
      });
    };
}
