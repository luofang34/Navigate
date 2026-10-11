{
  description = "Navigate native visual workflow";

  inputs.nixpkgs.url = "https://releases.nixos.org/nixpkgs/nixpkgs-26.11pre1027958.c4013e501c04/nixexprs.tar.xz";

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
            cargoTestFlags = [ "-p" "navigate-workflow" "-p" "navigate-visual-workflow" ];
            meta.mainProgram = "navigate-workflow";
          };
        in {
          default = workflow;
          inherit workflow;
        });
      apps = eachSystem (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.workflow}/bin/navigate-workflow";
        };
      });
    };
}
