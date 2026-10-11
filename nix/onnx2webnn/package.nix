{ pkgs }:
let
  converterSource = pkgs.fetchFromGitHub {
    owner = "rustnn";
    repo = "onnx2webnn";
    rev = "8f06e78b51e6d24f0d96dbb2003ddbfa7bfe3fe4";
    hash = "sha256-LSpgX1IhTF0m6ZT8airxejL/eLr0D2UbPAKvIaLRHI8=";
  };
  rustnnSource = pkgs.fetchgit {
    url = "https://github.com/rustnn/rustnn";
    rev = "7a273752263989a4bdcb4cf48e999575e2a59842";
    hash = "sha256-boZ7PupZWix9mJwDjUONg+ou1cr0x7M90isKKMFU+/w=";
  };
  runtime = "${pkgs.onnxruntime}/lib/libonnxruntime${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
in pkgs.rustPlatform.buildRustPackage {
  pname = "navigate-onnx2webnn";
  version = "0.1.0";
  src = pkgs.runCommand "source" {} ''
    mkdir -p $out
    cp -R ${converterSource} $out/onnx2webnn
    cp -R ${rustnnSource} $out/rustnn
  '';
  sourceRoot = "source/onnx2webnn";
  patches = [ ./conv2d-api.patch ];
  postPatch = ''
    cp ${./Cargo.lock} Cargo.lock
    chmod -R u+w ../rustnn
    patch -d ../rustnn -p1 < ${./explicit-cpu.patch}
  '';
  cargoLock = {
    lockFile = ./Cargo.lock;
    outputHashes = {
      "webnn-graph-0.3.0" = "sha256-KvxoUJDA00r2qX3CLLHA+he8apmOJvs5tfLeN62mypI=";
      "webnn-onnx-utils-0.1.1" = "sha256-3Z6+Jnk+64ATWd0m92oJ8eudIWwwJ+/oxW5Ky+2jmsI=";
      "huggingface-hub-0.0.1" = "sha256-DzaE2BH36cxyC9TpeLco95sPfMsnZKHdeje2k5TkYtY=";
    };
  };
  nativeBuildInputs = [ pkgs.protobuf pkgs.cmake pkgs.pkg-config pkgs.makeWrapper ];
  buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];
  env.ORT_DYLIB_PATH = runtime;
  cargoTestFlags = [ "--lib" ];
  postInstall = ''
    wrapProgram $out/bin/onnx2webnn --set-default ORT_DYLIB_PATH ${runtime}
  '';
  meta = {
    description = "Pinned ONNX to WebNN conversion and numerical validation";
    mainProgram = "onnx2webnn";
    license = pkgs.lib.licenses.asl20;
  };
}
