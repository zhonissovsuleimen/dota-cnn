{
  description = "Development shell for dota-cnn project";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          # The upstream torch wheel pulls in source-built NVSHMEM. Use the
          # matching precompiled NVIDIA wheel to avoid compiling CUDA code.
          nvshmemWheel = {
            x86_64-linux = {
              url = "https://files.pythonhosted.org/packages/5d/7b/2ab033584a3339552472ac8d79543c503a0e06dd0d082448b06697e7f716/nvidia_nvshmem_cu13-3.6.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
              hash = "sha256-QAGqvHLq0y7MPJrdPGeBvvy3Gty+KG1/WVYELmhmjHA=";
            };
            aarch64-linux = {
              url = "https://files.pythonhosted.org/packages/83/6b/4c642d2cce57c8bd32043b4a608b6acc8cd0579c2f53af9a7ef9e1e1ccca/nvidia_nvshmem_cu13-3.6.5-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl";
              hash = "sha256-lKoBSUkL5ljPlZRc6fFrqQqQ7b1qVkNm+Zbq10lvLyI=";
            };
          };
          cudaPackages = pkgs.cudaPackages_13_0 // {
            libnvshmem = pkgs.stdenvNoCC.mkDerivation {
              pname = "libnvshmem-bin";
              version = "3.6.5";
              src = pkgs.fetchurl nvshmemWheel.${system};
              nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook pkgs.autoAddDriverRunpath ];
              buildInputs = [
                pkgs.stdenv.cc.cc.lib
                pkgs.rdma-core
                pkgs.libfabric
                pkgs.pmix
                pkgs.mpi
                pkgs.ucx
              ];
              runtimeDependencies = [
                "${pkgs.lib.getLib pkgs.rdma-core}/lib/libibverbs.so.1"
                "${pkgs.lib.getLib pkgs.cudaPackages_13_0.nccl}/lib/libnccl.so.2"
                "${pkgs.lib.getLib pkgs.cudaPackages_13_0.gdrcopy}/lib/libgdrapi.so"
              ];
              dontConfigure = true;
              dontBuild = true;
              dontStrip = true;
              unpackPhase = ''
                runHook preUnpack
                unzip -q "$src"
                runHook postUnpack
              '';
              installPhase = ''
                runHook preInstall
                mkdir -p "$out"
                cp -r nvidia/nvshmem/. "$out/"
                mkdir -p "$out/share/licenses/libnvshmem"
                cp nvidia_nvshmem_cu13-*.dist-info/licenses/License.txt "$out/share/licenses/libnvshmem/"
                runHook postInstall
              '';
              meta = {
                license = pkgs.lib.licenses.nvidiaCudaRedist;
                sourceProvenance = [ pkgs.lib.sourceTypes.binaryNativeCode ];
                platforms = [ "x86_64-linux" "aarch64-linux" ];
              };
            };
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.ffmpeg
              pkgs.rustc
              pkgs.cargo
              pkgs.rustfmt
              pkgs.clippy
              pkgs.rust-analyzer
              (pkgs.python3.withPackages (ps:
                if pkgs.stdenv.hostPlatform.isLinux then [
                  # Match the CUDA 13.0 runtime used by the Linux wheels.
                  (ps.torch-bin.override { inherit cudaPackages; })
                  (ps.torchvision-bin.override {
                    inherit cudaPackages;
                    torch-bin = ps.torch-bin.override { inherit cudaPackages; };
                  })
                ] else [
                  ps.torch
                  ps.torchvision
                ]))
            ];
          };
        });
    };
}
