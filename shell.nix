let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "simpleEnv";
    buildInputs = with pkgs; [
    # basic python dependencies
      python38
      python38Packages.numpy
      python38Packages.scikitlearn
      python38Packages.scipy
      python38Packages.matplotlib
    # a couple of deep learning libraries
      python38Packages.pytorchWithCuda
    ];
   shellHook = ''
      '';
  }
