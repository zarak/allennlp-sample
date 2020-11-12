let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "simpleEnv";
    buildInputs = with pkgs; [
      python38
      python38Packages.pytorch
    ];
   shellHook = ''
      '';
  }
