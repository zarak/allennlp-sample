{ pkgs ? import <nixpkgs> {} }:

pkgs.python38Packages.buildPythonApplication {
  pname = "myapp";
  src = ./.;
  version = "0.1";
  propagatedBuildInputs = [ pkgs.python38Packages.pytorch ];
}
