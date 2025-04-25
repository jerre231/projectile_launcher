# shell.nix

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {
  buildInputs = [
    pkgs.python39
    pkgs.python39Packages.numpy
    pkgs.python39Packages.scipy
    pkgs.python39Packages.matplotlib
    pkgs.python39Packages.pip
  ];

  # Criação de um ambiente Python virtual
  shellHook = ''
    export PYTHONPATH=$(pwd)/lib/python${pkgs.python39.version}/site-packages
    echo "Ambiente configurado para Python 3.9 com as bibliotecas necessárias."
  '';
}
