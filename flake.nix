{
  description = "Fastmail email downloader and PDF exporter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        pythonPkgs = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPkgs.pip
            pythonPkgs.requests
            pythonPkgs.click
            pythonPkgs.weasyprint
            pythonPkgs.jinja2
            pythonPkgs.python-dateutil
            pythonPkgs.pypdf
          ];
        };
      });
}
