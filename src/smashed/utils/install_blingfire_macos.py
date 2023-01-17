#! /usr/bin/env python3

from subprocess import call

BASH_SCRIPT = """
#! /usr/bin/env bash

# path to the current directory
CURRENT_DIR="$(pwd)"

# remove any existing blingfire installation
pip uninstall -y blingfire 2>/dev/null

# clone blingfire repo to a temp directory
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
git clone "https://github.com/microsoft/BlingFire"
cd blingfire

# build blingfire
mkdir Release
cd Release
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release ..
make -j 4
cd ..

# copy freshly compiled blingfire to the python bindings directory
cp -rf Release/* dist-pypi/blingfire/

# build & install the python bindings
cd dist-pypi
python setup.py sdist bdist_wheel
pip install --force-reinstall dist/blingfire-*-py3-none-any.whl

# cleanup
cd $CURRENT_DIR
rm -rf $TMP_DIR
"""


def main():
    call(BASH_SCRIPT.strip(), shell=True)


if __name__ == "__main__":
    main()
