#! /usr/bin/env bash

# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
CURRENT_DIR="$(pwd)"

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
