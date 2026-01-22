# Install the new testing framework

- Create a new environment `venv3_pytorch` with Python 3.8–3.9 (the current pinned
  `numpy<=1.19` and `numba==0.53.1` do not support Python 3.10+). Then install the
  package there. The editable install invokes `dso/setup.py`, so make sure `pip`
  exists inside the venv.

```bash
# If using pyenv, pin the project to 3.9.16 first:
pyenv local 3.9.16
python -V  # should show Python 3.9.16

python -m venv venv3_pytorch
source venv3_pytorch/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.19.5" --only-binary=:all: Cython
# Avoid build isolation because setup.py uses deprecated fetch_build_eggs.
python -m pip install --no-build-isolation -e ./dso

# Sanity check (should show a pip version inside the venv)
python -m pip --version

# Optional: install all extra dependencies (control/regression, etc.)
# Note: control extras (gym[box2d], pybullet) build native code on macOS.
# If you see missing C++ headers like <cmath>, make sure the SDK is set:
#   export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
#   export CFLAGS="-isysroot $SDKROOT"
#   export CXXFLAGS="-isysroot $SDKROOT"
# If you see 'fatal error: <string> file not found', ensure CLT is selected:
#   sudo xcode-select --switch /Library/Developer/CommandLineTools
# And set libc++ include paths:
#   export CPATH="$SDKROOT/usr/include/c++/v1"
# And install build tools (e.g., `brew install swig cmake`) before retrying.
python -m pip install --no-build-isolation -e "./dso[all]"

# If numpy fails to install as a wheel, clear the pip cache and retry:
#   python -m pip cache purge
#   python -m pip install "numpy==1.19.5" --only-binary=:all: Cython

# macOS note: [all] pulls control extras (gym[box2d], pybullet, stable-baselines) that
# require Xcode Command Line Tools. Install them first:
#   xcode-select --install
# If box2d-py still fails, install swig (brew install swig) or skip control extras:
#   python -m pip install --no-build-isolation -e "./dso[regression]"

# VS Code: select the venv3_pytorch interpreter before running tests/debug.
```
