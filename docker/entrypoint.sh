#!/bin/bash --login
set -e
export PYOPENGL_PLATFORM='egl'
exec "$@"


