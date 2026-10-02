#!/bin/bash
# Fetch render assets not stored in git (148MB soundfont + fluidsynth build).
set -e
cd "$(dirname "$0")/.."

echo "== soundfont (FluidR3 GM, from Debian archive) =="
mkdir -p sf2
curl -sL --max-time 600 -o /tmp/fluid-soundfont.tar.gz \
  "http://deb.debian.org/debian/pool/main/f/fluid-soundfont/fluid-soundfont_3.1.orig.tar.gz"
tar -xzf /tmp/fluid-soundfont.tar.gz -C sf2
mv "sf2/fluid-soundfont-3.1/FluidR3_GM.sf2" sf2/FluidR3_GM.sf2
rm -rf sf2/fluid-soundfont-3.1 /tmp/fluid-soundfont.tar.gz
ls -la sf2/FluidR3_GM.sf2

echo "== fluidsynth binary =="
echo "If not already installed (brew fails on Tier-3 Intel), build from source:"
echo "  curl -sL https://github.com/FluidSynth/fluidsynth/archive/refs/tags/v2.6.1.tar.gz | tar -xz -C /tmp"
echo "  cd /tmp/fluidsynth-2.6.1 && cmake -B build -DCMAKE_BUILD_TYPE=Release \\"
echo "    -DCMAKE_CXX_FLAGS='-nostdinc++ -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1' \\"
echo "    -Denable-portaudio=false -Denable-libsndfile=ON -Denable-aufile=ON -Denable-network=OFF \\"
echo "    -DCMAKE_INSTALL_PREFIX=\$PWD/../thirdparty/fluidsynth"
echo "  cmake --build build -j && cmake --install build"
echo "Done."
