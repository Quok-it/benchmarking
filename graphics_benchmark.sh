#!/bin/bash
set -euo pipefail



wget https://phoronix-test-suite.com/releases/repo/pts.debian/files/phoronix-test-suite_10.8.3_all.deb
sudo apt-get install -y ./phoronix*.deb
# enter 11, 5
# sudo apt-get install unzip
phoronix-test-suite install unigine-heaven
sudo apt-get install libxrandr2
sudo apt-get install libxinerama1
phoronix-test-suite run unigine-heaven

# Need to setup graphics to forward to gpu, set up display
# Xorg (X11/X server)
# Xvfb: Xvfb: A virtual X server for running graphical applications without a display, but with only software rendering.

