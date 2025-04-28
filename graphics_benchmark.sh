#!/bin/bash

git clone https://github.com/phoronix-test-suite/phoronix-test-suite/
cd phoronix-test-suite
sudo ./install-sh
sudo apt-get install php-cli php-xml
# enter 11, 5


phoronix-test-suite benchmark fluidx3d