#!/bin/bash

set -x
ls /texdata
cp -r /texdata/. .
git clean -df
git reset --hard
git status
# GENERATE DATA
cd code/nmm/scripts || exit
export PYTHONPATH=$PYTHONPATH:/data:/data/code/nmm:.

ls -la /data/thesis/data/methodology
python3 methodology.py
ls -la /data/thesis/data/methodology

# COMPILE PDF
#cd thesis
#cat main.tex
#xelatex -shell-escape -8bit -synctex=1 -interaction=nonstopmode main.tex
#xelatex -shell-escape -8bit -synctex=1 -interaction=nonstopmode main.tex
#cp main.pdf /out/document.pdf
#ls /out