#!/bin/bash

function make_pdf {
    mkdir -p latex_out_dir
    cd latex_out_dir
    latexmk -xelatex -f -shell-escape -interaction=nonstopmode -file-line-error ../main_file.tex
    cd ..
}

while true
do
    inotifywait -q -e modify main_file.tex
    make_pdf
    cp latex_out_dir/main_file.pdf ryan_greenblatt_hw_02.pdf
done

