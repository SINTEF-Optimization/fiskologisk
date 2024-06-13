#!/usr/bin/env bash

# This script checks for directories with the current naming scheme static_param_matrix_mod<n>_spr<m>_tankvol<o> where n is number of modules, m is smolt price and o is tank volume.
# The script reads the direcotries and writes the results to a json file that contains the specs for each problem and its directory.
# This script is meant to be run in a deployment pipeline.

cd client/fiskui/public/data

file='../problems.json'

echo '[' > $file

for d in * ; do
    if [ "$print_comma" == "1" ]; then echo "," >>$file; fi
    print_comma=1
    modules=`echo $d | sed -e 's/.*_mod\([0-9]*\)_spr.*/\1/'`
    smolt_price=`echo $d | sed -e 's/.*_spr\([0-9]*\)_tank.*/\1/'`
    tank_volume=`echo $d | sed -e 's/.*_tankvol\([0-9]*\)/\1/'`

    echo '{' >>$file
    echo '"dir": "'$d'",' >>$file
    echo '"modules": '$modules',' >>$file
    echo '"smolt_price": '$smolt_price',' >>$file
    echo '"tank_volume": '$tank_volume >>$file
    echo '}' >>$file
done

echo ']' >> $file