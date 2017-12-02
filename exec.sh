#!/bin/bash

ODIR=$1
IDX=$2
MODE=$3
CVN=$4
OPT=$5

# change this
WALS_CSV=$HOME/research/bayes-autologistic/data/wals/language.csv

unset PYTHONUSERBASE
unset PYTHONPATH

# virtualenv Python3
. $HOME/local3/bin/activate

if [ -z "$MODE" ]; then
    MODE=param
fi
if [ -z "$CVN" ]; then
    CVN=-1
fi
if [ "$CVN" -ge 0 ]; then
    LOGFILE=$ODIR/$IDX.$CVN.log
    DONEFILE=$ODIR/$IDX.$CVN.done
else
    LOGFILE=$ODIR/$IDX.log
    DONEFILE=$ODIR/$IDX.done
fi


mkdir -p $ODIR

nice -19 python ./scripts/experiments.py autologistic exp $OPT -l $WALS_CSV -o $ODIR --hg $ODIR/horizontal.pkl --vg $ODIR/vertical.pkl $IDX $IDX $MODE > $LOGFILE 2>&1 && touch $DONEFILE
