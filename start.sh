#!/bin/bash
# Programa geral
read -p 'Freq_signal: ' freqsingvar
read -p 'Freq_sinc: ' freqsincvar
read -p 'Num samples: ' num_samplesvar
/home/rafael/DoA/librtlsdr-2freq/build/src/rtl_sdr -h $freqsingvar -f $freqsincvar -d 0 -n $num_samplesvar sinalL.bin & /home/rafael/DoA/librtlsdr-2freq/build/src/rtl_sdr -h $freqsingvar -f $freqsincvar -d 1 -n $num_samplesvar sinalR.bin
echo $uservar

