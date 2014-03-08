#! /usr/bin/env bash
pkill adc_snaps_2014_
pkill dig_dwn_conv_2_

sleep 2

cd /boffiles/
./adc_snaps_2014_Feb_13_2111.bof &

sleep 2

PID=`pidof ./adc_snaps_2014_Feb_13_2111.bof` #`pgrep 'adc_snaps_2014_'`
cd /proc/
echo $PID #$$
cd $PID
cd hw/ioreg/

pwd 
ls
echo -ne "\x00\x00\x00\x01" > trig
sleep 1
echo -ne "\x00\x00\x00\x00" > trig

#echo 
