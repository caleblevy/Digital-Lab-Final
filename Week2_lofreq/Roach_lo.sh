pkill adc_snaps_2014_
pkill dig_dwn_conv_2_

sleep .5
chmod 777 *

for i in {1,2,4,8}
do
	mkdir lo_mix_$i
	chmod 777 *
done

cd /boffiles/
./adc_snaps_2014_Feb_13_2111.bof &

sleep .5

PID=`pidof ./adc_snaps_2014_Feb_13_2111.bof` #`pgrep 'adc_snaps_2014_'`

cd /proc/
#echo $PID #$$
cd $PID
cd hw/ioreg/

for i in {1,2,4,8}
do
	echo -ne "\x00\x00\x00\x0$i" > lo_freq

	echo -ne "\x00\x00\x00\x01" > trig
	sleep .1
	echo -ne "\x00\x00\x00\x00" > trig
	
	cp ./cos_bram ~/lo_mix_$i
	cp ./sin_bram ~/lo_mix_$i
done

cd ~/
kill $PID

echo "lady gaga says hello"
