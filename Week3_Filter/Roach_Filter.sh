pkill adc_snaps_2014_
pkill dig_dwn_conv_2_

sleep .5
chmod 777 *


for i in {0,1,2,4,8}
do
	mkdir lo_Filter_$i
	chmod 777 *
done

cd /boffiles/
./dig_dwn_conv_2_2014_Feb_25_1332.bof &
sleep .5

PID=`pidof ./dig_dwn_conv_2_2014_Feb_25_1332.bof` #`pgrep 'adc_snaps_2014_'`

cd /proc/
#echo $PID #$$
cd $PID
cd hw/ioreg/

for i in {0,1,2,4,8}
do
	echo -ne "\x00\x00\x00\x0$i" > lo_freq
	 
	echo -ne "\x00\x00\x40\x00" > coeff_real0
	echo -ne "\x00\x03\xe5\x7e" > coeff_real1
	echo -ne "\x00\x03\xc0\x00" > coeff_real2
	echo -ne "\x00\x00\x9a\x82" > coeff_real3
	echo -ne "\x00\x01\x40\x00" > coeff_real4
	echo -ne "\x00\x00\x9a\x82" > coeff_real5
	echo -ne "\x00\x03\xc0\x00" > coeff_real6
	echo -ne "\x00\x03\xe5\x7e" > coeff_real7

	echo -ne "\x00\x00\x00\x01" > trig
	sleep .1
	echo -ne "\x00\x00\x00\x00" > trig
	
	cp ./ddc_imag_bram ~/lo_Filter_$i
	cp ./ddc_real_bram ~/lo_Filter_$i
done

cd ~/
kill $PID

echo "lady gaga says hello"
