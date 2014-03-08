pkill adc_snaps_2014_
pkill dig_dwn_conv_2_

mkdir -p "adc_SampleSig"
chmod 777 *

for i in {1,2,4,8}
do
        mkdir -p "adc_lo_freq_$i"
        chmod 777 *
done

sleep .5

cd /boffiles/
./adc_snaps_2014_Feb_13_2111.bof &


sleep .5

PID=`pidof ./adc_snaps_2014_Feb_13_2111.bof` #`pgrep 'adc_snaps_2014_'`
cd /proc/
#echo $PID #$$
cd $PID
cd hw/ioreg/

echo -ne "\x00\x00\x00\x01" > trig
sleep .1
echo -ne "\x00\x00\x00\x00" > trig

cp "./adc_bram" ~/adc_SampleSig/

for i in {1,2,4,8}
do
	echo -ne "\x00\x00\x00\x0$i" > lo_freq # Set lo_freq

	echo -ne "\x00\x00\x00\x01" > trig
	sleep .1
	echo -ne "\x00\x00\x00\x00" > trig
	
	cp ./adc_bram ~/adc_lo_freq_$i
	cp ./cos_bram ~/adc_lo_freq_$i
	cp ./sin_bram ~/adc_lo_freq_$i
done

# Begin dig sample

cd ~/
kill $PID
sleep .5

for i in {1,2,4,8}
do
	mkdir -p "dig_lo_freq_$i"
	chmod 777 *
done
cd /boffiles/

./dig_dwn_conv_2_2014_Feb_25_1332.bof &

sleep .5


PID=`pidof ./dig_dwn_conv_2_2014_Feb_25_1332.bof`

cd /proc/
cd $PID

cd hw/ioreg/


for i in {1,2,4,8}
do
	echo -ne "\x00\x00\x00\x0$i" > lo_freq # Set lo_freq
	
	echo -ne "\x00\x00\x40\x00" > coeff_real0
	echo -ne "\x00\x03\xe5\x7d" > coeff_real1
	echo -ne "\x00\x03\xc0\x00" > coeff_real2
	echo -ne "\x00\x00\x9a\x83" > coeff_real3
	echo -ne "\x00\x01\x40\x00" > coeff_real4
	echo -ne "\x00\x00\x9a\x83" > coeff_real5
	echo -ne "\x00\x00\x40\x00" > coeff_real6
	echo -ne "\x00\x03\xc0\x00" > coeff_real7
	
	
	echo -ne "\x00\x00\x00\x01" > trig
	sleep .1
	echo -ne "\x00\x00\x00\x01" > trig

	cp ./ddc_* ~/dig_lo_freq_$i

done

kill $PID

echo "lady gaga says hello"
