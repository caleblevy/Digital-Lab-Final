#! /usr/bin/env bash
for i in {1,2,4,8}
do
	mkdir -p "folder_$i"
done

chmod 700 *

for i in {1,2,4,8}
do
	cd "./folder_$i"
	touch file_1
	touch file_2
	echo "4" > file_1
	cd ..
done
