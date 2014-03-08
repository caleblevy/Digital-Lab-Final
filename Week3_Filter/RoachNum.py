#!/usr/bin/env python
import numpy as np 
from numpy.fft import *
from matplotlib import pyplot as plt
#import radiolab
import os

def Bin_Txt(Num_List):
	Txt = ''
	for I in Num_List:
		Txt += str(I)
	return Txt


def To_Binary(num,Prec,start):
	# Binary Expansion to Prec elements
	Bin_Num = [None]*Prec
	numa = abs(num)
	Approx = 0
	
	for I in range(Prec):
		if (Approx + 2**(start-I)) <= numa:
			Bin_Num[I] = 1
			Approx += 2**(start-I)
		else:
			Bin_Num[I] = 0
	
	if num < 0:
		for I in range(Prec):
			Bin_Num[I] = (Bin_Num[I]+1)%2
		J = 1
		while Bin_Num[Prec-J] == 1:
			Bin_Num[Prec-J] = 0
			J += 1
		Bin_Num[Prec-J] = 1
		
	return Bin_Num

def Bin_Txt(List):
	N = len(List)
	Text = ''
	for I in range(N):
		Text = Text + str(List[I])
	return Text

def Hex_Dict(List):
	List = List[:]
	assert len(List) < 5
	List.reverse()
	while len(List) < 4:
		List.append(0)

	List.reverse()
	Hex_num = 8*List[0] + 4*List[1] + 2*List[2] + 1*List[3]

	if Hex_num == 10:
		Hex_num = 'a'
	elif Hex_num == 11:
		Hex_num = 'b'
	elif Hex_num == 12:
		Hex_num = 'c'
	elif Hex_num == 13:
		Hex_num = 'd'
	elif Hex_num == 14:
		Hex_num = 'e'
	elif Hex_num == 15:
		Hex_num = 'f'
	else:
		Hex_num = str(Hex_num)

	return Hex_num	
		

def To_Hex(Bin_Num): # Don't mutate inputs
	Bin_Num = Bin_Num[:]

	Hex_Digs = []
	while len(Bin_Num) > 0:
		Temp_Dig = Bin_Num[-4:]
		for I in range(len(Temp_Dig)):
			Bin_Num = Bin_Num[:-1]
		Hex_Dig = Hex_Dict(Temp_Dig)
		Hex_Digs.append(Hex_Dig)
	
	Hex_Digs.reverse()
	return Hex_Digs

def Fix_18_17(num): # Find 18 fix 17 if less than 1
	assert abs(num) <= 1
	Bin_Rep = To_Binary(num,18,0)
	Hex_Rep = To_Hex(Bin_Rep)
	Hex_Rep.reverse()
	while len(Hex_Rep) < 8:
		Hex_Rep.append('0')

	Hex_Rep.reverse()

	Hex_Str = ''
	for I in range(4):
		Hex_Str = Hex_Str + '\\'
		Hex_Str = Hex_Str + Hex_Rep[2*I]
		Hex_Str = Hex_Str + Hex_Rep[2*I+1]

	return Hex_Str
