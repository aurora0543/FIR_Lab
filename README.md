# FIR_Lab

## Taksk 1
Create a function which calculates and returns the FIR filter **coefficients** analytically (= using **sinc** functions) for a **combined highpass and bandstop** filter. 
* The function should automatically decide how many coefficients are required. 
* The function arguments should be (a) the **sampling rate** and (b) the **cutoff frequencies**. 
* Decide which **cutoff frequencies** are needed and provide explanations by referring to the **spectra and/or fundamental frequencies** of the ECG

## Task 2
Create an efficient Python FIR filter class which implements an FIR filter and has a method of the form value dofilter(self,value) where both the value argument and return value are scalars and not vectors (!) so that it can be used in a real-time system. The constructor of the class takes the coefficients as its input: 

>class FIRfilter: \
>def __init__(self,_coefficients): \
>your code here \
>def dofilter(self,v): \
>your code here \
>return result 

## Task 3
Use an adaptive LMS filter to filter out DC and 50Hz by providing it with a 50Hz sine wave with DC as reference.
* Note that both the amplitudes for the 50Hz and DC references scale with the learning rate. 
* Make appropriate choices for the amplitudes and the learning rate so that both DC and 50Hz are removed. 
* Add an adaptive LMS filter method to your FIR filter class (from 2.) and name it: “doFilterAdaptive(self,signal,noise,learningRate)” which returns the cleaned up ECG. 
* As before also this function must receive only scalars (i.e. sample by sample) and return a scalar. Plot and compare the result from the adaptive filter and that from the FIR filter design.

## Task 4
ECG heartbeat detection: The task is to detect R-peaks in the noisy ECG recording. Use the FIR filter from 2. as a matched filter and use an R-peak as a template from the noise-free ECG. Plot the momentary heart rate (i.e. inverse intervals between R-peaks) against time. 