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

Filter both ECG recordings with the above FIR filter class using the coefficients from 1. Simulate real-time processing by feeding the ECGs sample by sample into your FIR filter class. Make sure that the ECGs look intact and that they are not distorted (PQRST intact). Provide appropriate plots in a vector-graphics format.