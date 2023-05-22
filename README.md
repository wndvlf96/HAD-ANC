# HAD-ANC: A Hybrid System Comprising an Adaptive Filter and Deep Neural Networks for Active Noise Control ANC system  

## Announcement

This GitHub account was created to comply with INTERSPEECH's double-blind regulations, and it will be relocated to a different address once the acceptance results are published.


## Model for HAD-ANC
### Gated convolutional recurrent network (GCRN) for HAD-ANC
Description of our PyTorch implementation of GCRN.
  
We uploaded 2 pre-trained GCRNs for HAD-ANC.
  
$GCRN_{1}$ follows the adaptive filter to handle nonlinear distortion by reducing the residual error of linear filtering and models the reverse of both loudspeaker and secondary path.
  
$GCRN_{2}$ models the loudspeaker and secondary path to force the adaptive filter to estimate only the primary path.

## Subscription of train and validation set preparation  
### 1. Download 16 kHz noise signal
Download DEMAND https://zenodo.org/record/1227121#.ZAxCu3ZByUk
  
and MS-SNSD https://github.com/microsoft/MS-SNSD.
  
Note that exclude labeled as "babble" signal in [noise train](https://github.com/microsoft/MS-SNSD/tree/master/noise_train) and [noise_test](https://github.com/microsoft/MS-SNSD/tree/master/noise_test) of MS-SNSD.

### 2. Create development set
Split all signals into 6-second audio clips and index them in order.
  
Then normalize all 15639 clips.

### 3. Create validation set
Select the audio clips indexed by a multiple of 10 in the development set.
  
Multiply each of them by random numbers between 0.3 and 1.0.

### 4. Create train set
Once the validation sets consisting of 1,563 signals are created, the remaining audio clips from the development set compose the train set.

## Acknowledge
In our process, we study an important project:
  
https://github.com/JupiterEthan/GCRN-complex.
  
Thanks for authors to open source code!
