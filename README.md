<img src="https://techsalesgroup.files.wordpress.com/2016/10/ironhack-logo-negro1.jpg" width="100" height="100" />
#Instrument Isolation with Deep Learning
### By Hernán Sosa



I'm a guitar player. Not a very good one, so I've always struggled on learning new songs. Doing it sometimes it's hard, because of the complexity of the song or also because I cannot listen properly to the guitar in the mix. Only if I had a way of isolate it... 

Wait! What if we make a program to do so? If it works for me, maybe it'll work for other people!

That's the reason for this project. The intention is to be able to take a full mixed song and extract only the desired instrument, also as an audio file. The thing is that audio is like colors in some way: once you mix them, there's no way of turning back. Or that's the theory.

Actually, we have plenty of tools to analyze and visualize audio. All this tools imply image processing, and image processing has suffered an exponential growth on its capabilities over the last years. And how can we process audio with images?

The spectrogram is a representation of an audio sample in which all the information for audio reproduction is taken into account. We'll for with them. But first, let's begin from the beginning:

# Import audio files

We'll get our train and test data directly from raw mp3 files. We'll get 1D numpy arrays and the constant sample rate (previously defined) by importing it with the `librosa.load` function. For this matter, in order to easen the task, we created a little function `import_audio_file()` which only returns the audio signal, since the sample rate is always the same 22050 samples/second. We're defining this constant to be used throughout the project:

 -  SR = 22050

With IPython we can easily reproduce our audio files:
 
 `ipd.Audio(sample, rate= SR)`
 
# Quick introduction to audio

In order to understand our approach we'll need to know the basics of digital audio signals:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/Single%20frequency%20waveform.png?raw=true" width="1000" height="500" />

This is the aspect of a single frequency waveform. The y-axis represent its amplitude (energy in an instant of time). Because sound is a vibration, the amplitude range goes from -1 to 1, but in both cases its energy is equal.

The dots represent the samples (single value), and the x-axis represent the time.

In this signal, a single second contains 22050 samples.

Our goal is to identify this frequencies in order to capture them. The frequency is measured by the time it completes a full cycle (Hz). By looking into the plot, this might seem an easy task.

But in RealWorld™, humans can hear frequencies from 20 to 20KHz  *(yea, right!)* and all this frequencies are mixed up in the same axis of time:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/Waveform%20of%20a%20snare%20hit.png?raw=true" width="1000" height="500" />

In this (real) audio waveform we see a mixture of nearly 20000 frequencies on the same axis. How to identify them?

Spectrogram is a visual representation of the audio which, through a series of processes, collects information for all audio characteristics (frequency, amplitude and phase) into a single image.

# Creating Spectrogram

Our approach for this model will be transforming our 1D audio signal into a 2D image without losing **much** information (you'll understand the bold statement in a moment). This can be done through the following process:

## Short-Time Fourier Transform

The Short-time Fourier transform (STFT), is a Fourier-related transform used to determine the sinusoidal frequency and phase content of local sections of a signal as it changes over time [(Wikipedia)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform#:~:text=The%20Short%2Dtime%20Fourier%20transform,as%20it%20changes%20over%20time.).

For a full understanding of the Fourier Transform you can check this [video by 3blue1brown](https://www.youtube.com/watch?v=spUNpyF58BY).

In a few words, we'll be recognizing and grouping each frequency in our signal by a time window (hop). This time window is a nº of samples (usually 512) the function needs to properly 'print' the value of each frequency in a 2D plane. In the process, we lose some information about each individual sample phase and amplitude (hence the **some** statement), but for our prupose this loss is acceptable since we'll be using a shorter time window of 512 samples.

The result is that we get something similar to the following plot, in which the x-axis is now the frequency, and the y-axis is the computed sum of the amplitude (power) of each one: 

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/Spectrum%20of%20audio%20signal.png?raw=true" width="1000" height="500" />

This representation is called the **spectrum** of a sound. Represents the timbrical characteristics of a sound in a time window.

Note that the y-axis is no longer negative. This is because in this transformation the phase of each frequency is computed in the time window. This is important because in this representation the phase is taken in account for each 'frame' (or group of samples) in a 2D plane so we'll be able to reproduce it in the following step.

## Spectrogram

The spectrogram is a representation of the sound in a 2D plane, in which amplitude and frequency are embodied over time. Taking the spectrum of a sound, we'll map the amplitude of each frequency in a color-scale:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/Colormapping.png?raw=true" width="1000" height="500" />

By generating a series of spectrums, stacking them in the X-axis and plotting its color value on the y-axis, we end up with a spectrogram:

<img src="https://i.makeagif.com/media/3-10-2016/I0DtmH.gif" width="500" height="200" />

And this is the looks of our static spectrogram:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/Spectrogram.png?raw=true" width="1000" height="500" />


Now, we've managed to condense our mp3 file onto an static 2D image. With this solution, we'll try to create a neural network capable of comparing the spectrogram of a mixed song, and generate the spectrogram of the instrument we want to extract.

The last step will be adapting the frequencies into mel-scale. The errors produced in the high frequencies can be easily neglected, but become extremely problematic in low frequencies due to human sensibility to this range of frequencies. This is why we transform the scale giving more space (think in a log scale) to the low frequencies to be more represented (thus, more accurate).

Additional info can be seen [here](http://kom.aau.dk/group/04gr742/pdf/MFCC_worksheet.pdf).

# Audio-to-Spectrogram-to-Audio

Before processing the images into our convolutional neural network, we want to test the loss of the quality we experiment on the conversion from audio to image to audio. To be able to train our model, we need the shape of our image arrays to be divisible by 32.

After lots of experimentation, we find out thet the array shape wich works better for our model, in terms of performance, quality and efficiency is **(512 mel-bins, 640 time-window)**.
For this matter, the number of samples we need for having a nearly 15sec sample and respect this structure is **327500 samples**.

Transformation of our audio files in Spectrogram 2D array format:

    def mel_15s(audio, mels= 512, SAMPLES = 327500, db= True):

    '''
    Function that transforms an audio file to a spectrogram with a fixed ~15sec window.
    Input =     type Array(1,)
    Output =    type Array(mels, 640)
    
        args:
          audio = selected audio file to process
          
          mels = nº of bins in which the frequency scale is transformed
          
          SAMPLES = Constant number of samples of the length of the audio array
          
          db = If True, applies scale on amplitude values (10*log10(x)). Recommended for
               better spectrogram visualization.
    '''
   
   
We'll define a function to convert mel-spectrograms to audio. This function uses the Griffin-Lim algorith for phase reconstruction. You can learn more about this algorithm [here](https://www.groundai.com/project/deep-griffin-lim-iteration/1).

We can see that we have some loss in the audio signal due to the conversion:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/mel-reconstruction.png?raw=true" width="1000" height="500" />


For now, this loss of quality is something we can afford. We'll proceed to create a function to automate the process of getting mel-spectrograms for some files:

# Prepare data for model

As we said earlier, we need our data shape to be divisible by 2 due to the reduction the CNN is performing.

In order to automate the data import in our particular case, we define a function to load all the mp3 files in our organized multitrack folders. This way, we can also arrange them in a pandas dataframe for further organization:

def mix_inst_tracks(filepath = '/Users/Sosa/Repos/project-final/your-project/Songs', mels= 512, song_sections= 1,  n_songs = 1, instrument = 'Drums', start_sec= 30, SAMPLES =327500):   
    ''' 
    Function that looks up in a structured folder (path/song/audiofile.mp3) and returns 
    n song sections in mel-spectrogram array.  
    Input = type String. Path for song folders
    Output = mix_array, inst_array type Array. Shape (n_songs*song_sections, mels, 640, 1)
    
        args:
            filepath = type STR. Path where the song folders are stored
              mels = type INT. nº of bins in which the frequency scale is transformed
              song_sections = type INT. nº of spectrogram arrays to generate for each song (evenly divided in song lenght)
              n_songs = type INT. nº of song folders to generate arrays in
              instrument = type STR. Instrument to process in inst_array. This string must be the name of the file
              SAMPLES = type INT. Constant number of samples to get 15s sample and respect model structure
              start_sec = type INT. Set a start second to extract the sample of (to avoid intros)
              
# U-NET model

UNet a neural network model originally designed for biomedical image segmentation. It's pixel-accuracy results are because the model classifies each pixel individually. 

Another reason for chosing this model is that the input and output share the same size. 

It shrinks the samples and them reconstructs the signal. You can learn more about this model [here](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)

<img src="https://miro.medium.com/max/1400/1*f7YOaE4TWubwaFF7Z1fzNw.png" width="1000" height="500" />

In just a few cycles of training, we can see some results:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/Learning%20curve.png?raw=true" width="500" height="500" />

And this is our first prediction:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/15s_predicted_sample3.png?raw=true" width="600" height="600" />

Compared with our desired output:

<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/0Test%2015s%20sample.png?raw=true" width="600" height="600" />


Well, it might seem that our model performs quite well! But we're **wrong**.

Our visual perception is tricky. If a pixel is displaced we wouldn't even notice. But, for audio reconstruction, a single pixel implies hundreds of frequencies and intensities, and to our perception, this amout of errors is A LOT!

As we can see, after the conversion to audio, our signal is not quite similar to the original one:


<img src="https://github.com/H89Sosa/project-final/blob/master/imgs/Readme/0Waveform%20comparison%20sample.png?raw=true" width="100" height="500" />

# Conclusions

Results speak for themselves. Our audio quality is not the desired one. But we'll get there, eventually.

We think we can improve our predictions by:

   - **More accurate spectrograms**. Since they're not perfect on transformation, we'll lose a lot of valuable information on them. Getting spectrograms with a much higher resolution can result in way better audio files.
   - **More samples**. Images are resource intensive, so we'll have to figure a way to be able to train our model with a lot more samples and not die trying.
   - **Better hyperparameter tuning**. Time was an obstacle for this project, so we had to explore the tuning just on the surface. Using techniques like cross validation, pipelines, k_fold etc could lead into better performances.
   - **Model exploring**. Similary, adjusting the structure of the model itself is crucial for better results.
   - **Better spectrogram-to-audio conversion**. The Griffin-lin algorithm for audio reconstruction is a deep and complex mathematical algorithm. We will study it better for a clean application.
   

Thanks for the long-lecture. I hope you enjoyed the trip!
   
   
# Resources

[WAVE-U-NET : A multi-scale neural network for end-to-end audio source separation](https://arxiv.org/pdf/1806.03185.pdf)

[Voice separation with deep U-NET Convolutional Networks](https://openaccess.city.ac.uk/id/eprint/19289/1/7bb8d1600fba70dd79408775cd0c37a4ff62.pdf)

[UNet Guide line by line](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)

[Segmenting images with Keras](https://www.depends-on-the-definition.com/unet-keras-segmenting-images/)

[Batch normalization in Neural Networks](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)