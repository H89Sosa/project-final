![](https://techsalesgroup.files.wordpress.com/2016/10/ironhack-logo-negro1.jpg =300x300)


# Hernán Sosa
## Ironhack Final Project - Audio processing

My final project is intended to be a tool which musicians and sound professionals would need. 

As a musician, I found myself struggling to learn to pay some of the parts on my instrument by ear. My idea is to come with a tool that takes a full audio mix, and returns each instrument track, enhances it, and also returns a musical sheet in pdf.

We'll be focusing our tool in one genre -rock- for several reasons:

	- The particular knowledge on this genre and the ability to categorize all the major features 
	and instruments that compose this style.
	
	- Luckily for me, this genre is pretty conservative in its majority in terms of structure and sound,
	which can improve the result of our product.
	
	- The target of this product is the average home musician. This features could fit their needs. 
	Generally speaking, this profile of people is usually interested in rock music and 
	technology, due to their age.
	
	- Focusing on one style narrows the nearly infinite concept of music, and could help our models 
	on better predictions.
	
	- The ability to extract and generate data.
	
	- Taking a modular approach on this project permits the eventual expansion to other genres.
	
	- As a product, it can be comercialized to this already existing consumer profile, just with this genre.
	
	- Personal preference. This might not seem important, but I feel I'll be more involved and focused 
	if I work on this style.
	
### - 1. Timbrical feature extraction

For this aspect, the approach we could take is to create a learning sound library, trying to compile all the possible combinations of instruments and variations of each. Then, we would train a neural network to be able to classify each instrument and extract it.

### - 2. Enhancement

As seen in other similar projects, the result of this extraction could not have the desired quality. In this step, we would transform our extracted audio file into a more sharp signal using a lower-level classification of the instrument, extracting its features and mergin them with the original signal.

### - 3. Sheet creation

Once we have this sharpened signals, we would create a system that finds the notes that the instrument is playing (tone and time) by searching its fundamental tone and harmonics over the time and try to map them in a structured dataframe, in order to be able to make a visual extraction of them. We will look for libraries to traduce this data into a music sheet format generator.

USAGE OF MIDI FILES TO HELP THE PROCESS????


We are aware that introducing all this features is a bit too much for the time we're given. We'll focus on the timbrical feature extraction by the moment, and if we achieve good results we could try to approach the other phases.
