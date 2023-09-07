# vissers_algorithm
Algoritm described by Gregal Vissers to detect Ellerman Bombs in 2015 paper.

This version of the algorithm consists of two functions: detect_bomb and track_bombs.

detect_bomb takes an array of image data, lower and upper threshold, as well as a minimum and (optional) maximum size. The function will detect any events in the image array that have pixel values above the lower threshold with some amount of pixels contained above the upper threshold. Contours smaller than the minimum size are discarded, and the results can either be returned, or saved to the disk.

track_bombs takes a list of save files produced by detect_bomb, and tracks the sources through the list of files, building a dictionary of sources as it does so. The window_length argument allows n consecutive nondetections before the source is considered to be inactive.

# Updated 2023:
This code has been rewritten to follow a class structure and take a configuration file as input. An example configuration file is included.

The code can by called in python by the following:

```
import vissers_algorithm as va

extractor = va.SourceExtraction(['camera0', 'camera1', ...], 'configFile')
extractor.extract_track()
```

The code will write a numpy save file with source parameters, which can be read with:

```
import numpy as np
source_overview = np.load("saveFile.npy", allow_pickle=True)
```

Within the source overview file, sources are stored in a structured array with a length equal to the number of frames analyzed. Each source has five saved parameters, which can be selected from the source overview array in the following pattern:

1. EBXXXX Centers
2. EBXXXX Contours
3. EBXXXX Flux
4. EBXXXX Extent
5. EBXXXX Origin

Where XXXX is the unique source number. 

Included are two quick overview python scripts, which can be called from the command line; read_source_extractor.py and source_animation.py

```
python source_animtion.py --help

and

python read_source_extractor.py --help
```

Should get you started.
