# vissers_algorithm
Algoritm described by Gregal Vissers to detect Ellerman Bombs in 2015 paper.

This version of the algorithm consists of two functions: detect_bomb and track_bombs.

detect_bomb takes an array of image data, lower and upper threshold, as well as a minimum and (optional) maximum size. The function will detect any events in the image array that have pixel values above the lower threshold with some amount of pixels contained above the upper threshold. Contours smaller than the minimum size are discarded, and the results can either be returned, or saved to the disk.

track_bombs takes a list of save files produced by detect_bomb, and tracks the sources through the list of files, building a dictionary of sources as it does so. The window_length argument allows n consecutive nondetections before the source is considered to be inactive.
