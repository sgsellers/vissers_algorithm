import astropy.io.fits as fits
import configparser
import cv2
import dask
import glob
import numpy as np
import os
import tqdm
import warnings

from dask.diagnostics import ProgressBar
from scipy.ndimage import binary_dilation


@dask.delayed
def detect_bomb(file,
                upper_threshold,
                lower_threshold,
                threshold_type,
                min_size,
                max_size,
                save_pattern,
                expand=None,
                extension=0):
    """Second version of my burst detection algorithm.
    This function takes an image and finds kernels above the lower threshold that contain
    pixels above the upper threshold. If the kernel is smaller than the minimum size, it
    is masked instead. Optional arguments include saving a dictionary of burst parameters to
    the disk, returning in addition to other parameters, and performing a binary expansion
    on the kernels to artificially increase their size. If you are running this function
    on a data set, I would highly recommend setting the expand keyword if you are anything
    other than perfectly confident in your alignment.

    Parameters
    ----------
    file : str
        The filename to extract burst candidates from
    upper_threshold : float
        The upper threshold for your burst core values. Gregal Vissers (2015) used 155% of the frame mean.
    lower_threshold : float
        The lower threshold for your burst core values. Vissers used 140% of the frame mean.
    threshold_type : str
        Either percentage or data. Data is in DN in the file.
    min_size : int
        The minimum number of pixels for a given kernels. Sources smaller than this are masked.
    max_size : int,optional
        The maximum size of the largest kernel.
    save_pattern : str
        Pattern for savefile
    expand : None or bool or NxN array, optional
        If the expand keyword is a structure compatible with scipy's binary_dilation function,
        the mask is expanded by that structure. If expand is set to True, or if it is set to an uncompatible structure,
        binary_dilation is performed with a 5x5 array of ones.
        If the expand keyword is set to False, you're trying to break my code, and I hate you. But it should do nothing.
    extension : int
        Extension of fits file to analyse

    Returns
    -------
    contours : list
        A list of all contours found in the mask.
        Each entry is an Mx2 numpy array containing X/Y coordinates of each kernel.
    centers : list
        A list of the center coordinate of each contous.
    flux : list
        Each entry is the sum of all pixel values within the corresponding contour
    mask : array-like,optional
        Boolean array that is true where a valid kernel is detected, and False elsewhere
    neb : int, optional
        Returned only if the save kwarg is set. The number of discrete sources detected.

    Raises
    ------
    ValueError: if the maximum size is exceeded for any contour found.

    Rewritten on 2022-09-23 from the ground up for speed and handling of extended sources.
    """

    image = fits.open(file)[extension].data
    if threshold_type == 'percent':
        lower_threshold = lower_threshold * np.nanmean(image)
        upper_threshold = upper_threshold * np.nanmean(image)

    mask_lo = (image >= lower_threshold)
    mask_hi = (image >= upper_threshold)
    ctrs_lo, _ = cv2.findContours(mask_lo.astype('uint8'),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_NONE)
    ctrs_hi, _ = cv2.findContours(mask_hi.astype('uint8'),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_NONE)

    for cont in ctrs_lo:
        cont = cont.reshape((cont.shape[0], cont.shape[-1]))
        if cont.shape[0] < min_size:
            for pix in range(cont.shape[0]):
                mask_lo[cont[pix][1], cont[pix][0]] = False
        elif cont.shape[0] > max_size:
            warnings.warn("ValueError: max_size exceeded, max_size is currently " + str(max_size))
        else:
            cont_set = set([tuple(x) for x in cont])
            is_there_a_bright_core = []
            for hi_cont in ctrs_hi:
                hi_cont = hi_cont.reshape((hi_cont.shape[0], hi_cont.shape[-1]))
                hi_cont_set = set([tuple(x) for x in hi_cont])
                is_there_a_bright_core.append(
                    cont_set.intersection(hi_cont_set))
            is_there_a_bright_core = np.array(is_there_a_bright_core, dtype=np.bool_)
            if len(is_there_a_bright_core[is_there_a_bright_core]) == 0:
                for pix in range(cont.shape[0]):
                    mask_lo[cont[pix][1], cont[pix][0]] = False
    mask = mask_lo
    if np.all(expand) is not None:
        if (type(expand) == bool) and expand:
            mask = binary_dilation(mask, structure=np.ones((5, 5)))
        elif (type(expand) == bool) and not expand:
            mask = mask
        else:
            mask = binary_dilation(mask, structure=expand)

    ctours, _ = cv2.findContours(mask.astype("uint8"),
                                 cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_NONE)
    contours = []
    centers = []
    flux = []
    extents = []
    for cont in ctours:
        cont = cont.reshape((cont.shape[0], cont.shape[-1]))
        contours.append(cont)
        x, y, w, h = cv2.boundingRect(cont)
        centers.append(np.array([x + w / 2, y + h / 2]))
        flux.append(np.sum(image[tuple(np.fliplr(cont).T.tolist())]))
        extents.append(cont.shape[0])

    # Now, each contours, centers, flux, extents is an array of length N
    # Where N is the number of extracted sources. These can all be cast as numpy arrays EXCEPT
    # for contours, which is a list of numpy arrays, each with shape X,2, where X is variable.
    # To cast this to an array, it needs to be a contour array.

    flux = np.array(flux)
    extents = np.array(extents)
    center_array = np.empty(len(flux), dtype=object)
    contour_array = np.empty(len(flux), dtype=object)
    for i in range(len(flux)):
        center_array[i] = tuple(centers[i])
        contour_array[i] = contours[i]

    source_information = np.rec.fromarrays(
        [
            contour_array,
            center_array,
            flux,
            extents
        ],
        names=[
            'CONTOURS',
            'CENTERS',
            'FLUX',
            'EXTENT'
        ]
    )

    np.save(save_pattern, source_information)
    return len(contours)


class SourceExtraction:
    """
    Master class for source extraction and tracking
    """

    def __init__(self, cameras, configfile):
        self.configFile = configfile
        self.cameras = cameras
        self.camera = ""
        self.dataBase = ""
        self.workBase = ""
        self.dataFilePattern = ""
        self.upperThreshold = 0
        self.lowerThreshold = 0
        self.thresholdType = ""
        self.minSize = 0
        self.maxSize = 0
        self.expand = []
        self.sourceSavePattern = ""
        self.windowLength = 15
        self.overviewSaveName = ""
        self.targetList = []
        self.sourceList = []
        self.numberSources = []
        return

    def extract_track(self):
        if type(self.cameras) == list:
            for i in range(len(self.cameras)):
                self.camera = self.cameras[i]
                print("Working on ", self.camera)
                self.setup()
                self.extract_sources()
                self.track_sources()
        else:
            self.camera = self.cameras
            print("Working on ", self.camera)
            self.setup()
            self.extract_sources()
            self.track_sources()
        return

    def setup(self):
        config = configparser.ConfigParser()
        config.read(self.configFile)
        self.dataBase = config[self.camera]['dataBase']
        self.workBase = config[self.camera]['workBase']
        if not os.path.isdir(self.workBase):
            print("{0}: os.mkdir: attempting to create directory:""{1}".format(__name__, self.workBase))
            try:
                os.mkdir(self.workBase)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise
        self.dataFilePattern = config[self.camera]['dataFilePattern']
        self.targetList = sorted(glob.glob(os.path.join(self.dataBase, self.dataFilePattern)))
        self.upperThreshold = float(config[self.camera]['upperThreshold'])
        self.lowerThreshold = float(config[self.camera]['lowerThreshold'])
        self.thresholdType = config[self.camera]['thresholdType']
        self.minSize = int(config[self.camera]['minSize'])
        self.maxSize = int(config[self.camera]['maxSize'])
        if config[self.camera]['expand'].lower() == 'none':
            self.expand = None
        else:
            self.expand = np.ones(
                (int(config[self.camera]['expand']),
                 int(config[self.camera]['expand'])
                 )
            )
        self.sourceSavePattern = config[self.camera]['sourceSavePattern']
        self.windowLength = int(config[self.camera]['windowLength'])
        self.overviewSaveName = config[self.camera]['overviewSaveName']
        self.sourceList = []
        self.numberSources = 0
        return

    def extract_sources(self):
        results = []
        for i in range(len(self.targetList)):
            savestr = os.path.join(self.workBase,
                                   self.sourceSavePattern.format(i))
            self.sourceList.append(savestr)
            results.append(
                detect_bomb(
                    self.targetList[i],
                    self.upperThreshold,
                    self.lowerThreshold,
                    self.thresholdType,
                    self.minSize,
                    self.maxSize,
                    savestr,
                    self.expand
                )
            )
        with ProgressBar():
            neb = dask.compute(results)[0]
        self.numberSources = np.array(neb)
        print("There were a total of",
              np.sum(self.numberSources),
              "Sources detected with the given parameters across ",
              len(self.targetList),
              " files")
        print("----------------------")
        print("Average Per Frame: ", np.nanmean(self.numberSources))
        print("Median Per Frame: ", np.nanmedian(self.numberSources))
        print("Maximum: ", np.nanmax(self.numberSources))
        return

    def track_sources(self):
        """This function takes a list of save objects, loads them, and tracks Ellerman bomb candidates contained within
        through the list of save objects, before returning a dictionary of EB candidates.

        Returns:
        --------
        sources : dict
            A dictionary with 5*N entries, where N is the total number of candidates found.
            Every bomb candidate has 5 entries.
            The dictionary is structured by having the keywords
            "EBN Centers", "EBN Contours", "EBN Flux", "EBN Extent", "EBN Origin",
            where N denotes the nth bomb detected. N is padded to have four digits.
            All of these keywords, EXCEPT "EBN Origin" is a numpy array of length equal to the length of your filelist.
            The array is zero where the bomb is not active, and has nonzero entries where it is.
            "EBN Origin" is either O or S, for an original detection (arising from nothing) and a split detection
            (a child of a previous candidate), respectively.
        """
        filelist = self.sourceList
        window_length = self.windowLength

        sources = {}

        key_templates = ["EB$$ Centers",
                         "EB$$ Contours",
                         "EB$$ Flux",
                         "EB$$ Extent",
                         "EB$$ Origin"]

        neb = 0

        for i in tqdm.tqdm(range(len(filelist)), desc="Tracking Sources"):
            meta = np.load(filelist[i], allow_pickle=True)

            if i == 0:
                # Initialize the dictionary with initial detections
                for j in range(self.numberSources[i]):
                    cen_arr = np.zeros(len(filelist), dtype=object)
                    cen_arr[i] = meta["CENTERS"][j]
                    contour_arr = np.zeros(len(filelist), dtype=object)
                    contour_arr[i] = list(meta["CONTOURS"][j])
                    ext_arr = np.zeros(len(filelist))
                    ext_arr[i] = len(meta["CONTOURS"][j])
                    flux_arr = np.zeros(len(filelist))
                    flux_arr[i] = meta["FLUX"][j]

                    dict_keys = [k.replace("$$", str(neb).zfill(4)) for k in key_templates]
                    dict_entries = [cen_arr, contour_arr, flux_arr, ext_arr, np.repeat("O", len(filelist))]
                    for k in range(len(dict_keys)):
                        sources.update({dict_keys[k]: dict_entries[k]})
                    neb += 1
            else:
                if i <= window_length:
                    startidx = 0
                else:
                    startidx = i - window_length

                # To determine overlaps, we initialize a 2d array of bool.
                # Each entry checks the contours in the master dictionary against the contours in the frame.
                # We do this by casting the contours to a set, and checking the sets against each other.
                # Then, having duplicates in any given row or column indicates a merge or split event.
                # 2022-09-23 I've found some issues here. In short, as the number of entries increases,
                # so does the time for each iteration as the number of comparisons required grows exponentially.
                # Since everything has to be compared to everything else, we end up with massive wait times to iterate
                # through the dictionary.

                # My planned fix here is to only iterate through certain dictionary keys
                # We need a quick loop up front here to check that the last window_length entries in each key was not
                # zero, i.e., the burst was still progressing.
                # If the burst is active, we'll append its key to a list, then use that list
                # In forming our discrimination matrix, which, as a result, should be MUCH smaller
                active_bursts = []
                for key in list(sources.keys())[2::5]:
                    flux_slice = sources[key][startidx:i]
                    if np.nanmax(flux_slice) > 0:
                        active_bursts.append(key.split(" ")[0])
                # Now we can base the next part off active_bursts
                duplicate_matrix = np.zeros((
                    len(active_bursts),
                    self.numberSources[i]),
                    dtype=np.bool_)
                for bc in range(len(active_bursts)):
                    key = active_bursts[bc] + " Contours"
                    for fr in range(self.numberSources[i]):
                        flat_bomb_contours = set(
                            [tuple(x) for x in
                             [pair for contour in sources[key][
                                                  startidx:i] if np.all(contour) != 0 for pair in contour]])
                        flat_frame_contours = set([tuple(x) for x in meta["CONTOURS"][fr]])
                        duplicate_matrix[bc, fr] = flat_bomb_contours.intersection(flat_frame_contours)
                for bc in range(duplicate_matrix.shape[0]):
                    column = duplicate_matrix[bc, :]
                    # Case 1: One matching entry along both axes. This indicates a continuing event.
                    if len(column[column]) == 1:
                        if (len(duplicate_matrix[:,
                                np.where(column)[0][0]][
                                    duplicate_matrix[:, np.where(column)[0][0]]]) == 1):
                            cen = meta["CENTERS"][np.where(column)[0][0]]
                            con = list(meta["CONTOURS"][np.where(column)[0][0]])
                            flux = meta["FLUX"][np.where(column)[0][0]]

                            # Add Candidates to master dictionary

                            sources[
                                active_bursts[bc] + " Centers"][i] = cen
                            sources[
                                active_bursts[bc] + " Contours"][i] = con
                            sources[
                                active_bursts[bc] + " Flux"][i] = flux
                            sources[
                                active_bursts[bc] + " Extent"][i] = len(con)
                    # Case 2: One matched entry in the column with two in the matched row.
                    # This would indicate that two bombs are merging, in which case, we need to discontinue the smaller
                    # event, and transfer the contours to the larger

                    elif len(column[column]) == 1:
                        if len(duplicate_matrix[:, np.where(column)[0][0]]) >= 2:
                            cen = meta["CENTERS"][np.where(column)[0][0]]
                            con = list(meta["CONTOURS"][np.where(column)[0][0]])
                            flux = meta["FLUX"][np.where(column)[0][0]]

                            row = duplicate_matrix[:, np.where(column)[0][0]]
                            parents = np.where(row)[0]
                            parent_size = np.zeros(len(parents))
                            for p in range(len(parents)):
                                flat_parent_contours = set(
                                    [tuple(x) for x in
                                     [pair for contour in sources[
                                                              active_bursts[parents[p]] + " Contours"][
                                                          startidx:i] if np.all(contour) != 0 for pair in contour]])
                                parent_size[p] = len(flat_parent_contours)

                                sources[
                                    active_bursts[bc] + " Centers"][i] = cen
                                sources[
                                    active_bursts[bc] + " Contours"][i] = con
                                sources[
                                    active_bursts[bc] + " Flux"][i] = flux
                                sources[
                                    active_bursts[bc] + " Extent"][i] = len(con)
                    # Case 3: Two or more matached entries in the column
                    # This indicates a split event!
                    elif len(column[column]) >= 2:
                        if len(duplicate_matrix[:, np.where(column)[0][0]]) == 1:
                            daughters = np.where(column)[0]
                            daughter_size = np.zeros(len(daughters))
                            for d in range(len(daughters)):
                                flat_daughter_contours = set(
                                    [tuple(x) for x in meta["CONTOURS"][daughters[d]]])
                                daughter_size[d] = len(flat_daughter_contours)
                            for d in range(len(daughters)):
                                cen = meta["CENTERS"][daughters[d]]
                                con = list(meta["CONTOURS"][daughters[d]])
                                flux = meta["FLUX"][daughters[d]]

                                if daughter_size[d] == daughter_size.max():
                                    sources[
                                        active_bursts[bc] + " Centers"][i] = cen
                                    sources[
                                        active_bursts[bc] + " Contours"][i] = con
                                    sources[
                                        active_bursts[bc] + " Flux"][i] = flux
                                    sources[
                                        active_bursts[bc] + " Extent"][i] = len(con)
                                else:
                                    cen_arr = np.zeros(len(filelist), dtype=object)
                                    cen_arr[i] = cen
                                    contour_arr = np.zeros(len(filelist), dtype=object)
                                    contour_arr[i] = con
                                    ext_arr = np.zeros(len(filelist))
                                    ext_arr[i] = len(con)
                                    flux_arr = np.zeros(len(filelist))
                                    flux_arr[i] = flux

                                    dict_keys = [k.replace("$$", str(neb).zfill(4)) for k in key_templates]
                                    dict_entries = [cen_arr,
                                                    contour_arr,
                                                    flux_arr,
                                                    ext_arr,
                                                    np.repeat("S", len(filelist))]
                                    for k in range(len(dict_keys)):
                                        sources.update({dict_keys[k]: dict_entries[k]})
                                    neb += 1

                    # Case 4: Two or more mateched entries in the column
                    # Most likely, this indicates the aftermath of a recent split or merge event
                    # In this case, we match the overlaps by size
                    # If there is a secondary split or merge event occurring at the same time, gods help you
                    elif len(column[column]) >= 2:
                        if len(duplicate_matrix[:, np.where(column)[0][0]]) >= 2:
                            daughters = np.where(column)[0]
                            daughter_size = np.zeros(len(daughters))
                            for d in range(len(daughters)):
                                flat_daughter_contours = set(
                                    [tuple(x) for x in meta["CONTOURS"][daughters[d]]])
                                daughter_size[d] = len(flat_daughter_contours)
                            row = duplicate_matrix[:, np.where(column)[0][0]]
                            parents = np.where(row)[0]
                            parent_size = np.zeros(len(parents))
                            for p in range(len(parents)):
                                flat_parent_contours = set(
                                    [tuple(x) for x in [
                                        pair for contour in sources[
                                                                active_bursts[parents[p]] + " Contours"][startidx:i]
                                        if np.all(contour) != 0 for pair in contour]])
                                parent_size[p] = len(
                                    flat_parent_contours)
                            parent_argsort = parents[np.argsort(parent_size)]

                            for p in range(len(parent_argsort)):
                                cen = meta["CENTERS"][daughters[p]]
                                con = list(meta["CONTOURS"][daughters[p]])
                                flux = meta["FLUX"][daughters[p]]
                                sources[
                                    active_bursts[
                                        parents[p]] + " Centers"][i] = cen
                                sources[
                                    active_bursts[
                                        parents[p]] + " Contours"][i] = con
                                sources[
                                    active_bursts[
                                        parents[p]] + " Flux"][i] = flux
                                sources[
                                    active_bursts[
                                        parents[p]] + " Extent"][i] = len(con)
                for fr in range(duplicate_matrix.shape[1]):
                    row = duplicate_matrix[:, fr]
                    if not np.any(row):
                        cen = meta["CENTERS"][fr]
                        con = list(meta["CONTOURS"][fr])
                        flux = meta["FLUX"][fr]

                        cen_arr = np.zeros(len(filelist), dtype=object)
                        cen_arr[i] = cen
                        contour_arr = np.zeros(len(filelist), dtype=object)
                        contour_arr[i] = con
                        ext_arr = np.zeros(len(filelist))
                        ext_arr[i] = len(con)
                        flux_arr = np.zeros(len(filelist))
                        flux_arr[i] = flux

                        dict_keys = [k.replace("$$", str(neb).zfill(4)) for k in key_templates]
                        dict_entries = [cen_arr, contour_arr, flux_arr, ext_arr, np.repeat("O", len(filelist))]
                        for k in range(len(dict_keys)):
                            sources.update({dict_keys[k]: dict_entries[k]})
                        neb += 1

        dictionary_keys = list(sources.keys())
        dictionary_fields = []
        for key in dictionary_keys:
            dictionary_fields.append(sources[key])

        # source_array = np.rec.fromarrays(dictionary_fields, names=dictionary_keys)
        savename = os.path.join(self.workBase, self.overviewSaveName)
        np.save(savename, source_array)
        return
