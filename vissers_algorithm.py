import numpy as np
import cv2
from scipy.ndimage import binary_dilation
import warnings
import time
import sys

def detect_bomb(image,upper_threshold,lower_threshold,min_size,max_size = 300, expand = None, return_mask = False, save = None):
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
    image : array-like
        The image to extract burst candidates from
    upper_threshold : float
        The upper threshold for your burst core values. Gregal Vissers (2015) used 155% of the frame mean.
    lower_threshold : float
        The lower threshold for your burst core values. Vissers used 140% of the frame mean.
    min_size : int
        The minimum number of pixels for a given kernels. Sources smaller than this are masked.
    max_size : int,optional
        The maximum size of the largest kernel.
    expand : None or bool or NxN array, optional
        If the expand keyword is a structure compatible with scipy's binary_dilation function, the mask is expanded by that structure. If expand is set to True, or if it is set to an uncompatible structure, binary_dilation is performed with a 5x5 array of ones. If the expand keyword is set to False, you're trying to break my code, and I hate you. But it should do nothing.
    return_mask : bool, optional
        If True, returns the final mask, or saves it to the file specified by the "save" kwarg
    save : None or str, optional
        If set, saves a dictionary of parameters using the numpy.save function containing the list of kernel centers, contours, fluxes, the number of bursts found, and, optionally, the mask.

    Returns
    -------
    contours : list
        A list of all contours found in the mask. Each entry is an Mx2 numpy array containing X/Y coordinates of each kernel.
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

    mask_lo = (image >= lower_threshold)
    mask_hi = (image >= upper_threshold)
    ctrs_lo,_ = cv2.findContours(mask_lo.astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE)
    ctrs_hi,_ = cv2.findContours(mask_hi.astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE)

    for cont in ctrs_lo:
        cont = cont.reshape((cont.shape[0],cont.shape[-1]))
        if cont.shape[0] < min_size:
            for pix in range(cont.shape[0]):
                mask_lo[cont[pix][1],cont[pix][0]] = False
        elif cont.shape[0] > max_size:
            warnings.warn("ValueError: max_size exceeded, max_size is currently "+str(max_size))
        else:
            cont_set = set([tuple(x) for x in cont])
            is_there_a_bright_core = []
            for hi_cont in ctrs_hi:
                hi_cont = hi_cont.reshape((hi_cont.shape[0],hi_cont.shape[-1]))
                hi_cont_set = set([tuple(x) for x in hi_cont])
                is_there_a_bright_core.append(
                        cont_set.intersection(hi_cont_set))
            is_there_a_bright_core = np.array(is_there_a_bright_core,dtype=np.bool_)
            if len(is_there_a_bright_core[is_there_a_bright_core]) == 0:
                for pix in range(cont.shape[0]):
                    mask_lo[cont[pix][1],cont[pix][0]] = False
    mask = mask_lo
    if np.all(expand) != None:
        if (type(expand) == bool) and expand:
            mask = binary_dilation(mask,structure = np.ones((5,5)))
        elif (type(expand) == bool) and not expand:
            mask = mask
        else:
            try:
                mask = binary_dilation(mask,structure = expand)
            except:
                warnings.warn("Expand keyword not a recognized structure. Using default (5x5 array of ones)")
                mask = binary_dilation(mask,structure = np.ones((5,5)))

    ctours,_ = cv2.findContours(mask.astype("uint8"),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE)
    contours = []
    centers = []
    flux = []
    for cont in ctours:
        cont = cont.reshape((cont.shape[0],cont.shape[-1]))
        contours.append(cont)
        x,y,w,h = cv2.boundingRect(cont)
        centers.append(np.array([x+w/2,y+h/2]))
        flux.append(np.sum(image[tuple(np.fliplr(cont).T.tolist())]))
    if type(save) == str:
        save_dict = {
                "Contours":contours,
                "Centers":centers,
                "Flux":flux,
                "Number":len(contours)}
        if return_mask:
            save_dict.update({"Mask":mask})
        np.save(save,save_dict)
        return len(contours)

    elif return_mask:
        return contours,centers,flux,mask
    else:
        return contours,centers,flux

def track_bombs(filelist,window_length,progress_bar = True):
    """This function takes a list of save objects, loads them, and tracks Ellerman bomb candidates contained within through the list of save objects, before returning a dictionary of EB candidates.
    Parameters:
    -----------
    filelist : list
        List of files to iterate over. Currently, the code assumes that every file in the list has the same structure as if you had used the "save" keyword argument in the "detect bomb" function. You may be asking yourself, Sean, why didn't you write this as an object if you are making these assumptions? You could have just read the files into memory, and dealt with it all in a selfsame way, rather than writing and reading so much to the disk. Let me tell you, dear reader, I wish I could have. God, would that have been nice. The dataset I wrote these functions to handle is 300GiB, and it's one of four datasets I need to run it all on. /rant
    window_length : int
        Seeing effects (from the ground) can affect the visibility of your bomb candidates. The window_length is essentially the number of missed detections that are allowable. I'd recommend it correspond to multiple seconds. For my 2.5fps data, I was having good success with a window length of 100, corresponding to about 40 seconds.
    progress_bar : bool
        Since this involves a lot of read operations and logic handling, it's likely to be quite slow, especially if you're trying to do this over a network or portable storage medium. If true, adds a nice little progress bar with a truly terrible estimate of remaining time.

    Returns:
    --------
    ellerman_bomb_candidates : dict
        A dictionary with 5*N entries, where N is the total number of candidates found. Every bomb candidate has 5 entries.
        The dictionary is structured by having the keywords "EBN Centers", "EBN Contours", "EBN Flux", "EBN Extent", "EBN Origin", where N denotes the nth bomb detected. N is padded to have four digits. All of these keywords, EXCEPT "EBN Origin" is a numpy array of length equal to the length of your filelist. The array is zero where the bomb is not active, and has nonzero entries where it is. "EBN Origin" is either O or S, for an original detection (arising from nothing) and a split detection (a child of a previous candidate), respectively.
    """
    
    ellerman_bomb_candidates = {}

    key_templates = ["EB$$ Centers",
                     "EB$$ Contours",
                     "EB$$ Flux",
                     "EB$$ Extent",
                     "EB$$ Origin"]

    neb = 0
    t_start = time.time()

    for i in range(len(filelist)):
        meta = np.load(filelist[i],allow_pickle = True).flat[0]

        if i == 0:
            ### Initialize the dictionary with initial detections ###
            for j in range(meta["Number"]):
                cen_arr = np.zeros((len(filelist),2))
                cen_arr[i,:] = meta['Centers'][j]
                contour_arr = np.zeros(len(filelist),dtype = list)
                contour_arr[i] = list(meta["Contours"][j])
                ext_arr = np.zeros(len(filelist))
                ext_arr[i] = len(meta["Contours"][j])
                flux_arr = np.zeros(len(filelist))
                flux_arr[i] = meta["Flux"][j]

                dict_keys = [k.replace("$$",str(neb).zfill(4)) for k in key_templates]
                dict_entries = [cen_arr,contour_arr,flux_arr,ext_arr,"O"]
                for k in range(len(dict_keys)):
                    ellerman_bomb_candidates.update({dict_keys[k]:dict_entries[k]})
                neb += 1
        else:
            if i <= window_length:
                startidx = 0
            else:
                startidx = i - window_length

            ### To determine overlaps, we initialize a 2d array of bool. Each entry checks the contours in the master dictionary against the contours in the frame. We do this by casting the contours to a set, and checking the sets against each other. Then, having duplicates in any given row or column indicates a merge or split event. ###
            ### 2022-09-23 I've found some issues here. In short, as the number of entries increases, so does the time for each iteration as the number of comparisons required grows exponentially. Since everything has to be compared to everything else, we end up with massive wait times to iterate through the dictionary. 

            ### My planned fix here is to only iterate through certain dictionary keys
            ### We need a quick loop up front here to check that the last window_length entries in each key was not zero, i.e., the burst was still progressing. 
            ### If the burst is active, we'll append its key to a list, then use that list
            ### In forming our discrimination matrix, which, as a result, should be MUCH smaller
            active_bursts = []
            for key in list(ellerman_bomb_candidates.keys())[2::5]:
                flux_slice = ellerman_bomb_candidates[key][startidx:i]
                if np.nanmax(flux_slice) > 0:
                    active_bursts.append(key.split(" ")[0])
            #### Now we can base the next part off active_bursts
            duplicate_matrix = np.zeros((
                len(active_bursts),
                meta["Number"]),
                dtype = np.bool_)
            for bc in range(len(active_bursts)):
                key = active_bursts[bc] + " Contours"
                for fr in range(meta["Number"]):
                    flat_bomb_contours = set(
                            [tuple(x) for x in 
                                [pair for contour in ellerman_bomb_candidates[key][
                                    startidx:i] if np.all(contour) != 0 for pair in contour]])
                    flat_frame_contours = set([tuple(x) for x in meta["Contours"][fr]])
                    duplicate_matrix[bc,fr] = flat_bomb_contours.intersection(flat_frame_contours)
            for bc in range(duplicate_matrix.shape[0]):
                column = duplicate_matrix[bc,:]
                ### Case 1: One matching entry along both axes. This indicates a continuing event. ###
                if (len(column[column]) == 1):
                    if (len(duplicate_matrix[:,
                        np.where(column)[0][0]][
                            duplicate_matrix[:,np.where(column)[0][0]]]) == 1):

                        cen = meta["Centers"][np.where(column)[0][0]]
                        con = list(meta["Contours"][np.where(column)[0][0]])
                        flux = meta["Flux"][np.where(column)[0][0]]

                        ### Add Candidates to master dictionary ###

                        ellerman_bomb_candidates[
                                active_bursts[bc]+" Centers"][i,:] = cen
                        ellerman_bomb_candidates[
                                active_bursts[bc]+" Contours"][i] = con
                        ellerman_bomb_candidates[
                                active_bursts[bc]+" Flux"][i] = flux
                        ellerman_bomb_candidates[
                                active_bursts[bc]+" Extent"][i] = len(con)
                ### Case 2: One matched entry in the column with two in the matched row. ###
                ### This would indicate that two bombs are merging, in which case, we need to discontinue the smaller event, and transfer the contours to the larger ###

                elif (len(column[column]) == 1):
                    if (len(duplicate_matrix[:,np.where(column)[0][0]]) >= 2):
                        cen = meta["Centers"][np.where(column)[0][0]]
                        con = list(meta["Contours"][np.where(column)[0][0]])
                        flux = meta["Flux"][np.where(column)[0][0]]

                        row = duplicate_matrix[:,np.where(column)[0][0]]
                        parents = np.where(row)[0]
                        parent_size = np.zeros(len(parents))
                        for p in range(len(parents)):
                            flat_parent_contours = set(
                                    [tuple(x) for x in 
                                        [pair for contour in ellerman_bomb_candidates[
                                            active_bursts[parents[p]]+" Contours"][
                                                startidx:i] if np.all(contour) != 0 for pair in contour]])
                            parent_size[p] = len(flat_parent_contours)
                            big_parent = np.where(parent_size == parent_size.max())[0][0]

                            ellerman_bomb_candidates[
                                    active_bursts[bc]+" Centers"][i,:] = cen
                            ellerman_bomb_candidates[
                                    active_bursts[bc]+" Contours"][i] = con
                            ellerman_bomb_candidates[
                                    active_bursts[bc]+" Flux"][i] = flux
                            ellerman_bomb_candidates[
                                    active_bursts[bc]+" Extent"][i] = len(con)
                ### Case 3: Two or more matached entries in the column ###
                ### This indicates a split event! ###
                elif (len(column[column]) >= 2):
                    if (len(duplicate_matrix[:,np.where(column)[0][0]]) == 1):
                        daughters = np.where(column)[0]
                        daughter_size = np.zeros(len(daughters))
                        for d in range(len(daughters)):
                            flat_daughter_contours = set(
                                    [tuple(x) for x in meta["Contours"][daughters[d]]])
                            daughter_size[d] = len(flat_daughter_contours)
                        for d in range(len(daughters)):
                            cen = meta["Centers"][daughters[d]]
                            con = list(meta["Contours"][daughters[d]])
                            flux = meta["Flux"][daughters[d]]

                            if daughter_size[d] == daughter_size.max():
                                ellerman_bomb_candidates[
                                    active_bursts[bc]+" Centers"][i,:] = cen
                                ellerman_bomb_candidates[
                                    active_bursts[bc]+" Contours"][i] = con
                                ellerman_bomb_candidates[
                                    active_bursts[bc]+" Flux"][i] = flux
                                ellerman_bomb_candidates[
                                    active_bursts[bc]+" Extent"][i] = len(con)
                            else:
                                cen_arr = np.zeros((len(filelist),2))
                                cen_arr[i,:] = cen
                                contour_arr = np.zeros(len(filelist),dtype = list)
                                contour_arr[i] = con
                                ext_arr = np.zeros(len(filelist))
                                ext_arr[i] = len(con)
                                flux_arr = np.zeros(len(filelist))
                                flux_arr[i] = flux

                                dict_keys = [k.replace("$$",str(neb).zfill(4)) for k in key_templates]
                                dict_entries = [cen_arr,contour_arr,flux_arr,ext_arr,"S"]
                                for k in range(len(dict_keys)):
                                    ellerman_bomb_candidates.update({dict_keys[k]:dict_entries[k]})
                                neb += 1

                ### Case 4: Two or more mateched entries in the column ###
                ### Most likely, this indicates the aftermath of a recent split or merge event ###
                ### In this case, we match the overlaps by size ###
                ### If there is a secondary split or merge event occurring at the same time, gods help you ###
                elif (len(column[column]) >= 2):
                    if (len(duplicate_matrix[:,np.where(column)[0][0]]) >= 2):
                        daughters = np.where(column)[0]
                        daughter_size = np.zeros(len(daughters))
                        for d in range(len(daughters)):
                            flat_daughter_contours = set(
                                    [tuple(x) for x in meta["Contours"][daughters[d]]])
                            daughter_size[d] = len(flat_daughter_contours)
                        row = duplicate_matrix[:,np.where(column)[0][0]]
                        parents = np.where(row)[0]
                        parent_size = np.zeros(len(parents))
                        for p in range(len(parents)):
                            flat_parent_contours = set(
                                    [tuple(x) for x in [
                                        pair for contour in ellerman_bomb_candidates[
                                            active_bursts[parents[p]]+" Contours"][startidx:i]
                                        if np.all(contour) != 0 for pair in contour]])
                            parent_size[p] = len(
                                    flat_parent_contours)
                        parent_argsort = parents[np.argsort(parent_size)]
                        daughter_argsort = daughters[np.argsort(daughter_size)]

                        for p in range(len(parent_argsort)):
                            cen = meta["Centers"][daughters[p]]
                            con = list(meta["Contours"][daughters[p]])
                            flux = meta["Flux"][daughters[p]]
                            ellerman_bomb_candidates[
                                    active_bursts[
                                        parent[p]]+" Centers"][i,:] = cen
                            ellerman_bomb_candidates[
                                    active_bursts[
                                        parent[p]]+" Contours"][i] = con
                            ellerman_bomb_candidates[
                                    active_bursts[
                                        parent[p]]+" Flux"][i] = flux
                            ellerman_bomb_candidates[
                                    active_bursts[
                                        parent[p]]+" Extent"][i] = len(con)
            for fr in range(duplicate_matrix.shape[1]):
                row = duplicate_matrix[:,fr]
                if not np.any(row):
                    cen = meta["Centers"][fr]
                    con = list(meta["Contours"][fr])
                    flux = meta["Flux"][fr]

                    cen_arr = np.zeros((len(filelist),2))
                    cen_arr[i,:] = cen
                    contour_arr = np.zeros(len(filelist),dtype = list)
                    contour_arr[i] = con
                    ext_arr = np.zeros(len(filelist))
                    ext_arr[i] = len(con)
                    flux_arr = np.zeros(len(filelist))
                    flux_arr[i] = flux

                    dict_keys = [k.replace("$$",str(neb).zfill(4)) for k in key_templates]
                    dict_entries = [cen_arr,contour_arr,flux_arr,ext_arr,"O"]
                    for k in range(len(dict_keys)):
                        ellerman_bomb_candidates.update({dict_keys[k]:dict_entries[k]})
                    neb += 1

        if progress_bar:
            sys.stdout.write("\r")
            pct_complete = 100 * (i+1) / len(filelist)
            pt = time.time() - t_start
            estt = pt/(pct_complete/100.) - pt

            sys.stdout.write("[{}{}] {}% ({}/{}) {}m {}s  |  EBs: {}  | Est. Remaining: {}m {}s".format(
                "="*int(pct_complete/2),
                " "*int(((100-pct_complete)/2)),
                "{:.2f}".format(round(pct_complete,2)),
                i+1,
                len(filelist),
                int(pt/60),
                int(pt) % 60,
                neb,
                int(estt/60),
                int(estt) % 60))
            sys.stdout.flush()

    return ellerman_bomb_candidates
