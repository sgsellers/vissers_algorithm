import numpy as np
import cv2
from scipy.ndimage import binary_dilation
import time
import sys
import warnings

def create_mask(array,mean,threshold):
    """Creates a mask of size array that passes true
    if the value is above mean * threshold, and false elsewhere
    Parameters
    ----------
    array : array-like
        The input array to create the mask from
    mean : float
        The user-defined frame mean.
    threshold : float
        The multiplier to apply to mean to find threshold pixels

    Returns
    -------
    mask : array-like, bool
        The numpy array of bools that correspond to values of mean*threshold
    """

    mask = (array >= mean * threshold)

    return mask

def mask_adjacent(array,mean,threshold,mask,size=2):
    """Modifies a given mask to include adjecent points that pass a second, smaller threshold.
    Now performs binary filter of given size, then dilates mask by same size, in order to cut
    out small-scale structure.

    Parameters
    ----------
    array : array-like
        The input array to modify the mask
    mean : float
        The user-defined frame mean.
    threshold : float
        The multiplier to apply to mean to find threshold pixels
    mask : array-like, bool
        The mask to modify
    size : int
        Size of the structure for binary filtering

    Returns
    -------
    mask : array-like, bool
    """

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if mask[i,j]:
                ## Iterate outwards until the circle no longer contains values above threshold
                radius = 1
                new_values = True
                while new_values:
                    sel = array[i-radius:i+radius+1,j-radius:j+radius+1]
                    mask_sel = mask[i-radius:i+radius+1,j-radius:j+radius+1]
                    expanded = 0
                    for k in range(sel.shape[0]):
                        for l in range(sel.shape[1]):
                            if (sel[k,l] >= mean*threshold) & (not mask_sel[k,l]):
                                mask_sel[k,l] = True
                                expanded += 1
                    mask[i-radius:i+radius+1,j-radius:j+radius+1] = mask_sel
                    if expanded > 0:
                        radius += 1
                        new_values = True
                    else:
                        new_values = False

    ftpt = np.ones((size,size))
    mask = binary_dilation(binary_opening(mask,structure=ftpt),structure=ftpt)
    return mask

def get_contours(mask):
    """From a given (cleaned) mask, get a set of contours for sources in the mask.
    Parameters
    ----------
    mask : array-like
        A mask array that has (nonzero) values where there's an EB candidate, zeros values elsewhere

    Returns
    -------
    conts : list
        A list with length n of all contours found in the mask. Each entry in the list has shape
        m,2, where m is the number of points in the array. The X/Y coords of each
    cens : list
        A list with length n of the centers of all contours. Each entry has the length 2, for X/Y coords
    """
    contours,_ = cv2.findContours(mask.astype("uint8"),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_NONE)
    conts = []
    cens = []
    for cont in contours:
        conts.append(cont.reshape((cont.shape[0],cont.shape[-1])))
        x,y,w,h = cv2.boundingRect(cont)
        cens.append(np.array([x+w/2,y+h/2]))

    return conts,cens

def detect_bomb(image,upper_threshold,lower_threshold,min_size,expand = None,return_mask = True,save = None,auto_raise_threshold = True):
    """A function that combines and simplifies the above functions. In short, this function:
    Takes an image, finds kernals above the upper threshold, expands those kernels
    to the edge of the lower threshold, masks kernels that contain fewer points than size,
    and returns a mask and set of contours.

    DISCLAIMER: If you're not me, and you're reading this, you're probably not using IBIS data from the Dunn. I wrote these routines for the Dunn, which is quite nice for alignment and derotation. Make SURE your data is fine aligned and derotated, OR make sure you're expand keyword is large. The contours returned are in pixel numbers, not any kind of real coordinate. In addition, any kind of flux normalization should be done before you run this function, or your flux won't be any kind of sensible.

    Parameters
    ----------
    image : array-like
        The image to extract Ellerman Bomb candidates from
    upper_threshold : float
        The threshold value for Ellerman Bomb cores. Vissers uses 155% of the frame mean.
    lower_threshold : float
        The lower threshold to expand kernels to.
    min_size : int
        The minimum number of points for a kernel to be considered.
    expand : None,array-like
        If the expand keyword is a structure compatible with scipy's binary_dilation function, it uses the structure to expand your mask by that structure. Suggested structures include np.ones((SIZE,SIZE)). This allows for a more relaxed contour tracking, as well as exaggerated source size, which may be good if your cadence is low.
    return_mask : bool
        If True, the function returns the mask used to generate contours. Useful for visual inspection, but not much else.
    save : None-type or str
        If set to a string, the function will use numpy's save function to save a dictionary containing the centers, contours, and, if the return_mask keyword is set, the mask to the disk.
        
    Returns
    -------
    contours : list
        A list with length n of all contours found in the mask.
        Each entry in the list has shape m,2, where m is the number of points in the array.
        The X/Y coords of each
    centers : list
        A list with length n of the centers of all contours.
        Each entry has the length 2, for X/Y coords
    flux : list
        List of floats of the sum of all pixels contained in the corresponding contour set.
    mask : array-like
        Boolean array that is True where an Ellerman Bomb candidate is detected, False elsewhere."""

    mask = (image >= upper_threshold)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i,j]:
                ## Iterate outwards until the circle no longer contains values above threshold
                radius = 1
                new_values = True
                while new_values:
                    sel = image[i-radius:i+radius+1,j-radius:j+radius+1]
                    mask_sel = mask[i-radius:i+radius+1,j-radius:j+radius+1]
                    expanded = 0
                    for k in range(sel.shape[0]):
                        for l in range(sel.shape[1]):
                            if (sel[k,l] >= lower_threshold) & (not mask_sel[k,l]):
                                mask_sel[k,l] = True
                                expanded += 1
                    mask[i-radius:i+radius+1,j-radius:j+radius+1] = mask_sel
                    if expanded > 0:
                        radius += 1
                        new_values = True
                    elif (radius >= mask.shape[0]) or (radius >= mask.shape[1]):
                        if auto_raise_threshold:
                            warnings.warn("Warning: Threshold error detected. Automatically raising threshold by 10%")
                            lower_threshold = 1.1 * lower_threshold
                            upper_threshold = 1.1 * upper_threshold
                            mask = (image >= upper_threshold)
                            radius = 1
                            new_values = True
                        else:
                            new_values = False
                            raise Exception("Your thresholds are too low, and the entire image has been selected")
                    else:
                        new_values = False

    ctours,_ = cv2.findContours(mask.astype("uint8"),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_NONE)
    contours = []
    centers = []
    flux = []
    for cont in ctours:
        cont = cont.reshape((cont.shape[0],cont.shape[-1]))
        if cont.shape[0] < min_size:
            for pix in range(cont.shape[0]):
                mask[cont[pix][1],cont[pix][0]] = False
        else:
            contours.append(cont)
            x,y,w,h = cv2.boundingRect(cont)
            centers.append(np.array([x+w/2,y+h/2]))
            flux.append(
                    np.sum(
                        image[
                            tuple(
                                np.fliplr(
                                    cont).T.tolist())]))

    if np.all(expand) != None:
        #### Binary Dilation and re-contour ####
        mask = binary_dilation(mask,structure = np.ones((3,3)))
        ctours,_ = cv2.findContours(mask.astype("uint8"),
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)
        contours = []
        centers = []
        flux = []
        for cont in ctours:
            cont = cont.reshape((cont.shape[0],cont.shape[-1]))
            if cont.shape[0] < min_size:
                for pix in range(cont.shape[0]):
                    mask[cont[pix][1],cont[pix][0]] = False
            else:
                contours.append(cont)
                x,y,w,h = cv2.boundingRect(cont)
                centers.append(np.array([x+w/2,y+h/2]))
                flux.append(
                        np.sum(
                            image[
                                tuple(
                                    np.fliplr(
                                        cont).T.tolist())]))
    if type(save) == str:
        save_dict = {
                "Contours":contours,
                "Centers":centers,
                "Fluxes":flux,
                "Number":len(contours)
                }
        if return_mask:
            save_dict.update({"Mask":mask})
        
        np.save(save,save_dict)
        return len(contours)

    if (return_mask) & (type(save) != str):
        return contours, centers, flux, mask
    else:
        return contours, centers, flux

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
        The dictionary is structured by having the keywords "EBN Centers", "EBN Contours", "EBN Flux", "EBN Extent", "EBN Origin", where N denotes the nth bomb detected. N is padded to have four digits. All of these keywords, EXCEPT "EBN Origin" is a numpy array of length equal to the length of your filelist. The array is zero where the bomb is not active, and has nonzero entries where it is. "EBN Origin" is either O or S, for an original detection (arising from nothing) and a split detection (a child of a previous candidate), respectively."""
    
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
                flux_arr[i] = meta["Fluxes"][j]

                dict_keys = [k.replace("$$",str(neb).zfill(3)) for k in key_templates]
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

            duplicate_matrix = np.zeros((
                len(
                    list(
                        ellerman_bomb_candidates.keys())[1::5]),
                    meta["Number"]),dtype = np.bool_)
            for bc in range(len(list(ellerman_bomb_candidates.keys())[1::5])):
                for fr in range(meta["Number"]):
                    flat_bomb_contours = set(
                            [tuple(x) for x in 
                                [pair for contour in ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[1::5][bc]][
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
                        flux = meta["Fluxes"][np.where(column)[0][0]]

                        ### Add Candidates to master dictionary ###

                        ellerman_bomb_candidates[
                                list(ellerman_bomb_candidates.keys())[0::5][bc]][i,:] = cen
                        ellerman_bomb_candidates[
                                list(ellerman_bomb_candidates.keys())[1::5][bc]][i] = con
                        ellerman_bomb_candidates[
                                list(ellerman_bomb_candidates.keys())[2::5][bc]][i] = flux
                        ellerman_bomb_candidates[
                                list(ellerman_bomb_candidates.keys())[3::5][bc]][i] = len(con)
                ### Case 2: One matched entry in the column with two in the matched row. ###
                ### This would indicate that two bombs are merging, in which case, we need to discontinue the smaller event, and transfer the contours to the larger ###

                elif (len(column[column]) == 1):
                    if (len(duplicate_matrix[:,np.where(column)[0][0]]) >= 2):
                        cen = meta["Centers"][np.where(column)[0][0]]
                        con = list(meta["Contours"][np.where(column)[0][0]])
                        flux = meta["Fluxes"][np.where(column)[0][0]]

                        row = duplicate_matrix[:,np.where(column)[0][0]]
                        parents = np.where(row)[0]
                        parent_size = np.zeros(len(parents))
                        for p in range(len(parents)):
                            flat_parent_contours = set(
                                    [tuple(x) for x in 
                                        [pair for contour in ellerman_bomb_candidates[
                                            list(ellerman_bomb_candidates.keys())[1::5][parents[p]]][
                                                startidx:i] if np.all(contour) != 0 for pair in contour]])
                            parent_size[p] = len(flat_parent_contours)/len(list(ellerman_bomb_candidates.keys())[1::5][parents[p]][
                                startidx:i])
                            big_parent = np.where(parent_size == parent_size.max())[0][0]

                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[0::5][bc]][i,:] = cen
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[1::5][bc]][i] = con
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[2::5][bc]][i] = flux
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[3::5][bc]][i] = len(con)
                ### Case 3: Two or more matached entries in the column ###
                ### This indicates a splie event! ###
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
                            flux = meta["Fluxes"][daughters[d]]

                            if daughter_size[d] == daughter_size.max():
                                ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[0::5][bc]][i,:] = cen
                                ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[1::5][bc]][i] = con
                                ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[2::5][bc]][i] = flux
                                ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[3::5][bc]][i] = len(con)
                            else:
                                cen_arr = np.zeros((len(filelist),2))
                                cen_arr[i,:] = cen
                                contour_arr = np.zeros(len(filelist),dtype = list)
                                contour_arr[i] = con
                                ext_arr = np.zeros(len(filelist))
                                ext_arr[i] = len(con)
                                flux_arr = np.zeros(len(filelist))
                                flux_arr[i] = flux

                                dict_keys = [k.replace("$$",str(neb).zfill(3)) for k in key_templates]
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
                                            list(ellerman_bomb_candidates.keys())[1::5][parents[p]]][startidx:i]
                                        if np.all(contour) != 0 for pair in contour]])
                            parent_size[p] = len(
                                    flat_parent_contours)/len(
                                            list(ellerman_bomb_candidates.keys())[1::5][parents[p]][startidx:i])
                        parent_argsort = parents[np.argsort(parent_size)]
                        daughter_argsort = daughters[np.argsort(daughter_size)]

                        for p in range(len(parent_argsort)):
                            cen = meta["Centers"][daughters[p]]
                            con = list(meta["Contours"][daughters[p]])
                            flux = meta["Fluxes"][daughters[p]]
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[0::5][
                                        parent[p]]][i,:] = cen
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[1::5][
                                        parent[p]]][i] = con
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[2::5][
                                        parent[p]]][i] = flux
                            ellerman_bomb_candidates[
                                    list(ellerman_bomb_candidates.keys())[3::5][
                                        parent[p]]][i] = len(con)
            for fr in range(duplicate_matrix.shape[1]):
                row = duplicate_matrix[:,fr]
                if not np.any(row):
                    cen = meta["Centers"][fr]
                    con = list(meta["Contours"][fr])
                    flux = meta["Fluxes"][fr]

                    cen_arr = np.zeros((len(filelist),2))
                    cen_arr[i,:] = cen
                    contour_arr = np.zeros(len(filelist),dtype = list)
                    contour_arr[i] = con
                    ext_arr = np.zeros(len(filelist))
                    ext_arr[i] = len(con)
                    flux_arr = np.zeros(len(filelist))
                    flux_arr[i] = flux

                    dict_keys = [k.replace("$$",str(neb).zfill(3)) for k in key_templates]
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
