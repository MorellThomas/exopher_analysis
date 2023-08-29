#!/bin/python


import  image_analysis as ia

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.stats import normaltest
from scipy.stats import ttest_ind
from scipy.spatial.distance import cdist
import os
import gc
#from skimage.io import imread
from skimage.filters.thresholding import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology import thin
from skimage.morphology import skeletonize
from skimage.filters import sobel
import tifffile
import seaborn as sns
from readlif.reader import LifFile
import multiprocessing as mp

from random import randint

def group_images(images: list[list], ignore=0):
	groups = []
	_sub_group = []
	diff_old = 1
	for i,_ in enumerate(images[:-1]):
		diff = np.sum(np.abs(np.array(images[i]) ^ np.array(images[i+1])))	#bitwise or
		diff_ratio = diff / diff_old
		if ((diff_ratio <= 0.85) or (diff_ratio >= 1.15)) and (i != 0):
			if len(_sub_group) > 2:
				groups.append(_sub_group)
				#print(f" group closed! ({diff_ratio}, {i}, {len(_sub_group)})")
			#print(f" group started! ({diff_ratio}, {i}, {len(_sub_group)})")
			_sub_group = []
			continue

		_sub_group.append(i)
		diff_old = diff
	return groups

def image_picker(signal: list[list], group, ignore=0):
	plt.ion()
	fig = ia.plot_channels(signal, group, [1]*len(group), cmaps=["gray"]*len(group), interactive=True)
	#fig = ia.plot_channels(signal, group, [4000]*len(group), cmaps=["gray"]*len(group), interactive=True)
	try:
		choice = int(input("# of image to take\t"))
	except:
		choice = -1
	plt.close(fig)
#
	choice = randint(group[0], group[-1])
	#choice = int(np.mean(group))
	print(f"picked {choice}")
	return choice

def process(sig, label, index):
	min_proximity = 35	# minimum distance of aggregate signals to be from the same worm
	worm_size_cuttoff = [12_000, 100_000_000]	# minimum and maximum size of a worm

	smooth = ndi.binary_dilation(sig, iterations=10)
	#ABC
	smooth = ndi.binary_closing(smooth, iterations=min_proximity)

	skel = skeletonize(smooth)

	for _ in range(5):
		smooth = ndi.median_filter(smooth, size=15)
		skel = skeletonize(smooth)	
		

	curve = sobel(sig)
	#smooth = skel

	#ia.plot_channels([smooth]+[curve], ["1","2","3"], [1,1,1],cmaps=["gray"]*3)

	s_labels = ndi.label(smooth)[0]

	#n_cells = [ cells.max() for cells in s_labels ]
	#o = [cells.max() for cells in labels]
	#ia.plot_channels(s_labels+labels,["1", "2", "3"]*2 , [n_cells[0], n_cells[1], n_cells[2]]+o, cmaps=["jet", "jet", "jet"]*2)

	border_mask = ndi.binary_dilation(np.zeros(label.shape, dtype=bool), border_value=1)

	areas = []
	for smooth_ID in np.unique(s_labels)[1:]:

		worm_mask = s_labels == smooth_ID
		area = np.sum(worm_mask)

		border_overlap = np.sum(np.logical_and(worm_mask, border_mask))

		if (area < worm_size_cuttoff[0]) or (worm_size_cuttoff[1] < area):
			print(1,area, "area removed")
			s_labels[worm_mask] = 0

		elif border_overlap > 0:
			s_labels[worm_mask] = 0

		else:
			areas.append(area)

			
	return areas, s_labels, index
			
def smudge_worm(signal: list, labels: list, ignore=0):
	"""
	smooth the aggregate signal to approximate worm shape
	re-label the signal
	"""

	pool = mp.Pool()
	processes = [ pool.apply_async(process, args=(signal[i], labels[i], i)) for i,_ in enumerate(signal) if i >= ignore ]

	areas = [0] * (len(processes)+ignore)
	s_labels = np.array([np.array([ l * 0 for l in row]) for row in labels])
	if ignore > 0:
		area[ignore-1] = 0

	pool.close()
	pool.join()
	for p in processes:
		result = p.get()
		s_labels[result[-1]] = result[1]
		areas[result[-1]] = result[0]

	#for i,_ in enumerate(s_labels):
		#for new_ID, smooth_ID in enumerate(np.unique(s_labels[i])[ignore:]):
			#s_labels[i][s_labels[i]==smooth_ID] = new_ID
	for i,channel in enumerate(s_labels[ignore:]):
		i += ignore
		for new_ID, old_ID in enumerate(np.unique(channel)[1:]):
			s_labels[i][s_labels[i]==old_ID] = new_ID + 1

			
	#s_labels, _ = ia.re_label(s_labels)

	## here labels change 
	for i,channel in enumerate(labels[ignore:]):
		i += ignore
		for cell_ID in np.unique(channel)[1:]:	#skip background

			cell_mask = channel == cell_ID
			
			#label_overlap = np.logical_and(s_labels[i], cell_mask)
			new_label = (s_labels[i]*cell_mask).max()
			labels[i][cell_mask] = new_label if new_label > 0 else 0

	#is this even needed?
	for i,channel in enumerate(labels[ignore:]):
		i += ignore
		for new_ID, cell_ID in enumerate(np.unique(labels[i])[ignore:]):
			labels[i][labels[i]==cell_ID] = new_ID
	
	n_cells = [ cells.max() for cells in s_labels ]
	o = [cells.max() for cells in labels]
	for i,cells in enumerate(o):
		if cells == 0:
			labels.pop(i)
			areas.pop(i)
	ia.plot_channels([s_labels[0]],["a"], [n_cells[0]], cmaps=["jet"]*len(s_labels))
	#ia.plot_channels(labels,["a"]*len(labels) , o, cmaps=["jet"]*len(labels))

	return labels, areas
	

def process_results(frame: list, sq_pix: int, area: float):
	_wids = []
	_msds = []
	_nas = []
	for worm_ID in np.unique(frame)[1:]:
		#print(f"\033[FCalculating intensity results . . . {str(round((k/sum(n_cells)) * 100, 1)).zfill(2)}%")
		worm_mask = frame == worm_ID

		#_results[i]["worm_id"].append(worm_ID)
		aggregates = frame * worm_mask
		_l = ndi.label(aggregates)[0]
		#smallest_dists = []
		#for _label in np.unique(_l)[1:]:
			#_dists = []
			#_coords = [ndi.center_of_mass(_l, labels=_l, index=_label)]
			#each_other_coords = [ ndi.center_of_mass(_l, labels=_l, index=eol) for eol in np.unique(_l)[1:] if eol != _label ]
			#smallest_dists.append(np.min(cdist(_coords, each_other_coords)))
		##print(smallest_dists)
		#mean_smallest_dist = np.mean(smallest_dists)
		mean_smallest_dist = 7

		n_aggregates = ndi.label(aggregates)[0].max()
		normalised_aggregates = n_aggregates / (area[worm_ID-1] / sq_pix)
		#_results[i]["normalised_mean_#aggregate"].append(normalised_aggregates)
		#del worm_mask, aggregates, _l, smallest_dists, n_aggregates
		_wids.append(worm_ID)
		_msds.append(mean_smallest_dist)
		_nas.append(normalised_aggregates)
	return _wids, _msds, _nas
	

def calculate_results(sample_number: int, result: dict, area: list, conditions: list, labels: list, n_cells: list):
	"""
	calculate <++>
	"""

	print(f"Calculating results results . . .")
	pool = mp.Pool(mp.cpu_count()-1)

	sq_pix = [ np.mean(ar) for ar in area ]
	processes = [ pool.apply_async(process_results, args=(frame,sq_pix[i], area[i])) for i,frame in enumerate(labels) ]

	pool.close()
	pool.join()

	for i,p in enumerate(processes):
		res = p.get()
		mdist = res[1]
		nagg = res[2]
		result[conditions[sample_number-1]].append([nagg, mdist])

	return result

if __name__ == "__main__":
	gc.enable()
	"""
	main code
	"""
	files_dir, files_names = ia.setup_files("data/Gonad", ".lif")
	conditions = ["Gonad 20C", "Gonad 25C"]
	#conditions = [ "Ced1 KO", "WT"]
	#conditions = ["Test 1", "Test2"]
	n = 1
	starting_n = n
	ignore_channel = 0
	stats_for_plot = {}


	#print(files_names)

	while (files := [ file for file in files_names if file[:len(str(n))] == str(n) ]) != []:

		print(f"Analyzing sample {conditions[n-1]}")

		files = files[0]
		stats_for_plot[conditions[n-1]] = []

		spin = mp.Process(target=ia.spinning, daemon=True)
		spin.start()

		f_frame = []

		for video in LifFile(os.path.join(files_dir, files)).get_iter_image():
			frames = list(video.get_iter_t())

			#frames = frames[0:200]
			#_, signal, _, _ = ia.bg_detection(frames, fine=False)
			#del _
			#groups = group_images(signal, ignore=ignore_channel)
			groups = group_images(frames, ignore=ignore_channel)
			_frames_to_take = []
			for group in groups:
				#g_signal = [ sig for i,sig in enumerate(signal) if i in group ]
				g_signal = [ sig for i,sig in enumerate(frames) if i in group ]
				_frames_to_take.append(image_picker(g_signal, group, ignore=ignore_channel))

			f_frame += [frames[frame_index] for frame_index in _frames_to_take if frame_index > -1 ]
			#del video, frames, signal, groups
			del video, frames, groups
			gc.collect()
		
		## prolly break loop here and pickle

		f_img_smooth, f_signal, f_label, f_n = ia.bg_detection(f_frame, fine=True)
		ia.plot_channels(f_signal, ["a"]*len(f_signal), [1]*len(f_signal), cmaps=["gray"]*len(f_signal))

		f_label, area = smudge_worm(f_signal, f_label, ignore=ignore_channel)
		f_n = [ cells.max() for cells in f_label ]
		
		ia.plot_channels(f_label, ["a"]*len(f_label), f_n, cmaps=["jet"]*len(f_label))
		#ia.plot_channels([f_label[1]], ["a"], [1], cmaps=["gray"])

		stats_for_plot = calculate_results(n, stats_for_plot, area, conditions, f_label, f_n)
		spin.terminate()
		del f_frame, f_img_smooth, f_signal, f_label, f_n, area, spin
		gc.collect()

		n += 1

	results = list(stats_for_plot.values())
	#n_agg = [ [ r[0] for r in result ] for result in results ]
	#n_agg = [[s for t in n for s in t] for n in n_agg ]
	n_agg = [[s for t in n for s in t if s <= 300] for n in [ [ r[0] for r in result ] for result in results ] ]
	#mean_dist = [ [ r[1] for r in result ] for result in results ]
	#mean_dist = [[s for t in m for s in t ] for m in mean_dist ]
	mean_dist = [[s for t in m for s in t ] for m in [ [ r[1] for r in result ] for result in results ] ]

	print("Done!")

	try:
		significance = ia.test_significance(n_agg)
		ia.plot_bar_scatter(n_agg, "Number of aggregates per worm per mean worm size", "#aggregates", conditions[starting_n-1:], significance=significance)
	except:
		ia.plot_bar_scatter(n_agg, "Number of aggregates per worm per mean worm size", "#aggregates", conditions[starting_n-1:])



	try:
		significance = ia.test_significance(mean_dist)
		ia.plot_bar_scatter(mean_dist, "Mean minimum distance between aggregates per worm", "mean min distance [px]", conditions[starting_n-1:], significance=significance)
	except:
		ia.plot_bar_scatter(mean_dist, "Mean minimum distance between aggregates per worm", "mean min distance [px]", conditions[starting_n-1:])
