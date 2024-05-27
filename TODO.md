# TODO

- [ ] Look into the false positives and false negatives to understand how to improve the spectrograms to make it easier for the model to learn the rumbles
- [ ] Generate bigger spectrograms to make is easier to learn the rumble patterns for the object detector
- [ ] Make a multiprocessing application that generates spectrograms with the number of cores and then run the yolov8 model to retrieve the offsets (and freq_high, freq_low) of the rumbles.
- [ ] Run random hyperparameter search on the small dataset to find parameters that could be useful
- [ ] Read into data augmentation techniques for audio data
