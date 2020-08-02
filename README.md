# extract-ifu-spectra
Python software to extract spectra from a 3D data cube (i.e. an IFU like MUSE@VLT).


prepare_extraction.py:
  Convenience function to prepare input file required by extract_spectra.py.
  Assumes that table with HST photometry exists (columns: RA, DEC, F336W, F814W)
  Adjusts the coordinate systems between HST and MUSE data. Requires the
  following defintions:
  spec_min_mag: faintest star for which a spectrum should be extracted
  min_mag: faintest star that should be considered in the extraction of a
           spectrum of another star
  crit_dist: pixel distance to a second star up to which it is considered in the
             PSF fitting
  max_star_number: maximum number of stars that are considered for one extraction
                   (important for the computation time of the extraction!)


extract_spectra.py [infile]
  Does the actual extraction via PSF fitting
  Requires an input file containing id,x,y,ra,dec,f336_mag,f814_mag (as created by prepare_extraction.py)
  Performs PSF fitting at a fixed position for the first star in the input file but simultaneously fitting
   the PSFs of all other stars in the input file to take into account their contributions
  
Dependencies:

*Author:*
Julia Bodensteiner
PhD student
Institute of Astronomy, KU Leuven
Celestijnenlaan 200D
3001 Leuven
Belgium
email: julia.bodensteiner@kuleuven.be

Date: 2nd of Aug, 2020
