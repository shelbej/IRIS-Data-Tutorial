import os
import numpy as np
import astropy.io.fits as pyfits
import astropy.constants as const
from scipy import interpolate as interp
from helita.utils.utilsfast import peakdetect, peakdetect2, replace_nans
from helita.utils.utilsmath import peakdetect_lcl
from helita.utils.shell import progressbar
import rh15d


class MgStatsIRIS:
    def __init__(self, infile, interp=True,  vrange=[-40, 40], verbose=False):
        """
        As MgStats but for use with real IRIS data (level 1, 1.5 or 2).
        """
        self.interp = interp
        self.vrange = np.array(vrange)
        self.verbose = verbose
        self.infile = infile
        self.lineprop = False
        # wavelengths in vacuum for IRIS data
        self.lines = [['k', 'MgII_k', 279.63509493726883 ],
                      ['h', 'MgII_h', 280.35297192113524 ]]
        self.feat = ['bp', 'lc', 'rp']
        for l in self.lines:
            setattr(self, l[0], rh15d.DataHolder())
            ll = getattr(self, l[0])
            ll.name = l[1]
            ll.wave_ref = l[2]
            for ft in self.feat:
                setattr(ll, ft, rh15d.DataHolder())

    def calc_lineprop(self):
        """
        Calculates the line properties (bp, lc, rp) for  Mg II h & k.
        """
        #p = [0, 15, 3, 2, 1]
        p = [2.e-11, 15, 3, 2, 1]
        prop_func = mg_properties_conv
        # Get Mg II spec according to data level
        fobj = pyfits.open(self.infile)
        try:
            lvl_num = fobj[0].header['LVL_NUM']
        except KeyError:
            if 'LVL_NUM' in fobj[1].header:
                lvl_num = fobj[1].header['LVL_NUM']
            else:
                raise ValueError('IRIS data level not found, aborting.')
        assert lvl_num >= 1
        if lvl_num == 1.:     # not tested, use with care. Missing wavelength.
            spec = fobj[1].data
            idx = get_img_window(spec, thres=30)
            spec = spec[idx[0]:idx[1], idx[2]:idx[3]]
        elif lvl_num == 1.5:  # not tested, use with care. Missing wavelength.
            spec = fobj[0].data
            idx = get_img_window(spec, thres=30)
            spec = spec[idx[0]:idx[1], idx[2]:idx[3]]
        elif lvl_num == 2:
            mg_idx = 1
            while True:   # Find location of Mg II spectrum
                cur_key = 'TDESC' + str(mg_idx)
                try:
                    line = fobj[0].header[cur_key]
                    if line == 'Mg II k 2796':
                        break
                    mg_idx += 1
                except KeyError:
                    raise ValueError('Could not find Mg II spectra in file.')
            spec = fobj[mg_idx].data.astype('d')
            wave_min = fobj[0].header['TWMIN' + str(mg_idx)]
            wave_max = fobj[0].header['TWMAX' + str(mg_idx)]
            wave_step = fobj[mg_idx].header['CDELT1']
            # convert from AA to nm
            wave = np.arange(wave_min, wave_max + wave_step, wave_step) / 10.
        else:
            raise(ValueError('calc_lineprop: invalid IRIS data level: %s' %
                             lvl_num))
        fobj.close()
        self.nx, self.ny = spec.shape[:2]

        for l in self.lines:
            line = getattr(self, l[0])
            if self.verbose:
                print('Going line ' + line.name)
            if line.name == 'MgII_k':    # for line guess to work for h line
                line.lc.vlambda = [None]
            if self.interp:
                vaxis = const.c.to('km/s').value * (wave - line.wave_ref) / line.wave_ref
                wi = max(np.where(vaxis >= self.vrange[0])[0][0]  - 1, 0)
                wf = min(np.where(vaxis <= self.vrange[1])[0][-1] + 1,
                         vaxis.shape[0] - 1)
                wvl = np.linspace(wave[wi - 2], wave[wf + 2], 300)
                f = interp.interp1d(wave[wi - 3:wf + 3],
                                    spec[:, :, wi - 3:wf + 3], kind='cubic')
                spc = f(wvl)
            else:
                spc = spec
                wvl = wave
            # multiply vrange by 3 to avoid a double cut on the last point
            result = prop_func(wvl, spc, line.wave_ref, vrange=self.vrange,
                               delta=p[0], pmin=p[1], pfit=p[2], gsigma=p[3],
                               gn=p[4], graph=False,
                               guess_in=self.k.lc.vlambda[0], single_spec=False)
            for i, ft in enumerate([getattr(line, a) for a in self.feat]):
                ft.vlambda = result[i][0]
                ft.i = result[i][1]
        del spec
        del spc
        del wave
        self.lineprop = True

    def write_data(self, outfile):
        """
        Writes this structure into a NetCDF file
        """
        import netCDF4
        # write skeleton structure
        f = netCDF4.Dataset(outfile, 'w')
        groups = {}
        for l in self.lines:
            groups[l[0]] = f.createGroup(l[1])
            for ft in self.feat:
                groups[ft + l[0]] = groups[l[0]].createGroup(ft)

        if self.lineprop:
            f.createDimension('nx', self.nx)
            f.createDimension('ny', self.ny)
            for l in self.lines:
                ll = getattr(self, l[0])
                g = groups[l[0]]
                # cycle over variables, skip internal methods and 'groups'
                for v in [a for a in dir(ll) if ((a[0] != '_') and
                                                 (a not in self.feat))]:
                    buf = getattr(ll, v)
                    # if string or number, write as attribute.
                    if type(buf) in [type(''), type(0.), type(0)]:
                        setattr(g, v, buf)
                    else:
                        var = g.createVariable(v, 'f4', ('nx', 'ny'))
                        var[:] = buf
                for ft in self.feat:
                    ff = getattr(ll, ft)
                    gg = groups[ft + l[0]]
                    for v in [a for a in dir(ff) if a[0] != '_']:
                            var = gg.createVariable(v, 'f4', ('nx', 'ny'))
                            var[:] = getattr(ff, v)
        f.close()

    def read_data(self, infile):
        """
        Reads data from file, written with write_data
        """
        if os.path.splitext(infile)[1].lower() == '.hdf5':
            import h5py
            f = h5py.File(infile, 'r')
            for l in self.lines:
                ll = getattr(self, l[0])
                g = f[l[1]]
                gvariables = [v for v, a in g.iteritems()
                              if type(a) == h5py._hl.dataset.Dataset]
                for v in gvariables:
                    buf = g[v][:]
                    setattr(ll, v, np.ma.MaskedArray(buf, mask=(buf==1e20)))
                for ft in self.feat:
                    ff = getattr(ll, ft)
                    gg = g[ft]
                    ggvariables = [v for v, a in gg.iteritems()
                                   if type(a) == h5py._hl.dataset.Dataset]
                    for v in ggvariables:
                        buf = gg[v][:]
                        setattr(ff, v, np.ma.MaskedArray(buf, mask=(buf==1e20)))
            f.close()
        elif os.path.splitext(infile)[1].lower() == '.ncdf':
            import netCDF4
            f = netCDF4.Dataset(infile, 'r')
            for l in self.lines:
                ll = getattr(self, l[0])
                g = f.groups[l[1]]
                for v in g.variables:
                    setattr(ll, v, g.variables[v][:])
                for ft in self.feat:
                    ff = getattr(ll, ft)
                    gg = g.groups[ft]
                    for v in gg.variables:
                        setattr(ff, v, gg.variables[v][:])
            f.close()
        else:
            raise(ValueError('read_data: file type not known.'))
        self.lineprop = True
        self.nx, self.ny = getattr(getattr(self, self.lines[0][0]),
                                   self.feat[0]).i.shape


def mg_properties(wave_in, spec_in, wave_ref, vrange=[-60, 60], graph=False,
                  lookahead=2, delta=5e-12, pmin=5, pfit=2, gsigma=6, gn=3,
                  glc=2, **kwargs):
    '''
    Extracts the line properties (blue peak, line centre, red peak)
    of a Mg II line (h or k).

    Parameters
    ----------
    wave - 1D array
        array of wavelengths/velocities
    spec - nD array
        spectral array, wavelength in last index
    vrange - list
        Doppler velocity limits around wave_ref, determing the range that
        will be analysed
    graph - boolean
        If True, will plot some results in interactive fashion.
    lookahead - integer
        Parameter for peak finding, how many points to look ahead for peaks
    delta - float
        Minimum threshold for peak in spec units
    pmin - integer
        Number of points to find around the minimum
    pfit - integer
        Number of points to fit a parabola around the minimum
    gsigma - integer
        Gaussian sigma (in pixels) to smooth velocities and find outliers
    gn - integer
        Kernel size for replace_nans
    glc - integer
        Gaussian sigma (in pixels) to smooth velocities and make final mask
        for lc.

    Returns
    -------
        results - tuple with results, bp, lc, rp

    --Tiago, 20130414
    '''
    import scipy.ndimage as ndimage

    # flatten spectrum
    shape_in = spec_in.shape[:-1]
    spec = np.reshape(spec_in, (np.prod(shape_in), spec_in.shape[-1]))

    # best parameters
    # full resolution: gsigma=6, gn=3, glc=2
    # 252 resolution: gsigma=2, gn=2
    #gsigma = 6
    #gn = 3
    #glc = 2

    # Determine wavelength range to use
    vaxis = const.c.to('km/s').value * (wave_in - wave_ref) / wave_ref
    wi = max(np.where(vaxis >= vrange[0])[0][0]  - 1, 0)
    wf = min(np.where(vaxis <= vrange[1])[0][-1] + 1, vaxis.shape[0] - 1)
    vel  = vaxis[wi:wf]
    spec = spec[:,wi:wf]
    nspec = len(spec)

    lc = np.ma.masked_all((2, nspec))   # line core (central reversal, k3 or h3)
    bp = np.ma.masked_all((2, nspec))   # blue peak
    rp = np.ma.masked_all((2, nspec))   # red peak

    # LINE CORE, main loop
    for i in range(nspec):
        guess = 5.
        pts_min = pmin

        xi = i // shape_in[-1]
        yi = i % shape_in[-1]

        lc[0, i], lc[1, i] = mg_single(vel, spec[i], guess, delta=delta,
                                       loc=(xi, yi), lookahead=lookahead,
                                       ppts=pfit,pts_min=pts_min, graph=graph)
        #lc[0, i], lc[1, i] = mg_single_jorrit(vel, spec[i])

    #lc = np.reshape(lc, (2,) + shape_in)
    #return lc

    # two iterations for cleaning up
    for n in range(2):
        print('Rerun %i' % (n + 1))

        # put back in initial shape
        lc = np.reshape(lc, (2,) + shape_in)

        # find outliers,distant more than 3 km/s from a smoothed Gaussian
        xpto = lc[0].data.copy()
        xpto[lc[0].mask] = 0.
        diff = np.abs(xpto - ndimage.gaussian_filter(xpto, gsigma, mode='wrap'))
        lc[:, diff > 3] = np.nan
        lc.data[:, lc[0].mask] = np.nan   # so that replace_nans will work

        # perform inpainting on masked image to get next best guess
        lc_guess = replace_nans(lc[0].data, 10, .5, gn, 'localmean')

        # put back into flat shape, redo loop with new guess
        lc = np.reshape(lc, (2, np.prod(shape_in)))
        lc_guess = np.reshape(lc_guess, (np.prod(shape_in)))

        for i in range(nspec):
            if not np.isfinite(lc[0, i]):
                lc[0, i], lc[1, i] = \
                    mg_single(vel, spec[i], lc_guess[i], ppts=pfit,
                              lookahead=lookahead, loc=(xi, yi), delta=delta,
                              pts_min=pmin, graph=graph, force_guess=True)

    # get better lc estimate for peak detection
    lc = np.reshape(lc, (2,) + shape_in)
    xpto = lc[0].data.copy()
    xpto[lc[0].mask] = 0.
    diff = np.abs(xpto - ndimage.gaussian_filter(xpto, 10. * gsigma / 6,
                                                mode='wrap'))
    xpto[(diff > 4) | lc[0].mask] = np.nan
    # perform inpainting on masked image to get next best guess
    lc_guess = replace_nans(xpto, 10, .5, 2, 'localmean')
    lc_guess = np.reshape(lc_guess, (np.prod(shape_in)))

    # PEAKS, main loop
    print('Peaks main loop...')
    for i in range(nspec):
        bp[:, i], rp[:, i] = mg_peaks_single(vel, spec[i], lc_guess[i],
                                             delta=5e-12,lookahead=lookahead,
                                             graph=graph, fast=False)

    # Fix lc by using minimum between the two peaks, when the peaks are identified
    print('Fix lc between peaks...')
    xpto = lc[0].data.copy()
    xpto[np.isnan(xpto) | lc[0].mask] = 0.
    diff = np.abs(xpto - ndimage.gaussian_filter(xpto, glc, mode='wrap')) > 4
    new_mask = ndimage.binary_dilation(diff, iterations=int(gsigma + 1))
    new_mask = np.reshape(ndimage.binary_fill_holes(new_mask), np.prod(shape_in))
    new_mask = new_mask & ~(rp[0].mask | bp[0].mask)
    lc = np.reshape(lc, (2, np.prod(shape_in)))

    lc = np.reshape(lc, (2,) + shape_in)
    bp = np.reshape(bp, (2,) + shape_in)
    rp = np.reshape(rp, (2,) + shape_in)
    return bp, lc, rp

    # find outliers,distant more than 3 km/s from a smoothed Gaussian
    xpto = bp[0].data.copy()
    xpto[bp[0].mask] = 0.
    diff_bp = np.abs(xpto - ndimage.gaussian_filter(xpto, 2, mode='wrap'))
    xpto = rp[0].data.copy()
    xpto[rp[0].mask] = 0.
    diff_rp = np.abs(xpto - ndimage.gaussian_filter(xpto, 2, mode='wrap'))
    peaks_mask = np.reshape((diff_bp > 3) | (diff_rp > 3), (np.prod(shape_in)))
    bp = np.reshape(bp, (2, np.prod(shape_in)))
    rp = np.reshape(rp, (2, np.prod(shape_in)))

    # repeat, now with spectrally smoothed version on problematic pixels
    for i in range(nspec):
        xi = i // shape_in[-1]
        yi = i % shape_in[-1]
        #if not np.all(np.isfinite([bp[0, i], rp[0, i]])):
        if peaks_mask[i]:
            bp[:, i], rp[:, i] = \
              mg_peaks_single(vel, spec[i], lc[:, i], delta=5e-12, smooth=True,
                              loc=(xi, yi), lookahead=lookahead, graph=graph)

    lc = np.reshape(lc, (2,) + shape_in)
    bp = np.reshape(bp, (2,) + shape_in)
    rp = np.reshape(rp, (2,) + shape_in)

    return bp, lc, rp


def mg_properties_conv(wave_in, spec_in, wave_ref, vrange=[-60, 60],
                       graph=False, lookahead=10, delta=2e-11, pmin=5, pfit=2,
                       gsigma=6, gn=3, single_spec=False, **kwargs):
    '''
    Extracts the line properties (blue peak, line centre, red peak)
    of a Mg II line (h or k).

    Parameters
    ----------
    wave - 1D array
        array of wavelengths/velocities
    spec - nD array
        spectral array, wavelength in last index
    vrange - list
        Doppler velocity limits around wave_ref, determing the range that
        will be analysed
    graph - boolean
        If True, will plot some results in interactive fashion.
    lookahead - integer
        Parameter for peak finding, how many points to look ahead for peaks
    delta - float
        Minimum threshold for peak in spec units
    pmin - integer
        Number of points to find around the minimum
    pfit - integer
        Number of points to fit a parabola around the minimum
    gsigma - integer
        Gaussian sigma (in pixels) to smooth velocities and find outliers
    gn - integer
        Kernel size for replace_nans
    single_spec - bool
        If True, will do the inpainting and outlier identification for each
        spectrogram independently (ie, only along the second dimension). To
        be used with IRIS data.

    Returns
    -------
        results - tuple with results, bp, lc, rp

    Notes
    -----

    What is different from the non-conv version? First, mg_single_conv is
    used, which takes greater care of points where the peaks are blended.
    In such cases a guess at lc is not attempted until the last step, where
    the derivative minimum method is used. Other things have been tweaked
    slightly to work better on convolved (and interpolated!) spectra.
    If interpolation is not used, it may not work reliably.

    --Tiago, 20130414
    '''
    import scipy.ndimage as ndimage

    # flatten spectrum
    nx = spec_in.shape[0]
    shape_in = spec_in.shape[:-1]
    spec = np.reshape(spec_in, (np.prod(shape_in), spec_in.shape[-1]))
    # best parameters
    # full resolution: gsigma=6, gn=3, glc=2
    # 252 resolution: gsigma=2, gn=2
    #gsigma = 6
    #gn = 3
    #glc = 2

    # Determine wavelength range to use
    vaxis = const.c.to('km/s').value * (wave_in - wave_ref) / wave_ref
    wi = max(np.where(vaxis >= vrange[0])[0][0]  - 1, 0)
    wf = min(np.where(vaxis <= vrange[1])[0][-1] + 1, vaxis.shape[0] - 1)
    vel  = vaxis[wi:wf]
    spec = spec[:,wi:wf]
    nspec = len(spec)

    lc = np.ma.masked_all((2, nspec))   # line core (central reversal, k3 or h3)
    bp = np.ma.masked_all((2, nspec))   # blue peak
    rp = np.ma.masked_all((2, nspec))   # red peak

    # LINE CORE, main loop
    for i in range(nspec):
        guess = 5. * 0.
        pts_min = pmin
        xi = i // shape_in[-1]
        yi = i % shape_in[-1]
        progressbar(i, nspec - 1)
        lc[0, i], lc[1, i] = mg_single_conv(vel, spec[i], guess, delta=delta,
                                       loc=(xi, yi), lookahead=lookahead,
                                       ppts=pfit,pts_min=pts_min, graph=graph,
                                       use_deriv=False)
    # two iterations for cleaning up
    for n in range(2):
        print('Rerun %i' % (n + 1))

        # put back in initial shape
        lc = np.reshape(lc, (2,) + shape_in)
        if single_spec:   # Do each spectrogram independently
            lc_guess = np.zeros_like(lc[0].data)
            for i in range(nx):
                xpto = lc[0, i].data.copy()
                xpto[lc[0, i].mask] = 0.
                diff = np.abs(xpto - ndimage.gaussian_filter(xpto, gsigma,
                                                             mode='wrap'))
                lc[:, i, diff > 3] = np.nan
                lc.data[:, i, lc[0, i].mask] = np.nan
                # perform inpainting on masked image to get next best guess
                lc_guess[i] = replace_nans(lc[0, i].data[np.newaxis],
                                                     10, .5, gn, 'localmean')
        else:
            # find outliers,distant more than 3 km/s from a smoothed Gaussian
            xpto = lc[0].data.copy()
            xpto[lc[0].mask] = 0.
            diff = np.abs(xpto - ndimage.gaussian_filter(xpto, gsigma, mode='wrap'))
            lc[:, diff > 3] = np.nan
            lc.data[:, lc[0].mask] = np.nan   # so that replace_nans will work

            # perform inpainting on masked image to get next best guess
            lc_guess = replace_nans(lc[0].data, 10, .5, gn, 'localmean')

        # put back into flat shape, redo loop with new guess
        lc = np.reshape(lc, (2, np.prod(shape_in)))
        lc_guess = np.reshape(lc_guess, (np.prod(shape_in)))

        for i in range(nspec):
            if not np.isfinite(lc[0, i]):
                lc[0, i], lc[1, i] = \
                    mg_single_conv(vel, spec[i], lc_guess[i], ppts=pfit,
                              lookahead=lookahead, loc=(xi, yi), delta=delta,
                              pts_min=pmin, graph=graph, force_guess=True)
    print('Last run with use_deriv')
    for i in range(nspec):
        if not np.isfinite(lc[0, i]):
            xi = i
            lc[0, i], lc[1, i] = \
                    mg_single_conv(vel, spec[i], guess, delta=0.,loc=(xi, yi),
                                   lookahead=lookahead, ppts=pfit, graph=graph,
                                   pts_min=pts_min, use_deriv=True)
    lc = np.reshape(lc, (2,) + shape_in)
    # clean up some isolated problem pixels
    xpto = lc[0].data.copy()
    xpto[lc[0].mask] = 40.
    idx = np.abs(xpto - ndimage.gaussian_filter(xpto, 1., mode='wrap')) > 10
    lc[:, idx] = np.nan
    lc[0] = replace_nans(lc[0].data, 10, .5, gn, 'localmean')
    lc[1] = replace_nans(lc[1].data, 10, .5, gn, 'localmean')
    # get mask for bad values
    bad_mask = ndimage.binary_fill_holes(np.abs(xpto -
                          ndimage.gaussian_filter(xpto, 1.7, mode='wrap')) > 6)
    lc[:, bad_mask] = np.ma.masked

    # get better lc estimate for peak detection
    lc = np.reshape(lc, (2,) + shape_in)
    xpto = lc[0].data.copy()
    xpto[lc[0].mask] = 0.
    diff = np.abs(xpto - ndimage.gaussian_filter(xpto, 10. * gsigma / 6,
                                                mode='wrap'))
    xpto[(diff > 4) | lc[0].mask] = np.nan
    # perform inpainting on masked image to get next best guess
    lc_guess = replace_nans(xpto, 10, .5, 2, 'localmean')
    lc_guess = np.reshape(lc_guess, (np.prod(shape_in)))

    # PEAKS, main loop
    print('Peaks main loop...')
    for i in range(nspec):
        progressbar(i, nspec - 1)
        bp[:, i], rp[:, i] = mg_peaks_single(vel, spec[i], lc_guess[i],
                                             delta=5e-12,lookahead=lookahead,
                                             graph=graph, fast=False)
    lc = np.reshape(lc, (2,) + shape_in)
    bp = np.reshape(bp, (2,) + shape_in)
    rp = np.reshape(rp, (2,) + shape_in)
    return bp, lc, rp



def mg_single(vel, spec, guess, lookahead=2, delta=5e-12, pts_min=5, ppts=2,
              graph=False, loc=(0,0), force_guess=False):
    ''' Calculates the properties for a single Mg II spectrum. Similar to
    specquant.linecentre, but also does peakdetect and everything more streamlined.

    '''
    from scipy import ndimage
    import matplotlib.pyplot as plt
    #
    # More conditions to put:
    #
    # * 2 max, 2 min: pick the min that is in between two maxima (if any)
    #
    #

    if hasattr(spec[0], 'mask') or np.isnan(guess):
        return np.ma.masked, np.ma.masked

    if graph:
        plt.clf()
        plt.plot(vel, spec)

    #guess = np.ma.masked
    #imin = np.ma.masked

    # Use peak detect unless forcing the use of supplied guess
    if not force_guess:
        pmax, pmin = peakdetect2(spec, vel, lookahead_max=2, lookahead_min=2,
                                delta_max=delta, delta_min=delta)

        if np.any(pmax) and np.any(pmin):
            ## manually remove the extra line
            #pmax = pmax[:, pmax[0] > -35]
            #pmin = pmin[:, pmin[0] > -35]
            lpmax = len(pmax[0])
            lpmin = len(pmin[0])

            if graph:
                plt.plot(pmax[0], pmax[1], 'ro')
                plt.plot(pmin[0], pmin[1], 'go')
            # was [(1,2), (3, 1), (3, 2), (3, 4), (5, 4), (7,6)]
            if (lpmin, lpmax) in [(1,2), (3, 1), (3, 2), (3, 4), (5, 4), (7,6)]: # added (3, 1)
                # most straightforward case: take the middle one
                guess = pmin[0][lpmin//2]
                imin = pmin[1][lpmin//2]
                pts_min = pts_min//2

            # new rule, 20130306
            elif (lpmin, lpmax) in [(2, 2), (2, 3), (3, 3), (4, 2), (4, 3), (4, 4)]:
                pts_min = pts_min//2
                # take the lowest minimum between the maxima
                idx = (pmin[0] > pmax[0,0]) & (pmin[0] < pmax[0,-1])
                if np.any(idx):
                    guess = pmin[0, idx][np.argmin(pmin[1, idx])]
                    imin = pmin[1, idx][np.argmin(pmin[1, idx])]

            elif lpmin == 2 and lpmax == 1: # was [1, 3]
                if np.max(pmin[1])/np.min(pmin[1]) > 1.3:
                    # use lowest minimum
                    guess = pmin[0, np.argmin(pmin[1])]
                    imin = pmin[1, np.argmin(pmin[1])]
                    pts_min = pts_min//2
                else:
                    # Take the minimum furthest away from the highest peak
                    guess = pmin[0, np.argmax(np.abs(pmin[0] - pmax[0, np.argmax(pmax[1])]))]
                    imin = pmin[1, np.argmin(pmin[1])]
                    pts_min = pts_min//2

            elif lpmin == 1: # and np.abs(pmin[0,0]) < 20.:
                guess = pmin[0,0]
                imin = pmin[1,0]
                pts_min = pts_min//2
            #elif lpmax == 1 and lpmin == 2:
            #    # take centroid in between lows
            #    idx = (wave >= pmin[0,0]) & (wave <= pmin[0,1])
            #    guess = np.trapz(wave[idx]*spec[idx], x=wave[idx])/np.trapz(spec[idx], x=wave[idx])

    if graph:
        plt.axvline(x=guess, color='r', ls='--')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))
        #raw_input()

    # no parabolic fit
    #return guess, imin

    # Approximate index of guess and of spectral minimum around it
    try:
        idg = np.argmin(np.abs(vel - guess))
        ini = np.argmin(spec[idg - pts_min:idg + pts_min]) + idg - pts_min
    except ValueError:
        print(idg, pts_min)
        print(pmax, pmin)
        print(guess, loc)
        print(spec[idg - pts_min:idg + pts_min])
        return np.ma.masked, np.ma.masked

    # if no points, return masked
    if len(vel[ini - ppts:ini + ppts + 1]) == 0:
        return np.ma.masked, np.ma.masked

    # Fit parabola
    fit = np.polyfit(vel[ini - ppts:ini + ppts + 1],spec[ini - ppts:ini + ppts + 1],2)
    # Convert poly to parabola coefficients
    lc     = -fit[1] / (2 * fit[0])
    lc_int = fit[2] - fit[0] * lc**2

    # If fitted minimum is furthen than 4 wavelenght points, redo the fit
    # using a bigger area (add 2 points to ppts) and using always quad_lsq.
    # This is a last resort measure, and line minimum intensity will be
    # affected, especially in weak lines
    if np.abs(lc - vel[ini]) > 4 * (vel[1] - vel[0]):
        lc     = np.ma.masked
        lc_int = np.ma.masked

    if graph:
        plt.axvline(x=lc, color='g')
        plt.axvline(x=guess, color='r', ls='--')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))
        raw_input()

    return lc, lc_int


def mg_single_conv(vel, spec, guess, lookahead=2, delta=5e-12, pts_min=5,
                    margin=15, ppts=2, graph=False, loc=(0,0),
                    force_guess=False, use_deriv=False):
    """
    Calculates the properties for a single Mg II spectrum. Similar to
    specquant.linecentre, but also does peakdetect and everything more
    streamlined.

    For the conv spectra.
    """
    from scipy import ndimage
    import matplotlib.pyplot as plt

    if hasattr(spec[0], 'mask') or np.isnan(guess):
        return np.ma.masked, np.ma.masked

    if graph:
        plt.clf()
        plt.plot(vel, spec)

    deriv_flag = False
    pmax, pmin = peakdetect2(spec, vel, delta_max=delta, delta_min=delta,
                 lookahead_max=lookahead,
                 lookahead_min=lookahead)

    # Use peak detect unless forcing the use of supplied guess
    if not force_guess:
        # remove edges
        #if np.abs(pmax[0, 0] - vel[0]) < 5.:
        #    pmax = pmax[:,1:

        if np.any(pmax) and np.any(pmin):
            pmax = pmax[:, (pmax[0] > vel[0] + 10) & (pmax[0] < vel[-1] - 10.)]
            pmin = pmin[:, (pmin[0] > vel[0] + 10) & (pmin[0] < vel[-1] - 10.)]
            lpmax = len(pmax[0])
            lpmin = len(pmin[0])

            if graph:
                plt.plot(pmax[0], pmax[1], 'ro')
                plt.plot(pmin[0], pmin[1], 'go')

            if (lpmin, lpmax) in [(1,2), (3, 1), (3, 2), (3, 4), (5, 4), (7,6)]:
                # most straightforward case: take the middle one
                guess = pmin[0][lpmin//2]
                imin = pmin[1][lpmin//2]
                pts_min = pts_min//2
            elif (lpmin, lpmax) in [(2, 2), (2, 3), (3, 3), (4, 2), (4, 3), (4, 4)]:
                pts_min = pts_min//2
                # when in absorption, take lowest minimum
                #if np.mean(spec[:2])/np.min(pmin[1]) > 4:
                #    guess = pmin[0,np.argmin(pmin[1])]
                # take the lowest minimum between the two largest maxima
                # locations of two largest maxima
                lmax = np.sort(pmax[0][np.argsort(pmax[1])[-2:]])
                # minima inside the above window
                idx = (pmin[0] > lmax[0]) & (pmin[0] < lmax[1])
                if np.any(idx):
                    guess = pmin[0, idx][np.argmin(pmin[1, idx])]
                    imin = pmin[1, idx][np.argmin(pmin[1, idx])]
            elif lpmin == 2 and lpmax == 1: # was [1, 3]
                if np.max(pmin[1])/np.min(pmin[1]) > 1.3:
                    # use lowest minimum
                    guess = pmin[0, np.argmin(pmin[1])]
                    imin = pmin[1, np.argmin(pmin[1])]
                    pts_min = pts_min//2
                else:
                    deriv_flag = True
                    ## Take the minimum furthest away from the highest peak
                    #guess = pmin[0, np.argmax(np.abs(pmin[0] - pmax[0, np.argmax(pmax[1])]))]
                    #imin = pmin[1, np.argmin(pmin[1])]
                    #pts_min = pts_min//2
            elif lpmax == 1:
                deriv_flag = True
            elif lpmin == 1: # and np.abs(pmin[0,0]) < 20.:
                guess = pmin[0,0]
                imin = pmin[1,0]
                pts_min = pts_min//2
        elif np.any(pmax):
            lpmax = len(pmax[0])
            if lpmax == 1:
                deriv_flag = True
    else:
        if np.any(pmax):
            pmax = pmax[:, (pmax[0] > vel[0] + 20) & (pmax[0] < vel[-1] -20.)]
        if np.any(pmax):
            lpmax = len(pmax[0])
            if lpmax == 1:
                deriv_flag = True

    if graph:
        plt.axvline(x=guess, color='r', ls='--')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))
        #raw_input()

    if force_guess:
        deriv_flag = False

    #deriv_flag = True  # DELETE!!!


    if deriv_flag:   # special case to use the derivatives
        if not use_deriv:
            return np.ma.masked, np.ma.masked
        dd = np.abs(np.diff(spec))
        # define the boundaries to inspect for peak asymmetry and derivative
        incr = int(15. / (vel[1] - vel[0]))
        idxm = np.argmin(np.abs(vel - pmax[0, 0]))
        vv = [max(0, idxm - incr), min(idxm + incr, spec.shape[0] - 1)]
        vv2 = [idxm - incr * 1.3, idxm + incr * 1.3]
        vv2 = vv
        if spec[vv[0]] > spec[vv[1]]:   # k3/h3 peak on left side
            try:
                der = dd[vv2[0] + margin:idxm - margin]
                pidx = vv2[0] + np.argmin(der) + margin + 1
            except:
                return np.ma.masked, np.ma.masked
            #print vel[1], vel[0], incr, vv2, margin, idxm
            #raise ValueError
        else:                           # k3/h3 peak on right side
            der = dd[idxm + margin:vv2[1] - margin]
            pidx = idxm + np.argmin(der) + margin + 1
        lc = vel[pidx]
        lc_int = spec[pidx]
        #print('USE_DERIV %i' % loc[0])
    else:
        # Approximate index of guess and of spectral minimum around it
        try:
            idg = np.argmin(np.abs(vel - guess))
            ini = np.argmin(spec[idg - pts_min:idg + pts_min]) + idg - pts_min
        except ValueError:
            print(idg, pts_min)
            print(pmax, pmin)
            print(guess, loc)
            print(spec[idg - pts_min:idg + pts_min])
            return np.ma.masked, np.ma.masked

        # if no points, return masked
        if len(vel[ini - ppts:ini + ppts + 1]) == 0:
            return np.ma.masked, np.ma.masked
        # Fit parabola
        fit = np.polyfit(vel[ini - ppts:ini + ppts + 1],
                        spec[ini - ppts:ini + ppts + 1],2)
        # Convert poly to parabola coefficients
        lc     = -fit[1] / (2 * fit[0])
        lc_int = fit[2] - fit[0] * lc**2

        # If fitted minimum is furthen than 4 wavelenght points, redo the fit
        # using a bigger area (add 2 points to ppts) and using always quad_lsq.
        # This is a last resort measure, and line minimum intensity will be
        # affected, especially in weak lines
        if np.abs(lc - vel[ini]) > 4 * (vel[1] - vel[0]):
            lc     = np.ma.masked
            lc_int = np.ma.masked

    if graph:
        plt.axvline(x=lc, color='g')
        plt.axvline(x=guess, color='r', ls='--')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))
        raw_input()

    return lc, lc_int


def mg_peaks_single(vel, spec, lc, lookahead=10, delta=5e-12, graph=False,
                    loc=(0,0), fast=False, pp=True, smooth=False):
    """
    Calculate the intensity and velocity of the two peaks (h2v and h2r,
    or k2v and k2r)
    """
    from scipy import ndimage, interpolate
    import matplotlib.pyplot as plt

    bp = np.ma.masked_all(2)
    rp = np.ma.masked_all(2)

    if hasattr(spec[0], 'mask'):
        return bp, rp


    if graph:
        plt.clf()
        plt.plot(vel, spec)

    if smooth:
        y = ndimage.convolve1d(spec, np.ones(5)/5, mode='nearest')
    else:
        y = spec

    pmax, pmin = peakdetect_lcl(y, vel, lookahead=lookahead)

    if np.any(pmax):
        # remove peaks more than 30 km/s from line centre
        if np.isfinite(lc):
            pmax = pmax[:, np.abs(pmax[0] - lc) < 50]
        else:
            pmax = pmax[:, np.abs(pmax[0] - 0) < 50]

        lpmax = len(pmax[0])
        lpmin = len(pmin[0])

        if graph:
            plt.plot(pmax[0], pmax[1], 'ro')
            plt.plot(pmin[0], pmin[1], 'go')

        # SELECTION OF TWO PEAKS
        if lpmax > 4:   # too many maxima, take inner 4
            pmax = pmax[:, lpmax // 2 - 2 : lpmax // 2 + 2]
            lpmax = 4

        if lpmax == 1:
            if pmax[0, 0] > lc:
                rp = pmax[:, 0]
            else:   # by default assign single max to blue peak if no core
                bp = pmax[:, 0]
        elif lpmax == 2:
            bp = pmax[:, 0]
            rp = pmax[:, 1]
        elif lpmax == 3:   # this kind of behaviour can also happen with 4 peaks!
            # take the one close to the line core, and from the
            # remaining two chose the strongest
            if pmax[0, 0] < lc < pmax[0, 1]:
                bp = pmax[:, 0]
                rp = pmax[:, np.argmax(pmax[1, 1:]) + 1]
            elif pmax[0, 1] < lc < pmax[0, 2]:
                bp = pmax[:, np.argmax(pmax[1, :-1])]
                rp = pmax[:, -1]
            else:
                # if line centre is not between maxima, take two strongest
                #aa = pmax[0][pmax[1] != np.min(pmax[1])]
                #bp_guess = aa[0]
                #rp_guess = aa[1]
                aa = pmax[:, pmax[1] != np.min(pmax[1])]
                bp = aa[:, 0]
                rp = aa[:, 1]
        elif lpmax == 4:
            # first look for special case when two close weak inner peaks
            # are taken as peaks. In this case take the outer peaks.
            sep_out = pmax[0, 3] - pmax[0, 0]
            sep_in = pmax[0, 2] - pmax[0, 1]
            rt = (pmax[1, 0] > 1.06 * pmax[1, 1]) and \
                 (pmax[1, 3] > 1.06 * pmax[1, 2])
            if (sep_out < 40) and (sep_in < 13) and rt:
                bp = pmax[:, 0]
                rp = pmax[:, 3]
            # if line core in the middle of the four, or outside the
            # maxima region, take inner two maxima
            elif (lc > pmax[0, 1]) and (lc < pmax[0, 2]):
                bp = pmax[:, 1]
                rp = pmax[:, 2]
            elif (lc > pmax[0, 3]) or (lc < pmax[0, 0]):
                # for now, just take inner two
                bp = pmax[:, 1]
                rp = pmax[:, 2]
            elif lc < pmax[0, 1]:  # proceed like for 3 maxima
                bp = pmax[:, 0]
                rp = pmax[:, np.argmax(pmax[1, 1:]) + 1]
            elif lc < pmax[0, 3]:
                bp = pmax[:, np.argmax(pmax[1, :3])]
                rp = pmax[:, 3]

    if graph:
        #plt.axvline(x=guess, color='r', ls='--')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))

    # fast option just returns the peak values
    if fast:
        return bp, rp

    # interpolation for more precise peaks
    if bp[0]:
        idg = np.argmin(np.abs(vel - bp[0]))
        llim = max(0, idg - 2)
        hlim = min(idg + 2, spec.shape[0] - 1)
        nvel = np.linspace(vel[llim], vel[hlim], 45)
        llim = max(0, idg -3)
        hlim = min(idg + 4, spec.shape[0])
        spl = interpolate.splrep(vel[llim:hlim], spec[llim:hlim], k=3, s=0)
        nspec = interpolate.splev(nvel, spl ,der=0)
        midx = np.argmax(nspec)
        bp[0] = nvel[midx]
        bp[1] = nspec[midx]
    if rp[0]:
        idg = np.argmin(np.abs(vel - rp[0]))
        llim = max(0, idg - 2)
        hlim = min(idg + 2, spec.shape[0] - 1)
        nvel = np.linspace(vel[llim], vel[hlim], 45)
        llim = max(0, idg -3)
        hlim = min(idg + 4, spec.shape[0])
        spl = interpolate.splrep(vel[llim:hlim], spec[llim:hlim], k=3, s=0)
        nspec = interpolate.splev(nvel, spl ,der=0)
        midx = np.argmax(nspec)
        rp[0] = nvel[midx]
        rp[1] = nspec[midx]

    if graph:
        plt.axvline(x=lc, color='g')
        plt.axvline(x=bp[0], color='y')
        plt.axvline(x=rp[0], color='y')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))
        plt.plot([bp[0]],[bp[1]], 'b^')
        plt.plot([rp[0]],[rp[1]], 'r^')
        #raw_input()

    return bp, rp


def mg_single_params(vel, spec, guess, lookahead=2, delta=0, pts_min=5,
                     ppts=2, graph=False, loc=(0,0), force_guess=False):
    """
    Calculates the properties for a single Mg II line. At the moment the
    properties detected are:

    * Central minimum intensity and wavelength
    * Intensity and wavelength for two peaks

    """
    from scipy import ndimage, interpolate
    import matplotlib.pyplot as plt

    bp = np.ma.masked_all(2)
    lc = np.ma.masked_all(2)
    rp = np.ma.masked_all(2)

    if hasattr(spec[0], 'mask') or np.isnan(guess):
        return bp, lc, rp

    if graph:
        plt.clf()
        plt.plot(vel, spec)

    lc_guess = guess

    pmax, pmin = peakdetect2(spec, vel, delta_max=5e-12, delta_min=5e-12,
                         lookahead_max=lookahead,
                         lookahead_min=lookahead)
    #pmax, pmin = peakdetect_lcl(spec, vel, lookahead=2)

    # Use peak detect unless forcing the use of supplied guess
    if not force_guess:
        if np.any(pmax) and np.any(pmin):
            lpmax = len(pmax[0])
            lpmin = len(pmin[0])

            if graph:
                plot(pmax[0], pmax[1], 'ro')
                plot(pmin[0], pmin[1], 'go')

            # SELECTION OF LINE CORE
            if (lpmin, lpmax) in [(1,2), (3, 1), (3, 2), (3, 4), (5, 4), (7,6)]: # added (3, 1)
                # most straightforward case: take the middle one for line core
                lc_guess = pmin[0][lpmin//2]
                pts_min = pts_min//2
            elif (lpmin, lpmax) in [(2, 2), (2, 3), (3, 3), (4, 2), (4, 3), (4, 4)]:
                pts_min = pts_min//2
                # locations of two largest maxima
                lmax = np.sort(pmax[0][np.argsort(pmax[1])[-2:]])
                # minima inside the above window
                idx = (pmin[0] > lmax[0]) & (pmin[0] < lmax[1])
                if np.any(idx):
                    lc_guess = pmin[0, idx][np.argmin(pmin[1, idx])]
                    imin = pmin[1, idx][np.argmin(pmin[1, idx])]
            elif lpmin == 2 and lpmax == 1: # was [1, 3]
                if np.max(pmin[1])/np.min(pmin[1]) > 1.3:
                    # use lowest minimum
                    lc_guess = pmin[0, np.argmin(pmin[1])]
                    pts_min = pts_min//2
                else:
                    # Take the minimum furthest away from the highest peak
                    lc_guess = pmin[0, np.argmax(np.abs(pmin[0] - pmax[0, np.argmax(pmax[1])]))]
                    pts_min = pts_min//2
            elif lpmin == 1: # and np.abs(pmin[0,0]) < 20.:
                lc_guess = pmin[0,0]
                pts_min = pts_min//2
            #elif lpmax == 1 and lpmin == 2:
            #    # take centroid in between lows
            #    idx = (wave >= pmin[0,0]) & (wave <= pmin[0,1])
            #    guess = np.trapz(wave[idx]*spec[idx], x=wave[idx])/np.trapz(spec[idx], x=wave[idx])

    # SELECTION OF TWO PEAKS
    if np.any(pmax):
        lpmax = len(pmax[0])
        if lpmax > 4:   # too many maxima, take inner 4
            pmax = pmax[:, lpmax // 2 - 2 : lpmax // 2 + 2]
            lpmax = 4

        if lpmax == 1:
            if pmax[0, 0] > lc_guess:
                rp = pmax[:, 0]
            else:   # by default assign single max to blue peak if no core
                bp = pmax[:, 0]
        elif lpmax == 2:
            bp = pmax[:, 0]
            rp = pmax[:, 1]
        elif lpmax == 3:   # this kind of behaviour can also happen with 4 peaks!
            # take the one close to the line core, and from the
            # remaining two chose the strongest
            if pmax[0, 0] < lc_guess < pmax[0, 1]:
                bp = pmax[:, 0]
                rp = pmax[:, np.argmax(pmax[1, 1:]) + 1]
            elif pmax[0, 1] < lc_guess < pmax[0, 2]:
                bp = pmax[:, np.argmax(pmax[1, :-1])]
                rp = pmax[:, -1]
            else:
                # if line centre is not between maxima, take two strongest
                #aa = pmax[0][pmax[1] != np.min(pmax[1])]
                #bp_guess = aa[0]
                #rp_guess = aa[1]
                aa = pmax[:, pmax[1] != np.min(pmax[1])]
                bp = aa[:, 0]
                rp = aa[:, 1]
        elif lpmax == 4:
            # if line core in the middle of the four, or outside the
            # maxima region, take inner two maxima
            if (lc_guess > pmax[0, 1]) and (lc_guess < pmax[0, 2]):
                bp = pmax[:, 1]
                rp = pmax[:, 2]
            elif (lc_guess > pmax[0, -1]) or (lc_guess < pmax[0, 0]):
                # for now, just take inner two
                bp = pmax[:, 1]
                rp = pmax[:, 2]
            elif lc_guess < pmax[0, 1]:  # proceed like for 3 maxima
                bp = pmax[:, 0]
                rp = pmax[:, np.argmax(pmax[1, 1:]) + 1]
            elif lc_guess < pmax[0, -1]:
                bp = pmax[:, np.argmax(pmax[1, :-1])]
                rp = pmax[:, -1]


    if graph:
        #axvline(x=guess, color='r', ls='--')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))

    # no parabolic fit
    #return guess, imin

    bp_guess = bp[0]
    rp_guess = rp[0]

    # Approximate index of guess and of spectral minimum around it
    try:
        idg = np.argmin(np.abs(vel - lc_guess))
        ini = np.argmin(spec[idg - pts_min:idg + pts_min]) + idg - pts_min
    except ValueError:
        print(idg, pts_min)
        print(pmax, pmin)
        print(guess, loc)
        print(spec[idg - pts_min:idg + pts_min])
        return bp, lc, rp

    # if no points, return masked lc
    if len(vel[ini - ppts:ini + ppts + 1]) == 0:
        return bp, lc, rp

    # Fit parabola
    fit = np.polyfit(vel[ini - ppts:ini + ppts + 1],
                    spec[ini - ppts:ini + ppts + 1], 2)
    # Convert poly to parabola coefficients
    lc[0] = -fit[1] / (2 * fit[0])
    lc[1] = fit[2] - fit[0] * lc[0]**2

    # interpolate for red peak
    if rp[0]:
        idg = np.argmin(np.abs(vel - rp[0]))
        nvel = np.linspace(vel[idg - 1], vel[idg + 1], 15)
        llim = max(0, idg -3)
        hlim = min(idg + 4, spec.shape[0])
        spl = interpolate.splrep(vel[llim:hlim], spec[llim:hlim], k=3, s=0)
        nspec = interpolate.splev(nvel, spl ,der=0)
        midx = np.argmax(nspec)
        rp[0] = nvel[midx]
        rp[1] = nspec[midx]



    # If fitted minimum is furthen than 4 wavelenght points, redo the fit
    # using a bigger area (add 2 points to ppts) and using always quad_lsq.
    # This is a last resort measure, and line minimum intensity will be
    # affected, especially in weak lines
    if np.abs(lc[0] - vel[ini]) > 4 * (vel[1] - vel[0]):
        lc[:] = np.ma.masked

    if graph:
        plt.axvline(x=lc[0], color='g')
        plt.axvline(x=bp[0], color='y')
        plt.axvline(x=rp[0], color='y')
        plt.axvline(x=lc_guess, color='r', ls='--')
        plt.axvline(x=bp_guess, color='b', ls=':')
        plt.axvline(x=rp_guess, color='r', ls=':')
        plt.title('(x,y) = (%i, %i)'  % (loc[0], loc[1]))
        raw_input()

    return bp, lc, rp


def mg_fix_core(vel, spec, bp, rp):
    """
    Sets the line core as the minimum between the two peaks.
    """
    from scipy import interpolate

    lc = np.ma.masked_all(2, dtype='d')
    if np.ma.is_masked(bp) or np.ma.is_masked(rp):
        return lc
    wi = np.argmin(np.abs(vel - bp))
    wf = np.argmin(np.abs(vel - rp))
    assert wi < wf
    smin = wi + np.argmin(spec[wi:wf])
    # interpolate
    try:
        nvel = np.linspace(vel[smin - 2], vel[smin + 2], 15)
    except IndexError:
        print(smin)
        raise ValueError
    llim = max(0, smin - 3)
    hlim = min(smin + 4, spec.shape[0])
    spl = interpolate.splrep(vel[llim:hlim], spec[llim:hlim], k=3, s=0)
    nspec = interpolate.splev(nvel, spl ,der=0)
    midx = np.argmin(nspec)
    lc[0] = nvel[midx]
    lc[1] = nspec[midx]
    return lc


def deriv_lc(vel, spec, margin, delta=0, lookahead=4, graph=False):
    pmax, pmin = peakdetect2(spec, vel, delta_max=delta, delta_min=delta,
                             lookahead_max=lookahead,
                             lookahead_min=lookahead)
    import matplotlib.pyplot as plt
    if graph:
        plt.clf()
        plt.plot(vel, spec)


    if np.any(pmax):
        #  remove spurious first maximum (happens with interpolation)
        pmax = pmax[:, (pmax[0] > vel[0] + 20) & (pmax[0] < vel[-1] -20.)]

        lpmax = len(pmax[0])
        lpmin = len(pmin[0])

        if lpmax > 1 or lpmax == 0:
            return np.ma.masked, np.ma.masked

        if graph:
            plt.plot(pmax[0], pmax[1], 'ro')
            plt.plot(pmin[0], pmin[1], 'go')

        dd = np.abs(np.diff(spec))
        # define the boundaries to inspect for peak asymmetry and derivative
        incr = int(15./(vel[1]-vel[0]))
        idxm = np.argmin(np.abs(vel - pmax[0, 0]))
        vv = [idxm - incr, idxm + incr]
        vv2 = [idxm - incr * 1.3, idxm + incr * 1.3]


        if graph:
            plot(vel[vv], spec[vv], 'k+', ms=8)
            if spec[vv[0]] > spec[vv[1]]:
                axvline(x=vel[vv[0]], color='k', ls='--')
            else:
                axvline(x=vel[vv[1]], color='k', ls='--')


        if spec[vv[0]] > spec[vv[1]]:   # peak on left side
            der = dd[vv2[0] + margin:idxm - margin]
            pidx = vv2[0] + np.argmin(der) + margin + 1
        else:                           # peak on right side
            der = dd[idxm + margin:vv2[1] - margin]
            pidx = idxm + np.argmin(der) + margin + 1


        if graph:
            plt.axvline(x=vel[pidx], color='g', ls='-')
            plt.plot([vel[pidx]], [spec[pidx]], 'bo')
            raw_input()

        return vel[pidx], spec[pidx]


def deriv_repair(vel, spec_in, lc, li, graph=False):
    ''' Estimates the line centre for Mg II by taking the extreme of the second
    derivative when only one maximum exists'''

    import matplotlib.pyplot as plt

    idx = np.isnan(lc)
    spec = spec_in[idx].copy()

    nspec = spec.shape[0]

    lc_fix = np.ma.masked_all(nspec)
    li_fix = np.ma.masked_all(nspec)

    for i in range(nspec):
        pmax, lixo = peakdetect(spec[i], vel, lookahead=1, delta=0)
        lixo, pmin = peakdetect(spec[i], vel, lookahead=2, delta=5e-12)

        if np.any(pmax) and np.any(pmin):
            ## manually remove the extra line
            #pmax = pmax[:, pmax[0] > -35]
            #pmin = pmin[:, pmin[0] > -35]
            lpmax = len(pmax[0])
            lpmin = len(pmin[0])

            if lpmax == 1 and lpmin in [2, 3, 4]:
                # discard anything but the two minima closest to the maximum
                if lpmin > 2:
                    pmin = pmin[np.argsort(pmin[0]-pmax[0,0])[:2]]

                # find minimum furthest from maximum: will apply second derivative here
                minloc = pmin[0, np.argmax(np.abs(pmin[0] - pmax[0,0]))]
                maxloc = pmax[0,0]

                if maxloc > minloc:
                    wi = np.where(vel == minloc)[0]
                    wf = np.where(vel == maxloc)[0]
                else:
                    wi = np.where(vel == maxloc)[0]
                    wf = np.where(vel == minloc)[0]

                # get second derivative, use extreme to estimate line centre
                loc = np.argmax(np.diff(np.diff(spec[i, wi:wf])))
                lc_fix[i] = vel[wi:wf][loc + 1]
                li_fix[i] = spec[i, wi:wf][loc + 1]

            if graph:
                plt.clf()
                plt.plot(vel, spec[i], 'b-')
                plt.plot(pmax[0], pmax[1], 'ro')
                plt.plot(pmin[0], pmin[1], 'go')
                plt.axvline(x=lc_fix[i], color='r', ls='--')
                plt.plot(vel[wi:wf][1:], np.diff(spec[i, wi:wf])*10, 'g-')
                plt.plot(vel[wi:wf][1:-1], np.diff(np.diff(spec[i, wi:wf]))*30, 'r-')
                plt.axhline(y=0, color='k', ls='--')
                raw_input()

    lc[idx] = lc_fix
    li[idx] = li_fix

    return

