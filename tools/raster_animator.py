import matplotlib.colors as colors
import matplotlib.pyplot as plt
from irispy.sji import SJIMap, SJICube

from sunpy.image.coalignment import calculate_shift
from astropy.coordinates import SkyCoord as SC

#from irispy.spectrograph import IRISSpectrograph

import os
import pickle

import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import sunpy.cm as cm
import sunpy.instr.iris
import sunpy.map
import numpy as np
import matplotlib.colors as colors
import astropy.units as u
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import visualization
import sunpy.physics.differential_rotation as diff_rot

from astropy.io import fits
plt.rcParams['animation.ffmpeg_path'] = '/Users/shelbe/anaconda/bin/ffmpeg'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams.update({'font.size': 6})

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='SunPy'), bitrate=100000)

iris_dir = '/kale/iris/data/level2/'
obs = 3640106077
date = datetime.datetime(2017,8,8)

def search(date, days, obsid, iris_dir = '/net/md5/Volumes/kale/iris/data/level2/', usetree=True):
    if (isinstance(obsid, str) or isinstance(input, int)):
        obsid = [obsid]
    results=[]
    if usetree:
        for d in range(0, days):
            print(date)
            for root, dir, file in os.walk(iris_dir + date.strftime('%Y/%m/%d')):
                for obs in obsid:
                    if (str(obs) in dir):
                        for i in file:
                            if ('raster' in i):
                                results.append(os.path.join(root, i))
            date += datetime.timedelta(days=1)
    else:
        for file in os.listdir(iris_dir):
            if file.endswith(".fits"):
                results.append(os.path.join(iris_dir, file))

    results.sort()
    #Todo: view results
    print('Contains ', len(results), ' files')
    #Todo: list adapter to add/remove
    f = open('obsid_'+str(obsid[0])+'.pckl', 'wb')
    pickle.dump(results, f)
    f.close()


def raster(file_list, outfile, wave_ref = [2795.80, 2798.65, 2796.18, 2796.36, 2796.49], rotate = True, coalign = False):

    for n,i in enumerate(file_list):

        print('--------------------------------------------------\nGetting FITS data from ' + i)
        main_header = fits.getheader(i, ext=0)
        exptime = main_header.get('exptime')
        Mg_header = None
        wing_header = None
        for h in range(1,9):
            tdesc='TDESC' + str(h)
            twave = 'TWAVE' + str(h)
            print(main_header.get(tdesc))
            if main_header.get(tdesc) != None:
                if '2832' in main_header.get(tdesc):
                    #get header and transpose data array
                    wing_data = fits.getdata(i, ext=h)
                    wing_header = fits.getheader(i, ext=h)
                    wing_header['naxis1']=wing_header['naxis3']
                    wing_header['crpix1'] = wing_header['crpix3']
                    wing_header['crval1'] = wing_header['crval3']
                    wing_header['cdelt1'] = wing_header['cdelt3']
                    wing_header['ctype1'] = wing_header['ctype3']
                    wing_header['cunit1'] = wing_header['cunit3']
                    pos=15
                    wavelength = round(2832.0, 1)
                if '2814' in main_header.get(tdesc):
                    wing_data = fits.getdata(i, ext=h)
                    wing_header = fits.getheader(i, ext=h)
                    wing_header['naxis1']=wing_header['naxis3']
                    wing_header['crpix1'] = wing_header['crpix3']
                    wing_header['crval1'] = wing_header['crval3']
                    wing_header['cdelt1'] = wing_header['cdelt3']
                    wing_header['ctype1'] = wing_header['ctype3']
                    wing_header['cunit1'] = wing_header['cunit3']
                    pos=59
                    wavelength = round(2815.8, 1)
                else:
                    print('No photospheric data available')

                if 'Mg' in main_header.get(tdesc):
                    Mg_data = fits.getdata(i, ext=h)
                    Mg_header = fits.getheader(i, ext=h)
                print(twave, int(main_header.get(twave)))



        if Mg_header == None:
            print('Header not found')
            return
        #plt.imshow(wing_data[0])
        #plt.show()
        naxis1 = Mg_header.get('NAXIS1')
        naxis2 = Mg_header.get('NAXIS2')
        naxis3 = Mg_header.get('NAXIS3')
        crpix1 = Mg_header.get('CRPIX1')
        crpix2 = int(Mg_header.get('CRPIX2'))
        crpix3 = int(Mg_header.get('CRPIX3'))
        crval1 = Mg_header.get('CRVAL1')
        crval2 = Mg_header.get('CRVAL2')
        crval3 = Mg_header.get('CRVAL3')
        cdelt1 = Mg_header.get('CDELT1')
        cdelt2 = Mg_header.get('CDELT2')
        cdelt3 = Mg_header.get('CDELT3')
        ctype3 = Mg_header.get('CTYPE3')
        cunit3 = Mg_header.get('CUNIT3')



        Mg_header['naxis1'] = naxis3
        Mg_header['crpix1'] = crpix3
        Mg_header['crval1'] = crval3
        Mg_header['cdelt1'] = cdelt3
        Mg_header['ctype1'] = ctype3
        Mg_header['cunit1'] = cunit3
        Mg_header['naxis'] = 2
        Mg_header.remove('crpix3')
        Mg_header.remove('crval3')
        Mg_header.remove('cdelt3')
        Mg_header.remove('ctype3')
        Mg_header.remove('cunit3')
        Mg_header.set('wavelnth', twave)
        Mg_header.set('dsun',main_header.get('DSUN_OBS'))

        Mg_header.set('instrume', main_header.get('instrume'))
        Mg_header.set('waveunit',"Angstrom")
        print(main_header)
        print(Mg_header)
        #2796.4 A & 2803.5 A, 124
        st = datetime.datetime.strptime(main_header.get('STARTOBS'), '%Y-%m-%d' + 'T' + '%H:%M:%S.%f')
        et = datetime.datetime.strptime(main_header.get('ENDOBS'), '%Y-%m-%d' + 'T' + '%H:%M:%S.%f')
        time_range = sunpy.time.TimeRange(st,et)


        dt = main_header.get('RASNRPT')
        nt = main_header.get('RASRPT')

        splits = time_range.split(dt)
        date = splits[nt-1].center

        timestamp = date.strftime('%Y-%m-%d %H:%M:%S.%f')
        print(str(nt)+' of '+str(dt))


        xcen = (crval3 ) * u.arcsec
        ycen = (crval2 ) * u.arcsec
        print(crval3, crval2)




        wavmin = crval1 - crpix1 * cdelt1
        wavmax = crval1 + (naxis1 - crpix1) * cdelt1
        title = []

        if wing_header == None:
            title.append('No photospheric data available')
        else:
            title.append('$' + str(wavelength) + '\AA$')
            wing_image = wing_data.T[pos] / exptime
            wing = SJIMap(wing_image, wing_header)
            wing.meta['wavelnth'] = wavelength
            wing.meta['date'] = date.isoformat()

        wave_idx = []
        wavelength=np.arange(0,naxis1)*(cdelt1) + wavmin


        print(Mg_header)

        for lam in wave_ref:
            wave = np.where(wavelength >= lam)[0][0]

            wave_idx.append(wave)
            title.append(str('$'+ str(round(wavelength[wave],1)) + '\AA$'))

        mg_wing_image = (Mg_data.T[wave_idx2[0]]/exptime)
        mg_triplet_image = (Mg_data.T[wave_idx2[1]]/exptime)
        mg_k2v_image = (Mg_data.T[wave_idx2[2]]/exptime)
        mg_k3_image = (Mg_data.T[wave_idx2[3]]/exptime)
        mg_k2r_image = (Mg_data.T[wave_idx2[4]]/exptime)


        mg_wing =SJIMap(mg_wing_image, Mg_header)
        mg_wing.meta['wavelnth'] = wave_idx2[0]
        mg_wing.meta['date'] = date.isoformat()

        mg_triplet = SJIMap(mg_triplet_image, Mg_header)
        mg_triplet.meta['wavelnth'] = wave_idx2[1]
        mg_triplet.meta['date'] = date.isoformat()

        mg_k2v = SJIMap(mg_k2v_image, Mg_header)
        mg_k2v.meta['wavelnth'] = wave_idx2[2]
        mg_k2v.meta['date'] = date.isoformat()

        mg_k3 = SJIMap(mg_k3_image, Mg_header)
        mg_k3.meta['wavelnth'] = wave_idx2[3]
        mg_k3.meta['date'] = date.isoformat()

        mg_k2r = SJIMap(mg_k2r_image, Mg_header)
        mg_k2r.meta['wavelnth'] = wave_idx2[4]


        print(mg_k3.meta)
        if n == 0:
            # Set Starting Coordinates (arcsec)

            layer = mg_k3.data[crpix2 - 250:crpix2 + 250, crpix3 - 80:crpix3 + 80]

            xshift = 0.*u.arcsec
            yshift = 0.*u.arcsec
            xcen = (crval3 - xshift.value) * u.arcsec
            ycen = (crval2 - yshift.value) * u.arcsec
            cen0 = SC(xcen, ycen, frame='helioprojective', obstime=date)

            xlength = 150 # Cropping dimensions
            ylength = 150

            x0 = (cen0.data._lon.arcsec - .5 * xlength)
            y0 = (cen0.data._lat.arcsec - .5 * ylength)
            bl = SC(x0 * u.arcsec, y0 * u.arcsec, frame='helioprojective',
                    obstime=st)  # Bottom Left and Top Right coordinates
            tr = SC((x0 + xlength) * u.arcsec, (y0 + ylength) * u.arcsec, frame='helioprojective', obstime=st)


            #gamma setup widget
            images = [wing,mg_wing,mg_triplet,mg_k2v,mg_k3, mg_k2r]
            gammas = [.5, .5, .5, .5, .5, .5]
            mins = [0., 0., 0, 0., 0., 0.]
            maxs = [np.percentile(wing.data, 98), np.percentile(mg_wing.data, 98), np.percentile(mg_triplet.data, 98),
                    np.percentile(mg_k2v.data, 98), np.percentile(mg_k3.data, 98), np.percentile(mg_k2r.data, 98)]
            cmaps = [cm.get_cmap(name='irissji2832'), cm.get_cmap(name='sohoeit171'), cm.get_cmap(name='irissji1330'),
                     cm.get_cmap(name='irissji1400'), cm.get_cmap(name='irissji1600'), cm.get_cmap(name='irissji2796')]
            print('Gamma:', gammas, '\n Min:', mins, '\nMax:', maxs)

            for j,i in enumerate(images):
                norm = colors.PowerNorm(gammas[j], mins[j], maxs[j])
                fig_gamma = plt.figure()
                ax_gamma = fig_gamma.add_axes([.1, .25, .8, .75])

                gamma_slider_ax = fig_gamma.add_axes([0.25, 0.15, 0.65, 0.03], axisbg='gray')
                gamma_slider = Slider(gamma_slider_ax, 'gamma', 0.1, 2.0, valinit=gammas[j])

                max_slider_ax = fig_gamma.add_axes([0.25, 0.1, 0.65, 0.03], axisbg='gray')
                max_slider = Slider(max_slider_ax, 'max value', i.mean(), i.max(), valinit=maxs[j])

                min_slider_ax = fig_gamma.add_axes([0.25, 0.05, 0.65, 0.03], axisbg='gray')
                min_slider = Slider(min_slider_ax, 'min value', 0.0, i.mean(), valinit=mins[j])

                def gamma_on_changed(val):
                    norm = colors.PowerNorm(val, mins[j], maxs[j])
                    i.plot(norm=norm, cmap=cmaps[j])
                    gammas[j] = gamma_slider.val

                def max_on_changed(val):
                    norm = colors.PowerNorm(gammas[j], mins[j], val)
                    i.plot(norm=norm, cmap=cmaps[j])
                    maxs[j] = max_slider.val

                def min_on_changed(val):
                    norm = colors.PowerNorm(gammas[j], val, maxs[j])
                    i.plot(norm=norm, cmap=cmaps[j])
                    mins[j] = min_slider.val

                gamma_slider.on_changed(gamma_on_changed)
                max_slider.on_changed(max_on_changed)
                min_slider.on_changed(min_on_changed)

                i.plot(axes=ax_gamma, cmap=cmaps[j],norm=norm)

                plt.show()

            print('Gamma:', gammas, '\n Min:', mins, '\nMax:', maxs)


            fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
            fig.tight_layout()
            plt.subplots_adjust(bottom=0.07, left=0.07, top=.98, right=.98, wspace=0.0, hspace=0.0)
            writer.setup(fig, outfile, dpi=120)

        elif n > 0:
            # Get new start time
            st = datetime.datetime.strptime(main_header.get('STARTOBS'), '%Y-%m-%d' + 'T' + '%H:%M:%S.%f')

            # Rotate Helio Projective Coordinates from end-start time (time elapsed between obs)
            if rotate and (x0+xlength < 1000):
                cen = diff_rot.solar_rotate_coordinate(cen0, date, frame_time='sidereal', rot_type='snodgrass')
            if n >= 273: #only for Chmon
                cen = SC(cen.data._lon.arcsec* u.arcsec, (crval2-50)* u.arcsec, frame='helioprojective', obstime=date)


            if coalign:
                ycrop = 225
                xcrop = 45
                yoffset = 25
                xoffset = 35
                template = mg_k3.data[crpix2 - ycrop:crpix2 + ycrop, crpix3 - xcrop:crpix3 + xcrop]
                shift = calculate_shift(layer, template)

            # if xyshift hits shift boundaries, resize template, rerun calculate shift
                while (shift[0].value >= 2. * yoffset) or (shift[1].value >= 2. * xoffset):
                    if shift[0].value == 2. * yoffset:
                        ycrop -= 10
                        yoffset += 10
                    if shift[1].value == 2. * xoffset:
                        xcrop -= 10
                        xoffset += 10
                        template = mg_k3.data[crpix2 - ycrop:crpix2 + ycrop, crpix3 - xcrop:crpix3 + xcrop]
                        shift = calculate_shift(layer, template)

                print(shift)
                xshift += (shift[1] - xoffset * u.pix) * mg_k3.scale[0]
                yshift += (shift[0] - yoffset * u.pix) * mg_k3.scale[1]
                print(i, ' Xcorr: ', (xshift.round(0), yshift.round(0)))
                layer = mg_k3.data[crpix2 - 250:crpix2 + 250, crpix3 - 80:crpix3 + 80]
                print('New layer created')


            if rotate and not coalign:
                print('Rotate coordinates, no image coalignment')
                ycen = cen.data._lat
                xcen = cen.data._lon
            elif coalign and not rotate:
                print('coalign image, no coordinate rotation')
                ycen = (crval2 - yshift.value) * u.arcsec
                xcen = (crval3 - xshift.value) * u.arcsec
            elif rotate and coalign:
                print('Rotate coordinates and coalign image')
                ycen = cen.data._lat
                xcen = (crval3 - xshift.value) * u.arcsec
            else:
                print('No shift')
                ycen = crval2 * u.arcsec
                xcen = crval3 * u.arcsec
        print('Center: ',xcen, ycen)
        cen = SC(xcen, ycen, frame='helioprojective', obstime=date)

        # New Plot Boundaries
        x0 = (cen.data._lon.arcsec - .5 * xlength)
        y0 = (cen.data._lat.arcsec - .5 * ylength)
        print('Bottom Left: ', x0, y0)
        # Bottom Left and Top Right coordinates - used for creating submaps, not currently used
        bl = SC(x0 * u.arcsec, y0 * u.arcsec, frame='helioprojective', obstime=date)
        tr = SC((x0 + xlength) * u.arcsec, (y0 + ylength) * u.arcsec, frame='helioprojective', obstime=date)
        if wing_header != None:
            wing.plot(axes=ax[0, 0], cmap=cmaps[0], norm=colors.PowerNorm(gammas[0], mins[0], maxs[0]))
        mg_wing.plot(axes=ax[0, 1], cmap=cmaps[1],norm=colors.PowerNorm(gammas[1], mins[1], maxs[1]))
        mg_triplet.plot(axes=ax[0, 2], cmap=cmaps[2], norm=colors.PowerNorm(gammas[2], mins[2], maxs[2]))
        mg_k2v.plot(axes=ax[1, 0], cmap=cmaps[3],norm=colors.PowerNorm(gammas[3], mins[3], maxs[3]))
        mg_k3.plot(axes=ax[1, 1], cmap=cmaps[4], norm=colors.PowerNorm(gammas[4], mins[4], maxs[4]))
        mg_k2r.plot(axes=ax[1, 2], cmap=cmaps[5],norm=colors.PowerNorm(gammas[5], mins[5], maxs[5]))

        ax[0, 0].set_title(title[0], visible=False)
        ax[0, 1].set_title(title[1], visible=False)
        ax[0, 2].set_title(title[2], visible=False)
        ax[1, 0].set_title(title[3], visible=False)
        ax[1, 1].set_title(title[4], visible=False)
        ax[1, 2].set_title(title[5], visible=False)
        ax[0, 0].set_xlabel('X [Arcsec]', visible=False)
        ax[0, 0].set_ylabel('Y [Arcsec]')
        ax[0, 0].set_xlim(x0, x0 + xlength)
        ax[0, 0].set_ylim(y0, y0 + ylength)

        plt.setp(ax[0, 1].get_yticklabels(), visible=False)
        plt.setp(ax[0, 2].get_yticklabels(), visible=False)
        plt.setp(ax[1, 1].get_yticklabels(), visible=False)
        plt.setp(ax[1, 2].get_yticklabels(), visible=False)

        ax[0, 1].set_xlabel('X [Arcsec]', visible=False)
        ax[0, 1].set_ylabel('Y [Arcsec]', visible=False)
        #
        ax[0, 2].set_xlabel('X [Arcsec]', visible=False)
        ax[0, 2].set_ylabel('Y [Arcsec]', visible=False)

        ax[1, 0].set_xlabel('X [Arcsec]', visible=False)
        ax[1, 0].set_ylabel('Y [Arcsec]')

        ax[1, 1].set_xlabel('X [Arcsec]')
        ax[1, 1].set_ylabel('Y [Arcsec]', visible=False)

        ax[1, 2].set_xlabel('X [Arcsec]', visible=False)
        ax[1, 2].set_ylabel('Y [Arcsec]', visible=False)

        ax[1, 0].annotate('IRIS' + ' ' + timestamp,
                          xy=(x0+10, y0+2), color='white', fontsize=6, zorder=1)
        ax[0, 0].annotate(title[0], xy=(x0+10, y0 + 4), color='white', fontsize=6, zorder=1)
        ax[0, 1].annotate('Mg II Wing '+ title[1], xy=(x0+10, y0 + 4), color='white', fontsize=6, zorder=1)
        ax[0, 2].annotate('Mg II Triplet ', xy=(x0+10, y0 + 4), color='white', fontsize=6, zorder=1)
        ax[1, 0].annotate('Mg II k$_{2v}$ ', xy=(x0+10, y0 + 10), color='white', fontsize=6, zorder=1)
        ax[1, 1].annotate('Mg II k$_{3}$ ', xy=(x0+10, y0 + 10), color='white', fontsize=6, zorder=1)
        ax[1, 2].annotate('Mg II k$_{2r}$ ', xy=(x0+10, y0 + 10), color='white', fontsize=6, zorder=1)



        writer.grab_frame()

        ax[0, 0].cla()
        ax[0, 1].cla()
        ax[0, 2].cla()
        ax[1, 0].cla()
        ax[1, 1].cla()
        ax[1, 2].cla()






obs = [3640106077]
#obs=[3600106076,3620106076,3620108076]
date = datetime.datetime(2017,8,8)

#search(date, 8, obs)
f = open('obsid_'+str(obs[0])+'.pckl', 'rb')
raster_list = pickle.load(f)


import warnings

def fxn():
    warnings.warn("Missing metadata", Warning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    raster(raster_list,'/saturn/sanhome/shelbe/IRIS/movies/CHmon_raster.mp4')
