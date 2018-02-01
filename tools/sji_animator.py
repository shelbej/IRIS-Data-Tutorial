
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import vso_query
from irispy.sji import SJICube, SJIMap, BAD_PIXEL_VALUE_UNSCALED
import sunpy.map
from astropy.coordinates import SkyCoord as SC
from sunpy.image.coalignment import calculate_shift
from scipy.ndimage import binary_dilation, generate_binary_structure

import matplotlib.colors as colors
import astropy.units as u
import time
import numpy as np
import numpy.ma as ma
import sunpy.physics.differential_rotation as diff_rot
### sji_animator is a procedural script that combines IRIS SJI and AIA data sets
### Created by Shelbe Timothy, 2016
rotate = True
coalign = True
#pyplot setup
plt.style.use('dark_background')
plt.rcParams['animation.ffmpeg_path'] = '/Users/shelbe/anaconda/bin/ffmpeg'
writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='SunPy'), bitrate=100000)
plt.rcParams.update({'font.size': 7})


#local directories
drive='/Volumes/HDrive/Users/Shelbe/IRIS/AR12641/SJI/'
iris_dir = '/net/md5/Volumes/kale/iris/data/level2/'
aia_dir = drive + 'AIA/'
outputfile = '/Users/shelbe/Documents/IRIS/writer_test2.mp4'

#grab all .fits in directory
import os
sji_list=[]
for file in os.listdir(drive):
    if file.endswith(".fits"):
        sji_list.append(os.path.join(drive, file))
print(len(sji_list))
aia_list=[]
for file in os.listdir(aia_dir):
    if file.endswith(".fits"):
        aia_list.append(os.path.join(aia_dir, file))
print(len(aia_list))
dimensions = u.Quantity([250,250],u.pixel) #resample size if using full disk aia image
aia = sunpy.map.Map(aia_list)              #Initialize aia maps
for n,map in enumerate(aia):           #Resample to 250 x 250 pix
  aia[n]=map.resample(dimensions)
print('aia list resampled')

"""sji_list=[]
date = datetime.datetime(2017,8,8)
for d in range(0,8):
    print(date)
    for root,dir,file in os.walk(iris_dir+date.strftime('%Y/%m/%d')):
        if (str(obs) in dir):
        #print(dir)
            for i in file:
                if ('SJI_1330' in i):
                    sji_list.append(root+'/'+i)

    date += datetime.timedelta(days=1)
"""

#Build plots
def sji_animator(sji_list):
    # Setup plot
    fig = plt.figure()
    ax0 = fig.add_axes([0.05, 0.08, .99, .9], zorder=0)
    ax0.get_xaxis().set_tick_params(direction='out', width=1)
    ax0.get_yaxis().set_tick_params(direction='out', width=1)
    #ax1 = fig.add_axes([.775, .779, .2, .2], zorder=1)

    writer.setup(fig, outputfile, dpi=100)
    start = time.perf_counter()

    n=len(sji_list)
    for i, file in enumerate(sji_list):

        mc = SJICube(file)
        dustbuster(mc)

        print('Getting FITS data from ' + file)

        crval1 = mc[0].meta.get('CRVAL1')
        crval2 = mc[0].meta.get('CRVAL2')
        crpix1 = int(mc[0].meta.get('CRPIX1'))
        crpix2 = int(mc[0].meta.get('CRPIX2'))
        fovx = mc[0].meta.get('FOVX')
        fovy = mc[0].meta.get('FOVY')

        wave = int(mc[0].meta.get('TWAVE1'))
        title = mc[0].meta.get('TELESCOP') + ' ' + mc[0].meta.get('INSTRUME') + ' $' + str(wave) + r'\AA$'

        nx = int(mc[0].meta.get('NRASTERP'))  # number of raster positions

        if i == 0:
            # Set Starting Coordinates (arcsec)
            if coalign == True:
                layer = mc.percentile(97).data[crpix2 - 250:crpix2 + 250, crpix1 - 80:crpix1 + 80]

            st = datetime.datetime.strptime(mc[0].meta['DATE-OBS'], '%Y-%m-%d' + ' ' + '%H:%M:%S.%f')
            et = datetime.datetime.strptime(mc[-1].meta['DATE-OBS'], '%Y-%m-%d' + ' ' + '%H:%M:%S.%f')
            xshift = -0.*u.arcsec
            yshift = 0.*u.arcsec
            xcen = (crval1 - xshift.value) * u.arcsec
            ycen = (crval2 - yshift.value) * u.arcsec
            cen = SC(xcen, ycen, frame='helioprojective', obstime=st)

            xlength = .45 * fovx  # Cropping dimensions
            ylength = .58 * fovy

            x0 = (cen.data._lon.arcsec - .5 * xlength)
            y0 = (cen.data._lat.arcsec - .5 * ylength)
            bl = SC(x0 * u.arcsec, y0 * u.arcsec, frame='helioprojective',
                    obstime=st)  # Bottom Left and Top Right coordinates
            tr = SC((x0 + xlength) * u.arcsec, (y0 + ylength) * u.arcsec, frame='helioprojective', obstime=st)


        elif i > 0:
            # Get new start time
            st = datetime.datetime.strptime(mc[0].meta['DATE-OBS'], '%Y-%m-%d' + ' ' + '%H:%M:%S.%f')

            # Rotate Helio Projective Coordinates from end-start time (time elapsed between obs)
            if rotate == True:
                cen = diff_rot.solar_rotate_coordinate(cen, st, frame_time='synodic', rot_type='snodgrass')

            if coalign == True:
                rerun_shift = False
                ycrop = 225
                xcrop = 45
                yoffset = 25
                xoffset = 35
                template = mc.percentile(97).data[crpix2 - ycrop:crpix2 + ycrop, crpix1 - xcrop:crpix1 + xcrop]
                shift = calculate_shift(layer, template)
                #if xyshift hits shift boundaries, resize template, rerun calculate shift
                # while (shift[0].value == 2. * yoffset) or (shift[1].value == 2.*xoffset):
                #     if shift[0].value == 2. * yoffset:
                #         ycrop -= 10
                #         yoffset += 10
                #     if shift[1].value == 2.* xoffset:
                #         xcrop -= 10
                #         xoffset +=10
                #     template = mc.percentile(97).data[crpix2 - ycrop:crpix2 + ycrop, crpix1 - xcrop:crpix1 + xcrop]
                #     shift = calculate_shift(layer, template)


                print(shift)
                xshift += (shift[1] - xoffset*u.pix) * mc.scale[0]
                yshift += (shift[0] - yoffset*u.pix) * mc.scale[1]
                print(i, ' Xcorr: ', (xshift.round(0), yshift.round(0)))
                layer = mc.percentile(97).data[crpix2 - 250:crpix2 + 250, crpix1 - 80:crpix1 + 80]
                print('New layer created')
            xcen = (crval1 - xshift.value) * u.arcsec
            if rotate == True:
                ycen = cen.data._lat
            else:
                ycen = (crval2 - yshift.value) * u.arcsec
            print(xcen,ycen)
            cen = SC(xcen, ycen, frame='helioprojective', obstime=st)


            # New Submap Boundaries
            x0 = (cen.data._lon.arcsec - .5 * xlength)
            y0 = (cen.data._lat.arcsec - .5 * ylength)
            print(x0, y0)
            print(i, crval1- .5 * xlength, crval2- .5 * xlength)
            # Bottom Left and Top Right coordinates
            bl = SC(x0 * u.arcsec, y0 * u.arcsec, frame='helioprojective',obstime=st)
            tr = SC((x0 + xlength) * u.arcsec, (y0 + ylength) * u.arcsec, frame='helioprojective', obstime=st)

            # Get new end time
            et = datetime.datetime.strptime(mc[-1].meta['DATE-OBS'], '%Y-%m-%d' + ' ' + '%H:%M:%S.%f')

            # (Note: Rotating during obs helps minimizes jitter)
        if i < n / 2:
            cadence = 24
        else:
            cadence = 18
        for j, sji in enumerate(mc):

            if j % 25 == cadence:#( (i<=.5*len(sji_list) and j % 20 == 19) or (i>.5*len(sji_list) and j % 20 == 1)):
                # Create Cropped Submap

                submap = sji.submap(bl, tr)

                # Plot Data

                submap.plot(axes=ax0, norm=colors.PowerNorm(.4,0,100))
                #submap.draw_rectangle(bl2, 53.232 * u.arcsec, 166.35 * u.arcsec, color='black')
                #aia[i].plot(axes=ax1, norm=colors.LogNorm(10, 2000))
                # Draw GUIs
                note = ax0.annotate(title + ' ' + sji.meta['DATE-OBS'][:-7],
                                    xy=(bl.data._lon.arcsec, bl.data._lat.arcsec), color='white')
                #aia[i].draw_rectangle(bl, xlength * u.arcsec, ylength * u.arcsec, color='black')
                #bl2 = SC((crval1 - 26.616)*u.arcsec,(crval2 - 83.175)*u.arcsec,frame='helioprojective',obstime=st)


                # Label axes
                ax0.set_title(title, visible=False)
                ax0.set_xlabel('X [Arcsec]')
                ax0.set_ylabel('Y [Arcsec]')
                print(j,xcen, ycen)
                ax0.set_xlim(x0, x0 + xlength)
                ax0.set_ylim(y0, y0 + ylength)
                ax0.set_facecolor('black')
                # ax1.set_title('', visible=False)
                # ax1.set_xlabel('', visible=False)
                # ax1.set_ylabel('', visible=False)
                # ax1.set_xticks([])
                # ax1.set_yticks([])
                # ax1.set_facecolor('black')

                # Save Plot to output file
                writer.grab_frame()

                # Clear Plots
                ax0.cla()
                #ax1.cla()
        # Rotate Helio Projective Coordinates from startend time (time elapsed during obs)
        if rotate == True:
            cen = diff_rot.solar_rotate_coordinate(cen, et, frame_time='synodic', rot_type='snodgrass')

    end = time.perf_counter() - start

    print(end)  # Print time ran
    #todo:build new mapcube and return

def dustbuster(mc):
    """
    Read SJI fits files and return Inpaint-corrected fits files.
    Image inpainting involves filling in part of an image or video
    using information from the surrounding area.
    Parameters
    ----------
    mc: `sunpy.map.MapCube`
        Mapcube to read
    Returns
    -------
    mc: `sunpy.map.MapCube`
        Inpaint-corrected Mapcube
    """
    intscale(mc)
    image_inpaint = (mc.percentile(97).data) #(97th percentile to avoid SAA snow)
    for i, m in enumerate(mc):
        image_orig = m.data

        #  Create mask with values < 10)
        dustmask = ma.masked_inside(image_orig,-199,0)
        if dustmask.mask.shape != image_orig.shape:
            print("Dust not detected")
            m.plot()

        else:
        #  Dilate dust spots by 1 pixel
            dilate = generate_binary_structure(2, 2)
            dustmask.mask = binary_dilation(dustmask.mask, structure=dilate)
            #if (i)%nx>=nx-skip:
            #    skip = -1*(skip)
           # image_inpaint = mc[i + skip].data.copy()
            image_orig[dustmask.mask] = image_inpaint[dustmask.mask]

            #m.mask[dustmask.mask] = True
            #plt.imshow(m.mask)
            m.meta.add_history('Dustbuster correction applied, dustmask added to map mask')
    return mc
def intscale(mc):
    offset = 31968
    for m in mc:
        imdata=m.data
        if imdata.min() == BAD_PIXEL_VALUE_UNSCALED:
            imdata[:, :] = (imdata[:, :]+offset) * .25#.astype('uint8'))
            m.meta.add_history('Intscale applied')
    return mc

sji_animator(sji_list[0:])