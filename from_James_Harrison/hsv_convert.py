import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import netcdf
from copy import deepcopy
import xarray as xr
from os.path import dirname, realpath, join, getsize
from mastvideo import write_ipx_file, IpxHeader, IpxSensor, SensorType, ImageEncoding
from PIL import Image

def load_mraw(filename, HEIGHT_FULL=1024, WIDTH_FULL=1024, T0=-0.1):
    """
    Load HSV raw data into memory.

    :param int/str shot: MAST-U shot number.
    :param str fpath_data: Path to data directory, see above.
    :return: HSV data as an xarray.DataArray object.
    """
    
    # Split the input file name and path from the file extension
    fname, ext = filename.split('.')
    cih = load_cih(fname+'.cih')
    num, denom = cih['Shutter Speed(s)'].split('/')
    cih['Shutter Speed(s)'] = float(num) / float(denom)
    nframes = cih['Total Frame']
    height = cih['Image Height']
    width = cih['Image Width']
    fps = cih['Record Rate(fps)']

    xpix = np.arange(width) - width // 2 + WIDTH_FULL // 2
    ypix = np.arange(height)

    times = np.arange(nframes) * 1 / fps + T0
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # optimised loading of 12-bit uint data from binary file into numpy array
    # source: https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    data = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    data = data.reshape([nframes, height, width, ])

    return xr.DataArray(data, coords=(times, ypix, xpix), dims=('time', 'y', 'x'), attrs=cih)


def load_cih(filename):
    """
    Load HSV metadata.

    source: https://github.com/ladisk/pyMRAW

    :param int/str shot: MAST-U shot number.
    :param str fpath_data: Path to data directory, see above.
    :return: dict of metadata.
    """
    
    cih = dict()
    with open(filename, 'r') as f:
        for line in f:
            if line == '\n':  # end of cif header
                break
            line_sp = line.replace('\n', '').split(' : ')
            if len(line_sp) == 2:
                key, value = line_sp
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                    cih[key] = value
                except:
                    cih[key] = value
    return cih

data_dir = '/home/jrh/mast_data/44621/C001H001S0001/'
filename = 'C001H001S0001'
subwindow = [350, 750, 280, 680]

hsv_data = load_mraw(data_dir+filename+'.mraw')

tmp = hsv_data.attrs['Date'].split('/')
datestr = tmp[0] + '-' + "{:02d}".format(int(tmp[1])) + '-' + "{:02d}".format(int(tmp[2]))
tmp = hsv_data.attrs['Time'].split(':')
timestr = "{:02d}".format(int(tmp[0])) + ':' + "{:02d}".format(int(tmp[1])) + ':00'

# Fill in the header information
if subwindow is None:
    header = IpxHeader(shot = 44621,
                       date_time = datestr + 'T' + timestr,  #2008-05-07T14:20:40',
                       camera = hsv_data.attrs['Camera Type'],
                       view = 'HM08',
                       lens = 'HSV relay',
                       trigger = -0.1,
                       exposure = int(hsv_data.attrs['Shutter Speed(s)']*1.0E6),
                       num_frames = len(hsv_data),
                       frame_width = hsv_data.attrs['Image Width'],
                       frame_height = hsv_data.attrs['Image Height'],
                       depth = hsv_data.attrs['EffectiveBit Depth'])
    
    # Create a sensor object
    sensor = IpxSensor(type=SensorType.MONO)
    
else:
    header = IpxHeader(shot = 44621,
                       date_time = datestr + 'T' + timestr,  #2008-05-07T14:20:40',
                       camera = hsv_data.attrs['Camera Type'],
                       view = 'HM08',
                       lens = 'HSV relay',
                       trigger = -0.1,
                       exposure = int(hsv_data.attrs['Shutter Speed(s)']*1.0E6),
                       num_frames = len(hsv_data),
                       frame_width = subwindow[1] - subwindow[0],
                       frame_height = subwindow[3] - subwindow[2],
                       depth = hsv_data.attrs['EffectiveBit Depth'])

    # Create a sensor object
    sensor = IpxSensor(type=SensorType.MONO,
                       window_left = subwindow[0],
                       window_right = subwindow[1],
                       window_top = subwindow[2],
                       window_bottom = subwindow[3])

with write_ipx_file('testing.ipx', header, sensor, version = 1, encoding=ImageEncoding.JPEG2K) as ipx:
    for i in np.arange(len(hsv_data)):
        img = np.array(hsv_data[i].data[subwindow[2]:subwindow[3], subwindow[0]:subwindow[1]]*16, dtype=np.uint16)
        ipx.write_frame(hsv_data.time.data[i], Image.fromarray(img, 'I;16'))
