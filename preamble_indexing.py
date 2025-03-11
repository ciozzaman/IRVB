# Created 03/12/2018
# Fabio Federici

collection_of_records = dict([])
# integration time 1ms, freme rate 50Hz, no view port, cooling ramp, temperature probe bottom right looking from camera      !!! ! ! B A D   D A T A ! ! !!!
# temperature1=[38.4,35.4,33.7,32.6,31.6,30.6,29.6,28.6,27.6,26.6,25.6]
# #files1=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000011']
temperature1=[33.7,32.6,31.6,30.6,29.6,28.6,27.6,26.6,25.6]
files1=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000011']
full_pathfile_index = xr.Dataset({'files1':files1})


# integration time 1ms, freme rate 50Hz, no view port, cooling ramp, temperature probe top right looking from camera
# temperature2=[44.0,42.9,41.9,40.9,39.9,38.9,37.9,36.9,35.9,34.9,33.9,32.9,31.9,30.9,29.9,28.9,27.9,26.9,25.9,24.9]
# #files2=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000031']
temperature2=[41.9,40.9,39.9,38.9,37.9,36.9,35.9,34.9,33.9,32.9,31.9,30.9,29.9,28.9,27.9,26.9,25.9,24.9]
files2=['/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar07_2018/ff_full-000031']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files2':files2}))


# integration time 1ms, freme rate 50Hz, no view port, cooling ramp, temperature probe centre right looking from camera
# temperature3=[41.8,40.8,39.3,38.8,37.8,36.8,35.8,34.8,33.8,32.8,31.8,30.8,29.8,28.8,27.8]
# #files3=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000015']
temperature3=[39.3,38.8,37.8,36.8,35.8,34.8,33.8,32.8,31.8,30.8,29.8,28.8,27.8]
files3=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000015']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files3':files3}))


# integration time 1ms, freme rate 50Hz, no view port, cooling ramp, temperature probe centre left looking from camera
# temperature4=[44.3,43.3,41.1,38.8,37.2,36.2,34.6,32.9,32.2,31.2,30.4,29.6,29.0,28.4,27.8]
# #files4=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000030']
#temperature4=[41.1,38.8,37.2,36.2,34.6,32.9,32.2,31.2,30.4,29.6,29.0,28.4,27.8]
#files4=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000030']
temperature4=[41.1,38.8,36.2,34.6,32.9,32.2,31.2,30.4,29.6,29.0,28.4,27.8]
files4=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000030']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files4':files4}))


# integration time 1ms, freme rate 50Hz, no view port, cooling ramp, temperature probe bottom right looking from camera
# temperature5=[45.1,44.1,43.1,42.1,41.1,40.1,39.1,38.1,37.1,36.1,35.1,34.1,33.1,32.1,31.1,30.1,29.1,28.1]
# #files5=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000048']
temperature5=[43.1,42.1,41.1,40.1,39.1,38.1,37.1,36.1,35.1,34.1,33.1,32.1,31.1,30.1,29.1,28.1]
files5=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000048']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files5':files5}))


# integration time 1ms, freme rate 50Hz, no view port, heating ramp, temperature probe top left looking from camera
#  temperature6=[13.3,13.5,13.7,14.5,14.9,15.6,16.3,16.9,17.6,18.2,18.9,19.6,20.2,20.8,21.4]
# #files6=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000063']
temperature6=[13.7,14.5,14.9,15.6,16.3,16.9,17.6,18.2,18.9,19.6,20.2,20.8,21.4]
files6=['/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000063']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files6':files6}))

# integration time 2ms, freme rate 383Hz, no view port, heating ramp, temperature probe bottom left looking from camera
# temperature7=[16,16.2,16.8,17.2,17.7,18.2,18.7,19.2,19.7,20.2,20.7,21.2,21.6,22]
# #files7=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000014']
temperature7=[16.8,17.2,17.7,18.2,18.7,19.2,19.7,20.2,20.7,21.2,21.6,22]
files7=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000014']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files7':files7}))


# integration time 2ms, freme rate 383Hz, no view port, cooling ramp, temperature probe bottom left looking from camera
# temperature8=[36.8,36.1,35.5,35.0,34.5,33.8,33.0,32.3,31.4,30.6,30.0,29.2,28.5,28.1,24.9,24.6,24.3,23.9,23.6,23.3]
# #files8=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000041']
temperature8=[33.8,33.0,32.3,31.4,30.6,30.0,29.2,28.5,28.1,24.9,24.6,24.3,23.9,23.6,23.3]
files8=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000041']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files8':files8}))


# integration time 2ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
#temperature10=[39.6,37.4,36.8,36.0,35.1,34.1,32.7,31.5,30.2,29.6,29.5,29.0,28.6,28.1,27.7,27.3,26.9,26.6,26.4,26.2,26.0]
temperature10=[29.6,29.5,29.0,28.6,28.1,27.7,27.3,26.9,26.6,26.4,26.2,26.0]
#files10=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000034']
files10=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000034']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files10':files10}))


# integration time 1ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
temperature11=[31.8,30.8,29.9,29.2,28.7,28.3,27.9,27.5,27.2,26.8,26.5,26.3,26.1,25.9]
files11=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000035']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files11':files11}))


# integration time 2ms, freme rate 383Hz, view port, off centre aberration, cooling ramp, temperature probe bottom right looking from camera
#temperature12=[41.3,40.1,39.1,38.1,37.1,36.4,35.7,34.9,33.8,33.3,32.5,31.8,30.9,30.1,29.5,28.8,28.1,27.6,27.1,26.5,26.1]
temperature12=[30.1,29.5,28.8,28.1,27.6,27.1,26.5,26.1]
#files12=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000081']
files12=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000081']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files12':files12}))


# integration time 1ms, freme rate 383Hz, view port, off centre aberration, cooling ramp, temperature probe bottom right looking from camera
#temperature13=[44.6,43.8,43.0,42.5,41.9,40.8,39.6,38.6,37.6,36.8,36.0,35.3,34.2,33.5,32.9,32.2,31.5,30.6,29.8,28.6,28.4,27.7,27.4,26.6,26.2]
temperature13=[43.8,43.0,42.5,41.9,40.8,39.6,38.6,37.6,36.8,36.0,35.3,34.2,33.5,32.9,31.5,30.6,29.8,28.6,28.4,27.7,27.4,26.6,26.2]
# temperature13=[31.5,30.6,29.8,28.9,28.4,27.7,27.4,26.6,26.2]
#files13=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000080']
files13=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000080']
# files13=['/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May10_2018/ff_full-000080']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files13':files13}))


# integration time 1ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
# temperature14=[43.3,42.2,40.9,40.1,39.5,39.0,38.5,38.0,37.2,35.9,34.9,34.0,33.3,32.5,31.8,30.9,30.3,29.6,29.0,28.6,28.1,27.7,27.3,26.9,26.5,26.2,25.8,25.4,25.0,24.5,24.1]
# files14=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000053']
temperature14=[42.2,40.9,40.1,39.5,39.0,38.5,38.0,37.2,35.9,34.9,34.0,33.3,32.5,31.8,30.9,30.3,29.6,29.0,28.6,28.1,27.7,27.3,26.9,26.5,26.2,25.8,25.4,25.0,24.5,24.1]
files14=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000053']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files14':files14}))


# integration time 2ms, freme rate 383Hz, view port, off centre aberration, cooling ramp, temperature probe bottom right looking from camera
# temperature15=[36.5,35.4,34.5,33.7,32.9,32.1,31.4,30.6,29.9,29.3,28.8,28.4,27.8,27.5,27.1,26.7,26.4,26.0,25.6,25.2,24.8,24.3,23.9]
# files15=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000054']
temperature15=[34.5,33.7,32.9,32.1,31.4,30.6,29.9,29.3,28.8,28.4,27.8,27.5,27.1,26.7,26.4,26.0,25.6,25.2,24.8,24.3,23.9]
files15=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000054']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files15':files15}))


# integration time 1ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
# temperature16=[41.2,40.6,39.8,38.8,38.2,37.5,36.9,36.3,35.4,34.7,34.0,33.3,32.7,32.1,31.6,31.0,30.4,29.8,29.2,28.6,27.9,27.5,26.8,26.4,26.1,25.7,25.4,25.0,24.6]
# files16=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000082','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000102']
# temperature16=[40.6,39.8,38.8,38.2,37.5,36.9,36.3,35.4,34.7,34.0,33.3,32.7,32.1,31.6,31.0,30.4,29.8,29.2,28.6,27.9,27.5,26.8,26.4,26.1,25.7,25.4,25.0,24.6]
# files16=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000082','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000102']
temperature16=[40.6,39.8,38.8,38.2,37.5,36.9,36.3,35.4,34.7,34.0,33.3,32.7,32.1,31.6,31.0,30.4,29.8,29.2,28.6,27.9,27.5,26.8,26.4,25.7,25.4,25.0,24.6]
files16=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000082','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000102']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files16':files16}))


# integration time 2ms, freme rate 383Hz, view port, off centre aberration, cooling ramp, temperature probe bottom right looking from camera
# temperature17=[35.7,35.1,34.3,33.7,33.0,32.3,31.9,31.3,30.7,30.1,29.4,28.9,28.3,27.7,27.2,26.6,25.6,25.1,24.8,24.4]
# files17=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000081','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000083','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000085','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000087','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000103']
temperature17=[34.3,33.7,33.0,32.3,31.9,31.3,30.7,30.1,29.4,28.9,28.3,27.7,27.2,26.6,25.6,25.1,24.8,24.4]
files17=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000081','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000083','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000085','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000087','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000103']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files17':files17}))


# integration time 1ms, freme rate 383Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera
# temperature18=[15.7,16.2,16.8,17.3,17.8,18.3,19.1,19.5,20.0,20.5,21.0,21.5,22.1,22.4,22.8,23.2,23.4,23.8]
# files18=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000104','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000106','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000114','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000117','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000119','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000121','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000123','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000125','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000127','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000129','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000131','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000133','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000135','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000137','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000139']
temperature18=[16.8,17.3,17.8,18.3,19.1,19.5,20.0,20.5,21.0,21.5,22.1,22.4,22.8,23.2,23.4,23.8]
files18=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000114','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000117','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000119','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000121','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000123','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000125','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000127','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000129','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000131','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000133','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000135','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000137','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000139']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files18':files18}))


# integration time 2ms, freme rate 383Hz, view port, off centre aberration, heating ramp, temperature probe bottom right looking from camera
# temperature19=[15.9,16.5,17.0,17.6,18.0,18.6,19.3,19.8,20.2,20.8,21.3,21.7,22.2,22.7,22.9,23.2,23.6]
# files19=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000105','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000107','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000115','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000118','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000120','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000122','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000124','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000126','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000128','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000130','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000132','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000134','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000136','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000138']
temperature19=[17.0,17.6,18.0,18.6,19.3,19.8,20.2,20.8,21.3,21.7,22.2,22.7,22.9,23.2,23.6]
files19=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000115','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000118','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000120','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000122','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000124','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000126','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000128','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000130','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000132','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000134','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000136','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000138']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files19':files19}))


# integration time 1ms, freme rate 383Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera
#temperature20=[16.2,16.7,17.2,17.7,18.3,18.9,19.5,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0,23.4]
#files20=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000140','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000142','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000144','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000146','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000148','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000150','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000152','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000154','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000156','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000158','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000160','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000162','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000164','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000166','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000168','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000170']
# temperature20=[16.2,16.7,17.2,17.7,18.9,19.5,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0,23.4]
# files20=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000140','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000142','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000144','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000146','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000150','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000152','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000154','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000156','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000158','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000160','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000162','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000164','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000166','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000168','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000170']
temperature20=[17.2,17.7,18.9,19.5,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0,23.4]
files20=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000144','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000146','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000150','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000152','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000154','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000156','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000158','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000160','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000162','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000164','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000166','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000168','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000170']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files20':files20}))


# integration time 2ms, freme rate 383Hz, view port, off centre aberration, heating ramp, temperature probe bottom right looking from camera
# temperature21=[16.4,16.9,17.5,18.0,18.6,19.2,19.8,20.3,20.8,21.2,21.6,22.0,22.4,22.8,23.2,23.6]
# files21=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000141','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000143','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000145','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000147','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000149','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000151','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000153','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000155','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000157','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000159','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000161','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000163','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000165','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000167','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000169','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000171']
temperature21=[17.5,18.0,18.6,19.2,19.8,20.3,20.8,21.2,21.6,22.0,22.4,22.8,23.2,23.6]
files21=['/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000145','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000147','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000149','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000151','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000153','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000155','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000157','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000159','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000161','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000163','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000165','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000167','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000169','/home/ffederic/work/irvb/flatfield/May14_2018/ff_full-000171']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files21':files21}))


# integration time 1ms, freme rate 994Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera, partial frame (width 320 height 64 x offset 0 y offset 96)
temperature22=[40.1,39.5,39.0,38.3,37.8,37.2,36.6,35.8,35.3,34.8,34.1,33.5,33.1,32.7,32.3,31.9,31.5,31.1,30.7,30.3,29.7,29.2,28.6,28.1,27.5,27.0,26.5,26.0,25.5,24.8]
files22=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000002','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000003','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000004','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000005','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000006','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000007','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000008','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000009','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000010','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000011','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000012','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000013','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000014','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000015','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000016','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000017','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000018','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000019','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000020','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000021','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000022','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000023','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000024','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000025','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000026','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000027','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000028','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000029','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000030','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000031']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files22':files22}))


# integration time 1ms, freme rate 994Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera, partial frame (width 320 height 64 x offset 0 y offset 96)
# temperature23=[15.8,16.1,16.6,17.7,18.0,18.1,18.5,18.7,18.9,19.3,19.7,20.0,20.3,20.6,20.9,21.2,21.5,21.8,22.1,22.4,23.0]
# files23=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000032','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000033','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000034','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000036','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000037','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000038','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000039','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000040','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000041','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000042','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000043','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000044','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000045','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000046','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000047','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000048','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000049','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000050','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000051','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000052','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000053']
temperature23=[16.1,16.6,17.7,18.0,18.1,18.5,18.7,18.9,19.3,19.7,20.0,20.3,20.6,20.9,21.2,21.5,21.8,22.1,22.4,23.0]
files23=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000033','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000034','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000036','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000037','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000038','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000039','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000040','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000041','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000042','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000043','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000044','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000045','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000046','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000047','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000048','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000049','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000050','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000051','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000052','/home/ffederic/work/irvb/flatfield/May15_2018/ff_part-000053']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files23':files23}))


# integration time 0.5ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
temperature24=[39.2,38.3,37.2,36.4,35.6,34.9,34.3,33.7,33.1,32.5,31.9,31.3,30.7,30.1,29.5,28.9,28.3,27.6,27.1,26.5,25.9,25.2]
files24=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000044']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files24':files24}))


# integration time 1.5ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
temperature25=[38.7,37.7,36.8,35.9,35.3,34.6,34.0,33.4,32.8,32.2,31.6,31.0,30.4,29.8,29.2,28.6,28.0,27.7,27.4,26.8,26.2,25.6,25.0]
files25=['/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May15_2018/ff_full-000045']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files25':files25}))


# integration time 0.5ms, freme rate 383Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera
# temperature26=[15.0,15.4,15.9,16.4,16.9,17.6,18.0,18.5,18.8,19.3,19.7,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0]
# files26=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000001','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000003','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000037']
temperature26=[15.9,16.4,16.9,17.6,18.0,18.5,18.8,19.3,19.7,20.1,20.6,21.0,21.4,21.8,22.2,22.6,23.0]
files26=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000005','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000007','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000009','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000011','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000013','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000015','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000017','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000019','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000021','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000023','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000025','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000037']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files26':files26}))


# integration time 1.5ms, freme rate 383Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera
# temperature27=[15.2,15.7,16.2,16.7,17.3,17.8,18.2,18.6,19.1,19.5,19.9,20.4,20.8,21.4,21.6,22.0,22.4,22.8]
# files27=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000002','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000036']
temperature27=[15.7,16.2,16.7,17.3,17.8,18.2,18.6,19.1,19.5,19.9,20.4,20.8,21.4,21.6,22.0,22.4,22.8]
files27=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000004','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000006','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000008','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000010','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000012','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000014','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000016','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000018','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000020','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000022','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000024','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000026','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000036']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files27':files27}))


# integration time 0.5ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
temperature28=[41.1,40.0,39.0,38.0,37.1,36.0,35.2,34.4,33.6,32.9,32.2,31.5,30.8,30.1,29.4,28.8,28.2,27.4,26.8,26.2,25.6,25.0,24.4]
files28=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000042','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000044','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000046','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000048','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000050','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000052','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000054','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000056','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000058','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000060','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000062','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000064','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000066','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000068','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000070','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000072','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000074','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000076','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000078','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000080','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000082']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files28':files28}))


# integration time 1.5ms, freme rate 383Hz, view port, central aberration, cooling ramp, temperature probe bottom right looking from camera
temperature29=[40.5,39.5,38.5,37.4,36.7,35.6,34.8,34.0,33.3,32.5,31.9,31.1,30.4,29.8,29.1,28.5,27.7,27.1,26.5,25.9,25.3,24.7,24.2]
files29=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000041','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000043','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000045','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000047','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000049','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000051','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000053','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000055','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000057','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000059','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000061','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000063','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000065','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000067','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000069','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000071','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000073','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000075','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000077','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000079','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000081','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000083']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files29':files29}))


# integration time 0.5ms, freme rate 383Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera
# temperature30=[16.5,17.0,17.5,18.0,18.4,18.8,19.3,19.7,20.1,20.5,20.9,21.3,21.7,22.2,22.6,23.0]
# files30=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000085','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000087','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000103','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000105','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000107','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000115']
temperature30=[17.5,18.0,18.4,18.8,19.3,19.7,20.1,20.5,20.9,21.3,21.7,22.2,22.6,23.0]
files30=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000089','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000091','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000093','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000095','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000097','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000099','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000101','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000103','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000105','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000107','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000109','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000111','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000113','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000115']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files30':files30}))


# integration time 1.5ms, freme rate 383Hz, view port, central aberration, heating ramp, temperature probe bottom right looking from camera
# temperature31=[16.3,16.7,17.3,17.7,18.2,18.6,19.1,19.5,19.9,20.3,20.7,21.1,21.5,22.0,22.4,22.8]
# files31=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000084','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000102','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000104','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000106','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000114']
temperature31=[16.7,17.3,17.7,18.2,18.6,19.1,19.5,19.9,20.3,20.7,21.1,21.5,22.0,22.4,22.8]
files31=['/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000086','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000088','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000090','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000092','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000094','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000096','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000098','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000100','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000102','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000104','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000106','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000108','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000110','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000112','/home/ffederic/work/irvb/flatfield/May16_2018/ff_full-000114']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files31':files31}))


# added 2021/12/08 after the calibration with the black body source
# NOTE: the coating on the plasma side of the window got completely destroyed before the measurement, so the consistency with the MASTU data is not assured

# integration time 2.5ms, freme rate 50Hz, view port, central aberration (copy from 44786), heating ramp with black body source starting from cold winter ambient temperature
# temperature40=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
# files40=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000014','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000015','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000024','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000025','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000034','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000035','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000044','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000045','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000050','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000055','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000060','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000065','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000070','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000075','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000080','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000085','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000090','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000095','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000100','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000105','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000110','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000115']
temperature40=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
files40=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000014','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000015','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000024','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000025','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000034','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000035','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000044','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000045','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000050','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000055','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000060','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000065','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000070','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000075','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000080','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000085','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000090','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000095','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000100','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000105']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files40':files40}))

# integration time 2ms, freme rate 50Hz, view port, central aberration (copy from 44786), heating ramp with black body source starting from cold winter ambient temperature
# temperature41=[9.4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
# files41=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000005','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000009','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000010','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000016','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000023','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000026','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000033','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000036','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000043','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000046','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000051','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000056','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000061','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000066','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000071','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000076','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000081','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000086','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000091','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000096','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000101','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000106','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000111','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000116','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000120','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000124','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000128','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000132','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000136','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000140','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000144','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000148']
temperature41=[9.4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
files41=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000005','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000009','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000010','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000016','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000023','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000026','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000033','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000036','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000043','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000046','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000051','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000056','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000061','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000066','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000071','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000076','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000081','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000086','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000091','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000096','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000101','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000106','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000111','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000116','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000120','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000124','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000128','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000132','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000136','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000140']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files41':files41}))

# integration time 1.5ms, freme rate 50Hz, view port, central aberration (copy from 44786), heating ramp with black body source starting from cold winter ambient temperature
temperature45=[9.4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
files45=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000004','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000008','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000011','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000017','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000022','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000027','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000032','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000037','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000042','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000047','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000052','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000057','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000062','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000067','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000072','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000077','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000082','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000087','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000092','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000097','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000102','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000107','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000112','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000117','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000121','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000125','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000129','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000133','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000137','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000141','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000145','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000149','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000152','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000155','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000158','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000161','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000164']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files45':files45}))

# integration time 1ms, freme rate 50Hz, view port, central aberration (copy from 44786), heating ramp with black body source starting from cold winter ambient temperature
temperature42=[9.4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
files42=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000002','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000007','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000012','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000018','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000021','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000028','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000031','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000038','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000041','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000048','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000053','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000058','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000063','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000068','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000073','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000078','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000083','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000088','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000093','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000098','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000103','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000108','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000113','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000118','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000122','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000126','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000130','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000134','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000138','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000142','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000146','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000150','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000153','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000156','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000159','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000162','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000165']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files42':files42}))

# integration time 0.5ms, freme rate 50Hz, view port, central aberration (copy from 44786), heating ramp with black body source starting from cold winter ambient temperature
temperature43=[9.4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
files43=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000003','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000006','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000013','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000019','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000020','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000029','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000030','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000039','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000040','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000049','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000054','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000059','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000064','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000069','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000074','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000079','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000084','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000089','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000099','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000104','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000109','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000114','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000119','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000123','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000127','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000131','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000135','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000139','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000143','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000147','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000151','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000154','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000157','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000160','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000163','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000166']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files43':files43}))

# integration time 2.5ms, freme rate 50Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature44=[25,27,28]
files44=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000167','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000172','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000177']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files44':files44}))

# integration time 2ms, freme rate 50Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature46=[25,27,28,32,36]
files46=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000168','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000173','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000178','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000182','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000186']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files46':files46}))

# integration time 1.5ms, freme rate 50Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature47=[25,27,28,32,36,43]
files47=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000169','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000174','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000179','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000183','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000187','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000190']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files47':files47}))

# integration time 1ms, freme rate 50Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature48=[25,27,28,32,36,43]
files48=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000170','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000175','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000180','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000184','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000188','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000191']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files48':files48}))

# integration time 0.5ms, freme rate 50Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature49=[25,27,28,32,36,43]
files49=['/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000171','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000176','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000181','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000185','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000189','/home/ffederic/work/irvb/flatfield/Dec07_2021/flat_field-000192']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files49':files49}))


# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 2.5ms, freme rate 10Hz, view port, ~ central aberration, heating ramp with black body source starting from cold winter ambient temperature
temperature55=[17,19,21,23,25,27]
files55=['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000016','/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000026','/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000036','/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000046','/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000061','/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000071']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files55':files55}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 2.0ms, freme rate 10Hz, view port, ~ central aberration, heating ramp with black body source starting from cold winter ambient temperature
temperature56=[17,19,21,23,25,27,30,34]
files56 = ['017','027','037','047','062','072','081','089']
files56 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files56]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files56':files56}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 1.5ms, freme rate 10Hz, view port, ~ central aberration, heating ramp with black body source starting from cold winter ambient temperature
temperature57=[17,19,21,23,25,27,30,34]
files57 = ['018','028','038','048','063','073','082','090']
files57 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files57]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files57':files57}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 1ms, freme rate 10Hz, view port, ~ central aberration, heating ramp with black body source starting from cold winter ambient temperature
temperature58=[17,19,21,23,25,27,30,34]
files58 = ['019','029','039','049','064','074','083','091']
files58 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files58]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files58':files58}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 0.5ms, freme rate 10Hz, view port, ~ central aberration, heating ramp with black body source starting from cold winter ambient temperature
temperature59=[17,19,21,23,25,27,30,34]
files59 = ['020','030','040','050','065','075','084','092']
files59 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files59]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files59':files59}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 2.5ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature60=[17,19,21,23,25,27]
files60 = ['011','021','031','041','056','066']
files60 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files60]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files60':files60}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 2ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature61=[17,19,21,23,25,27,30,34]
files61 = ['012','022','032','042','057','067','077','085']
files61 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files61]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files61':files61}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 1.5ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature62=[17,19,21,23,25,27,30,34]
files62 = ['013','023','033','043','058','068','078','086']
files62 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files62]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files62':files62}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 1ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature63=[17,19,21,23,25,27,30,34]
files63 = ['014','024','034','044','059','069','079','087']
files63 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files63]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files63':files63}))

# 24/02/2022 new scan with BB source keeping the camera as far as possible (~18cm) to try have the most homogeneous radiation field possible (smaller solid angle)
# integration time 0.5ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature64=[17,19,21,23,25,27,30,34]
files64 = ['015','025','035','045','060','070','080','088']
files64 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files64]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files64':files64}))

# 24/02/2022 new scan with BB source keeping the camera such that the BB source is in focus (65.7cm) as prescribed by Andrew Thornton
# integration time 2ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature65=[17,19,22,25,27,30,34]
files65 = ['096','098','100','102','104','106','108']
files65 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files65]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files65':files65}))

# 24/02/2022 new scan with BB source keeping the camera such that the BB source is in focus (65.7cm) as prescribed by Andrew Thornton
# integration time 1ms, freme rate 10Hz, no view port, heating ramp with black body source starting from cold winter ambient temperature
temperature66=[17,19,22,25,27,30,34]
files66 = ['097','099','101','103','105','107','109']
files66 = ['/home/ffederic/work/irvb/flatfield/Feb24_2022/flat_field-000'+value for value in files66]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files66':files66}))


# 20/06/2024 Temperature calibration done with the new BB source capable of filling the full FOV and the old FLIR SC7500 camera. distance camera/BB source as per the current geometry with the 60mm stand-off
# integration time 1ms, freme rate 383Hz, view port damaged after the vacuum break after MU01, but with the coating facing the camera, heating ramp with new HGH BB source.
temperature70 = np.arange(40,-6,-1)
files70 = ['01','04','05','08','09','12','13','16','17','20','21','24','25','28','29','32','33','36','37','40','41','44','45','48','49','52','53','56','57','60','61','64','65','68','69','72','73','76','77','80','81','84','85','88','89','92']
files70 = ['/home/ffederic/work/irvb/flatfield/Jun20_2024/flat_field_00'+value for value in files70]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files70':files70}))

# 20/06/2024 Temperature calibration done with the new BB source capable of filling the full FOV and the old FLIR SC7500 camera. distance camera/BB source as per the current geometry with the 60mm stand-off
# integration time 2ms, freme rate 383Hz, view port damaged after the vacuum break after MU01, but with the coating facing the camera, heating ramp with new HGH BB source.
temperature71 = np.arange(40,-6,-1)
files71 = ['02','03','06','07','10','11','14','15','18','19','22','23','26','27','30','31','34','35','38','39','42','43','46','47','50','51','54','55','58','59','62','63','66','67','70','71','74','75','78','79','82','83','86','87','90','91']
files71 = ['/home/ffederic/work/irvb/flatfield/Jun20_2024/flat_field_00'+value for value in files70]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'files71':files71}))





##  FILES RELATIVE TO THE LASER MEASUREMENTS

reflaserpower=[5,4.97,3.97,2.97,2.03,1.52,0.99,0.62,0.28,0.06,0.03,0]
reflaserpower=np.multiply(0.001,np.flip(reflaserpower,0))
reflaserfvoltage=[3,1.4,1.2,1,0.8,0.7,0.6,0.5,0.45,0.4,0.35,0]
reflaserfvoltage=np.flip(reflaserfvoltage,0)

# laser 08/03/2018 1ms 50Hz
# Use     reflaserpower    and   reflaserfvoltage
laser1=['/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Mar08_2018/irvb_full-000011']
voltlaser1=[2,1.6,1.4,1.2,1,0.8,0.6,0.4,0.3,0.2,2]
freqlaser1=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,10]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser1':laser1}))


# laser 09/03/2018 2ms 383Hz
# Use     reflaserpower    and   reflaserfvoltage
laser2=['/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Mar09_2018/irvb_full-000015']
voltlaser2=[1.4,1.2,1,0.8,0.7,0.6,0.5,0.45,0.4,0.35,1.2,1.2,1.2,1.2,1.2]
freqlaser2=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser2':laser2}))


# Laser experiments 25/07/2018 1ms 383Hz focused laser straight on pinhole out of aberration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser10=['/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000014']
voltlaser10=[0.05,0.1,0.25,0.35,0.5,0.6,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser10=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser10=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser10':laser10}))


# Laser experiments 25/07/2018 1ms 383Hz partially_defocused laser straight on pinhole out of aberration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser11=['/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000028']
voltlaser11=[0.05,0.1,0.25,0.35,0.5,0.6,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser11=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser11=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser11':laser11}))


# Laser experiments 25/07/2018 1ms 994Hz (width=320, height=32, xoffset=0, yoffset=64, invert (V flip) selected) partially_defocused laser straight on pinhole out of aberration with low duty cycle
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
# NOTE I CAN'T USE THIS DATA, I DON'T HAVE ITS REFERENCE FRAME!
laser12=['/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000036','/home/ffederic/work/irvb/laser/Jul25_2018/irvb_full-000037']
voltlaser12=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser12=[10,50,100,10,50,100,10,50,100]
dutylaser12=[0.02,0.02,0.02,0.05,0.05,0.05,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser12':laser12}))


# Laser experiments 02/08/2018 1ms 383Hz focused laser further out of aberration, away from the feature given by calibration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser13=['/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000007']
voltlaser13=[0.05,0.1,0.25,0.35,0.5,0.6,0.7]
freqlaser13=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]
dutylaser13=[0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser13':laser13}))


# Laser experiments 02/08/2018 1ms 383Hz partially_defocused laser position on the foil as high as possible
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !reflaserpower1=[4.14,4.14,0,0]
# Use     reflaserpower1    and   reflaserfvoltage1
laser14=['/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Aug02_2018/irvb_full-000014']
voltlaser14=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser14=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]
dutylaser14=[0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser14':laser14}))


# Laser experiments 02/08/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum1=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug02_2018/irvb_full-000004']
vacuumframerate1=[383,994,383,994]
vacuuminttime1=[1,1,1,1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum1':vacuum1}))


# Voltage / power correlation of red laser from 02/08/2018
# reflaserpower1=[4.14,4.14,0,0]
# reflaserpower1=np.multiply(0.001,np.flip(reflaserpower1,0))
# reflaserfvoltage1=[10,0.505,0.012,0]
# reflaserfvoltage1=np.flip(reflaserfvoltage1,0)
# power_interpolator1 = interp1d([0,0.0079,0.5,10],[0,0,4.16*1e-3,4.16*1e-3])
power_interpolator1 = interp1d([0,0.0079,0.5,10],[0,0,4.15*1e-3,4.15*1e-3])	# this is obtained using R=0.46, that it is more accurate that the previous estimate of R=0.47

vacuumtest1=['/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000007-001_03_30_27_189','/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000008-001_03_49_16_589','/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000009-001_03_59_09_189','/home/ffederic/work/irvb/vacuum_chamber_testing/Jul18_2018/irvb_sample-000010-001_04_13_48_036']
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuumtest1':vacuumtest1}))


# Laser experiments 08/08/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum2=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000029','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000030','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000031','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000032','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000033','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000034','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug08_2018/irvb_sample-000035']
vacuumframerate2=[383,994,994,994,994,994,994,994,994,994,383,994,994,994,994]
vacuuminttime2=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
vacuumROI2 = ['ff' , [[64,159],[0,319]] , [[64,127],[0,319]] , [[64,95],[0,319]] , [[0,255],[192,255]] , [[0,255],[192,255]] , [[64,159],[0,319]] , [[64,127],[0,319]] , [[64,95],[0,319]] , \
			[[0,255],[192,255]] , 'ff' , 'ff' , 'ff' , 'ff' ,  [[64,127],[0,319]] ,  [[64,127],[0,319]] ,  [[64,127],[0,319]] ,  [[64,127],[0,319]] , 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , \
			 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , [[64,159],[0,319]] , [[64,127],[0,319]] , [[64,95],[0,319]] , [[0,255],[192,255]]]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum2':vacuum2}))


# Laser experiments 08/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) focused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser15=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000015']
voltlaser15=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser15=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser15=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser15':laser15}))
laserROI15 = [ [[64,127],[0,319]] ] * len(laser15)
collection_of_records['laser15'] = dict([])
collection_of_records['laser15']['path_files_laser'] = laser15
collection_of_records['laser15']['voltlaser'] = voltlaser15
collection_of_records['laser15']['freqlaser'] = freqlaser15
collection_of_records['laser15']['dutylaser'] = dutylaser15
collection_of_records['laser15']['laserROI'] = laserROI15
collection_of_records['laser15']['reference_clear'] = [vacuum2] * len(laser15)
collection_of_records['laser15']['power_interpolator'] = [power_interpolator1] * len(laser15)
collection_of_records['laser15']['focus_status'] = ['focused'] * len(laser15)
collection_of_records['laser15']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser15)
collection_of_records['laser15']['scan_type'] = 'freq&duty'


# Laser experiments 08/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) partially_defocused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser16=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000030']
voltlaser16=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser16=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser16=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser16':laser16}))
laserROI16 = [ [[64,127],[0,319]] ] * len(laser16)
collection_of_records['laser16'] = dict([])
collection_of_records['laser16']['path_files_laser'] = laser16
collection_of_records['laser16']['voltlaser'] = voltlaser16
collection_of_records['laser16']['freqlaser'] = freqlaser16
collection_of_records['laser16']['dutylaser'] = dutylaser16
collection_of_records['laser16']['laserROI'] = laserROI16
collection_of_records['laser16']['reference_clear'] = [vacuum2] * len(laser16)
collection_of_records['laser16']['power_interpolator'] = [power_interpolator1] * len(laser16)
collection_of_records['laser16']['focus_status'] = ['partially_defocused'] * len(laser16)
collection_of_records['laser16']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser16)
collection_of_records['laser16']['scan_type'] = 'freq&duty'


# Laser experiments 08/08/2018 1ms 383Hz focused laser as low as possible close to the corner
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser17=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000036','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000037','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000039','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000040','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000042','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000043']
voltlaser17=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser17=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser17=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser17':laser17}))
laserROI17 = [ 'ff' ] * len(laser17)
collection_of_records['laser17'] = dict([])
collection_of_records['laser17']['path_files_laser'] = laser17
collection_of_records['laser17']['voltlaser'] = voltlaser17
collection_of_records['laser17']['freqlaser'] = freqlaser17
collection_of_records['laser17']['dutylaser'] = dutylaser17
collection_of_records['laser17']['laserROI'] = laserROI17
collection_of_records['laser17']['reference_clear'] = [vacuum2] * len(laser17)
collection_of_records['laser17']['power_interpolator'] = [power_interpolator1] * len(laser17)
collection_of_records['laser17']['focus_status'] = ['focused'] * len(laser17)
collection_of_records['laser17']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser17)
collection_of_records['laser17']['scan_type'] = 'power&freq'	# other: 'freq&duty'


# Laser experiments 08/08/2018 1ms 383Hz focused laser as high as possible close to the side
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser18=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000044','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000045','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000046','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000048','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000049','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000050','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000053','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000054','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000055','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000056']
voltlaser18=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser18=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser18=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser18':laser18}))
laserROI18 = [ 'ff' ] * len(laser18)
collection_of_records['laser18'] = dict([])
collection_of_records['laser18']['path_files_laser'] = laser18
collection_of_records['laser18']['voltlaser'] = voltlaser18
collection_of_records['laser18']['freqlaser'] = freqlaser18
collection_of_records['laser18']['dutylaser'] = dutylaser18
collection_of_records['laser18']['laserROI'] = laserROI18
collection_of_records['laser18']['reference_clear'] = [vacuum2] * len(laser18)
collection_of_records['laser18']['power_interpolator'] = [power_interpolator1] * len(laser18)
collection_of_records['laser18']['focus_status'] = ['focused'] * len(laser18)
collection_of_records['laser18']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser18)
collection_of_records['laser18']['scan_type'] = 'power&freq'	# other: 'freq&duty'


# Laser experiments 08/08/2018 1ms 383Hz focused laser as close as possible to the aberration
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser19=['/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000057','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000059','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000060','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000061','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000062','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000063','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000065','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000066','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000067','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Aug08_2018/irvb_full-000069']
voltlaser19=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser19=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser19=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser19':laser19}))
laserROI19 = [ 'ff' ] * len(laser19)
collection_of_records['laser19'] = dict([])
collection_of_records['laser19']['path_files_laser'] = laser19
collection_of_records['laser19']['voltlaser'] = voltlaser19
collection_of_records['laser19']['freqlaser'] = freqlaser19
collection_of_records['laser19']['dutylaser'] = dutylaser19
collection_of_records['laser19']['laserROI'] = laserROI19
collection_of_records['laser19']['reference_clear'] = [vacuum2] * len(laser19)
collection_of_records['laser19']['power_interpolator'] = [power_interpolator1] * len(laser19)
collection_of_records['laser19']['focus_status'] = ['focused'] * len(laser19)
collection_of_records['laser19']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser19)
collection_of_records['laser19']['scan_type'] = 'power&freq'	# other: 'freq&duty'



# power of fully_defocused laser in 13/08/2018, from 02-08-2018 FF CCFE v6.ods
# I don't use the data from 23/08/2018 (that is more presice) because from images the laser seemed more dofucused here.
power_interpolator4 = interp1d([-1,0.0079,0.50,1],[0,0,0.76147*1e-3,0.76147*1e-3])

# Laser experiments 13/08/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum3=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000029']
vacuumframerate3=[383,994,994,994,994,994,994,994,383,994,994,994,994,994,994,994,383,383,383,994,994,994,383,994,994,994,994,994,994,994]
vacuuminttime3=[1,1,1,1,1,1,1,1,1,1,1,1,1,1]
vacuumROI3 = ['ff' , [[64,127],[0,319]] , [[96,159],[0,319]] , [[128,191],[0,319]] , [[32,127],[0,319]] , [[64,159],[0,319]] , [[96,191],[0,319]] , 'ff' , [[64,127],[0,319]] , \
			[[96,159],[0,319]] , [[128,191],[0,319]] , [[32,127],[0,319]] , [[64,159],[0,319]] , [[96,191],[0,319]] , [[128,224],[0,319]] , 'ff' , 'ff' , 'ff' , [[64,127],[0,319]] , \
			[[64,127],[0,319]] , [[64,127],[0,319]] , 'ff' , [[64,127],[0,319]] , [[96,159],[0,319]] , [[128,191],[0,319]] , [[32,127],[0,319]] , [[64,159],[0,319]] , [[96,191],[0,319]] , \
			[[128,224],[0,319]] ]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum3':vacuum3}))


# Laser experiments 13/08/2018 1ms 383Hz fully_defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser20=['/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000009']
voltlaser20=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser20=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser20=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser20':laser20}))
laserROI20 = [ 'ff' ] * len(laser20)
collection_of_records['laser20'] = dict([])
collection_of_records['laser20']['path_files_laser'] = laser20
collection_of_records['laser20']['voltlaser'] = voltlaser20
collection_of_records['laser20']['freqlaser'] = freqlaser20
collection_of_records['laser20']['dutylaser'] = dutylaser20
collection_of_records['laser20']['laserROI'] = laserROI20
collection_of_records['laser20']['reference_clear'] = [vacuum3] * len(laser20)
collection_of_records['laser20']['power_interpolator'] = [power_interpolator4] * len(laser20)
collection_of_records['laser20']['focus_status'] = ['fully_defocused'] * len(laser20)
collection_of_records['laser20']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser20)
collection_of_records['laser20']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq'


# Laser experiments 13/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) fully_defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser21=['/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug13_2018/irvb_full-000018']
voltlaser21=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser21=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser21=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser21':laser21}))
laserROI21 = [ [[64,127],[0,319]] ] * len(laser21)
collection_of_records['laser21'] = dict([])
collection_of_records['laser21']['path_files_laser'] = laser21
collection_of_records['laser21']['voltlaser'] = voltlaser21
collection_of_records['laser21']['freqlaser'] = freqlaser21
collection_of_records['laser21']['dutylaser'] = dutylaser21
collection_of_records['laser21']['laserROI'] = laserROI21
collection_of_records['laser21']['reference_clear'] = [vacuum3] * len(laser21)
collection_of_records['laser21']['power_interpolator'] = [power_interpolator4] * len(laser21)
collection_of_records['laser21']['focus_status'] = ['fully_defocused'] * len(laser21)
collection_of_records['laser21']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser21)
collection_of_records['laser21']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq'



# Laser experiments 20/08/2018 1ms+2ms+0.5ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum4=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000020','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000021','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000022','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000023','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000024','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000025','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000026','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000027','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000028','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000029','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000030','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000031','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000032','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000033','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000034','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000035','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000036','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000037','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000038','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000039','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000040','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000041','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000042','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000043','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000044','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug20_2018/irvb_sample-000045']
vacuumframerate4=[383,383,383,383,994,994,994,994,994,994,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,1976,383,383,383,383,1976,1976,1976,1976,1976,1976,1976,383,383,383,1976,1976,1976,1976,1976,383,383,383,383]
vacuuminttime4=[1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,2,2,2,2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,2,2,2,0.5,0.5,0.5,0.5,0.5,1,1,1,1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum4':vacuum4}))
vacuumROI4 = [ 'ff' , 'ff' , 'ff' , 'ff' , [[64,127],[0,319]] , [[64,127],[0,319]] , [[64,127],[0,319]] , [[64,127],[0,319]] , [[64,127],[0,319]] , \
			[[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , \
			[[64,127],[128,319]]  , [[64,127],[128,319]]  , [[64,127],[128,319]]  , [[64,127],[128,319]]  , [[64,127],[128,319]]  , [[64,127],[128,319]]  , 'ff' , 'ff' , 'ff' , 'ff' , \
			[[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , 'ff' , 'ff' , 'ff' , \
			[[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , [[64,127],[128,319]] , 'ff' , 'ff' , 'ff' , 'ff']


# Laser experiments 20/08/2018 1ms 383Hz focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser22=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000013']
voltlaser22=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser22=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser22=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser22':laser22}))
laserROI22 = [ 'ff' ] * len(laser22)
collection_of_records['laser22'] = dict([])
collection_of_records['laser22']['path_files_laser'] = laser22
collection_of_records['laser22']['voltlaser'] = voltlaser22
collection_of_records['laser22']['freqlaser'] = freqlaser22
collection_of_records['laser22']['dutylaser'] = dutylaser22
collection_of_records['laser22']['laserROI'] = laserROI22
collection_of_records['laser22']['reference_clear'] = [vacuum4] * len(laser22)
collection_of_records['laser22']['power_interpolator'] = [power_interpolator1] * len(laser22)
collection_of_records['laser22']['focus_status'] = ['focused'] * len(laser22)
collection_of_records['laser22']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser22)
collection_of_records['laser22']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 1ms 994Hz (width=320, height=64, xoffset=0, yoffset=64, invert (V flip) selected) focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser23=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000036']#,'/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000037']
voltlaser23=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser23=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90,120,150,180,210,240,270,300,330,360,390]
dutylaser23=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser23':laser23}))
laserROI23 = [ [[64,127],[0,319]] ] * len(laser23)
collection_of_records['laser23'] = dict([])
collection_of_records['laser23']['path_files_laser'] = laser23
collection_of_records['laser23']['voltlaser'] = voltlaser23
collection_of_records['laser23']['freqlaser'] = freqlaser23
collection_of_records['laser23']['dutylaser'] = dutylaser23
collection_of_records['laser23']['laserROI'] = laserROI23
collection_of_records['laser23']['reference_clear'] = [vacuum4] * len(laser23)
collection_of_records['laser23']['power_interpolator'] = [power_interpolator1] * len(laser23)
collection_of_records['laser23']['focus_status'] = ['focused'] * len(laser23)
collection_of_records['laser23']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser23)
collection_of_records['laser23']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser24=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000039','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000040','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000042','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000043','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000044','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000045','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000046','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000048','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000049','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000050','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000053','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000054','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000055','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000056','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000057','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000059','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000060','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000061','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000062']
voltlaser24=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser24=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90,140,190,240,290,340,390,440,490,540,590,640,690]
dutylaser24=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser24':laser24}))
laserROI24 = [ [[64,127],[128,319]] ] * len(laser24)
collection_of_records['laser24'] = dict([])
collection_of_records['laser24']['path_files_laser'] = laser24
collection_of_records['laser24']['voltlaser'] = voltlaser24
collection_of_records['laser24']['freqlaser'] = freqlaser24
collection_of_records['laser24']['dutylaser'] = dutylaser24
collection_of_records['laser24']['laserROI'] = laserROI24
collection_of_records['laser24']['reference_clear'] = [vacuum4] * len(laser24)
collection_of_records['laser24']['power_interpolator'] = [power_interpolator1] * len(laser24)
collection_of_records['laser24']['focus_status'] = ['focused'] * len(laser24)
collection_of_records['laser24']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser24)
collection_of_records['laser24']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 2ms 383Hz focused laser on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser25=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000093','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000094','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000095','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000096','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000097','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000098','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000099','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000100','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000101','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000102','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000103','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000104','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000105']
voltlaser25=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser25=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser25=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser25':laser25}))
laserROI25 = [ 'ff' ] * len(laser25)
collection_of_records['laser25'] = dict([])
collection_of_records['laser25']['path_files_laser'] = laser25
collection_of_records['laser25']['voltlaser'] = voltlaser25
collection_of_records['laser25']['freqlaser'] = freqlaser25
collection_of_records['laser25']['dutylaser'] = dutylaser25
collection_of_records['laser25']['laserROI'] = laserROI25
collection_of_records['laser25']['reference_clear'] = [vacuum4] * len(laser25)
collection_of_records['laser25']['power_interpolator'] = [power_interpolator1] * len(laser25)
collection_of_records['laser25']['focus_status'] = ['focused'] * len(laser25)
collection_of_records['laser25']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser25)
collection_of_records['laser25']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) focused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser26=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000063','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000065','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000066','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000067','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000069','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000070','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000071','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000072','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000073','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000074','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000075','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000076','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000077']
voltlaser26=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser26=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser26=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser26':laser26}))
laserROI26 = [ [[64,127],[128,319]] ] * len(laser26)
collection_of_records['laser26'] = dict([])
collection_of_records['laser26']['path_files_laser'] = laser26
collection_of_records['laser26']['voltlaser'] = voltlaser26
collection_of_records['laser26']['freqlaser'] = freqlaser26
collection_of_records['laser26']['dutylaser'] = dutylaser26
collection_of_records['laser26']['laserROI'] = laserROI26
collection_of_records['laser26']['reference_clear'] = [vacuum4] * len(laser26)
collection_of_records['laser26']['power_interpolator'] = [power_interpolator1] * len(laser26)
collection_of_records['laser26']['focus_status'] = ['focused'] * len(laser26)
collection_of_records['laser26']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser26)
collection_of_records['laser26']['scan_type'] = 'freq&duty'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) focused laser right in the pinhole at half power
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser27=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000078','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000079','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000080','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000081','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000082','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000083','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000084','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000085','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000086','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000087','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000088','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000089','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000090','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000091','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000092']
voltlaser27=[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
freqlaser27=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser27=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser27':laser27}))
laserROI27 = [ [[64,127],[128,319]] ] * len(laser27)
collection_of_records['laser27'] = dict([])
collection_of_records['laser27']['path_files_laser'] = laser27
collection_of_records['laser27']['voltlaser'] = voltlaser27
collection_of_records['laser27']['freqlaser'] = freqlaser27
collection_of_records['laser27']['dutylaser'] = dutylaser27
collection_of_records['laser27']['laserROI'] = laserROI27
collection_of_records['laser27']['reference_clear'] = [vacuum4] * len(laser27)
collection_of_records['laser27']['power_interpolator'] = [power_interpolator1] * len(laser27)
collection_of_records['laser27']['focus_status'] = ['focused'] * len(laser27)
collection_of_records['laser27']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser27)
collection_of_records['laser27']['scan_type'] = 'freq&duty'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) partially_defocused laser right in the pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser28=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000106','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000107','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000108','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000109','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000110','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000111','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000112','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000113','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000114','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000115','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000116','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000117','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000118','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000119','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000120']
voltlaser28=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser28=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser28=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser28':laser28}))
laserROI28 = [ [[64,127],[128,319]] ] * len(laser28)
collection_of_records['laser28'] = dict([])
collection_of_records['laser28']['path_files_laser'] = laser28
collection_of_records['laser28']['voltlaser'] = voltlaser28
collection_of_records['laser28']['freqlaser'] = freqlaser28
collection_of_records['laser28']['dutylaser'] = dutylaser28
collection_of_records['laser28']['laserROI'] = laserROI28
collection_of_records['laser28']['reference_clear'] = [vacuum4] * len(laser28)
collection_of_records['laser28']['power_interpolator'] = [power_interpolator1] * len(laser28)
collection_of_records['laser28']['focus_status'] = ['partially_defocused'] * len(laser28)
collection_of_records['laser28']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser28)
collection_of_records['laser28']['scan_type'] = 'freq&duty'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) partially_defocused laser right in the pinhole at half power
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser29=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000121','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000122','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000123','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000124','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000125','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000126','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000127','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000128','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000129','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000130','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000131','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000132','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000133','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000134','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000135']
voltlaser29=[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
freqlaser29=[10,25,50,75,100,10,25,50,75,100,10,25,50,75,100]
dutylaser29=[0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser29':laser29}))
laserROI29 = [ [[64,127],[128,319]] ] * len(laser29)
collection_of_records['laser29'] = dict([])
collection_of_records['laser29']['path_files_laser'] = laser29
collection_of_records['laser29']['voltlaser'] = voltlaser29
collection_of_records['laser29']['freqlaser'] = freqlaser29
collection_of_records['laser29']['dutylaser'] = dutylaser29
collection_of_records['laser29']['laserROI'] = laserROI29
collection_of_records['laser29']['reference_clear'] = [vacuum4] * len(laser29)
collection_of_records['laser29']['power_interpolator'] = [power_interpolator1] * len(laser29)
collection_of_records['laser29']['focus_status'] = ['partially_defocused'] * len(laser29)
collection_of_records['laser29']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser29)
collection_of_records['laser29']['scan_type'] = 'freq&duty'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 2ms 383Hz fully_defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser30=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000136','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000137','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000138','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000139','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000140','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000141','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000142','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000143','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000144']
voltlaser30=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser30=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser30=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser30':laser30}))
laserROI30 = [ 'ff' ] * len(laser30)
collection_of_records['laser30'] = dict([])
collection_of_records['laser30']['path_files_laser'] = laser30
collection_of_records['laser30']['voltlaser'] = voltlaser30
collection_of_records['laser30']['freqlaser'] = freqlaser30
collection_of_records['laser30']['dutylaser'] = dutylaser30
collection_of_records['laser30']['laserROI'] = laserROI30
collection_of_records['laser30']['reference_clear'] = [vacuum4] * len(laser30)
collection_of_records['laser30']['power_interpolator'] = [power_interpolator4] * len(laser30)
collection_of_records['laser30']['focus_status'] = ['fully_defocused'] * len(laser30)
collection_of_records['laser30']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser30)
collection_of_records['laser30']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 0.5ms 1976Hz (width=192, height=64, xoffset=128, yoffset=64, invert (V flip) selected) fully_defocused laser straight on pinhole
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser31=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000145','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000146','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000147','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000148','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000149','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000150','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000151','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000152','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000153','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000154','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000155','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000156','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000157','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000158','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000159','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000160','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000161','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000162','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000163','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000164']
voltlaser31=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser31=[0.2,0.5,1,3,5,10,30,60,90,140,190,240,290,340,390,440,409,540,590,640]
dutylaser31=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser31':laser31}))
laserROI31 = [ [[64,127],[128,319]] ] * len(laser31)
collection_of_records['laser31'] = dict([])
collection_of_records['laser31']['path_files_laser'] = laser31
collection_of_records['laser31']['voltlaser'] = voltlaser31
collection_of_records['laser31']['freqlaser'] = freqlaser31
collection_of_records['laser31']['dutylaser'] = dutylaser31
collection_of_records['laser31']['laserROI'] = laserROI31
collection_of_records['laser31']['reference_clear'] = [vacuum4] * len(laser31)
collection_of_records['laser31']['power_interpolator'] = [power_interpolator4] * len(laser31)
collection_of_records['laser31']['focus_status'] = ['fully_defocused'] * len(laser31)
collection_of_records['laser31']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser31)
collection_of_records['laser31']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 20/08/2018 1ms 383Hz focused laser as high and left as possible
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser32=['/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000165','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000166','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000167','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000168','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000169','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000170','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000171','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000172','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000173','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000174','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000175','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000176','/home/ffederic/work/irvb/laser/Aug20_2018/irvb_full-000177']
voltlaser32=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser32=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser32=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser32':laser32}))
laserROI32 = [ 'ff' ] * len(laser32)
collection_of_records['laser32'] = dict([])
collection_of_records['laser32']['path_files_laser'] = laser32
collection_of_records['laser32']['voltlaser'] = voltlaser32
collection_of_records['laser32']['freqlaser'] = freqlaser32
collection_of_records['laser32']['dutylaser'] = dutylaser32
collection_of_records['laser32']['laserROI'] = laserROI32
collection_of_records['laser32']['reference_clear'] = [vacuum4] * len(laser32)
collection_of_records['laser32']['power_interpolator'] = [power_interpolator1] * len(laser32)
collection_of_records['laser32']['focus_status'] = ['focused'] * len(laser32)
collection_of_records['laser32']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser32)
collection_of_records['laser32']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'



# Laser experiments 23/08/2018 1ms+0.5ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum5=['/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000006','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000008','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000009','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000010','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000011','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000012','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000013','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000014','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000015','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000016','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000017','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000018','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000019','/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018/irvb_sample-000020']
vacuumframerate5=[383,383,383,994,994,994,994,994,994,994,1976,1976,1976,1976,1976,1976]
vacuuminttime5=[1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1]
vacuumROI5 = ['ff' , [[96,159],[0,319]] , [[64,127],[0,319]] , [[96,159],[0,319]] , [[64,127],[0,319]] , [[128,191],[0,319]] , [[128,223],[0,319]] , [[96,191],[0,319]] , \
 			[[64,159],[0,319]] , [[32,127],[0,319]] , [[128,191],[64,255]] , [[96,159],[64,255]] , [[64,127],[64,255]] , [[128,191],[128,319]] , [[96,159],[128,319]] , \
			[[64,127],[128,319]], 'ff', 'ff', 'ff', 'ff' ]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum5':vacuum5}))


# power of fully_defocused laser in 23/08/2018, from 02-08-2018 FF CCFE v5.ods
power_interpolator2 = interp1d([-1,0.0079,0.5,1],[0,0,1.0511*1e-3,1.0511*1e-3])

# Laser experiments 23/08/2018 1ms 383Hz fully_defocused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser33=['/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Aug23_2018/irvb_full-000012']
voltlaser33=[0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser33=[0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser33=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser33':laser33}))
laserROI33 = [ 'ff' ] * len(laser33)
collection_of_records['laser33'] = dict([])
collection_of_records['laser33']['path_files_laser'] = laser33
collection_of_records['laser33']['voltlaser'] = voltlaser33
collection_of_records['laser33']['freqlaser'] = freqlaser33
collection_of_records['laser33']['dutylaser'] = dutylaser33
collection_of_records['laser33']['laserROI'] = laserROI33
collection_of_records['laser33']['reference_clear'] = [vacuum5] * len(laser33)
collection_of_records['laser33']['power_interpolator'] = [power_interpolator2] * len(laser33)
collection_of_records['laser33']['focus_status'] = ['fully_defocused'] * len(laser33)
collection_of_records['laser33']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laser33)
collection_of_records['laser33']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 25/10/2018 1ms+0.5ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability
vacuum6=['/home/ffederic/work/irvb/vacuum_chamber_testing/Oct25_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct25_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct25_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct25_2018/irvb_sample-000004','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000037','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000044','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000049','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000050','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000056','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000062','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000066','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000067','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000073','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000079','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000085','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000091','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000097']
vacuumframerate6=[383,383,383,383,383,383,383,994,994,994,1976,1976,1976,994,994,994,994,1976,1976,1976,1976,1976,1976]	# [Hz]
vacuuminttime6=[1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5]	# [ms]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum6':vacuum6}))
vacuumROI6 = [ 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , 'ff' , [[82,173],[0,319]] , [[82,173],[0,319]] , [[82,173],[0,319]] , [[80,175],[160,255]] , \
			[[80,175],[160,255]] , [[80,175],[160,255]] , [[82,173],[0,319]] , [[82,173],[0,319]] , [[82,173],[0,319]] , [[82,173],[0,319]] , [[80,175],[160,255]] , \
			[[80,175],[160,255]] , [[80,175],[160,255]] , [[80,175],[160,255]] , [[80,175],[160,255]] , [[80,175],[160,255]] ]


# Laser experiments 25/10/2018 1ms 383Hz focused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser34=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000013']
voltlaser34=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]	# [V]
freqlaser34=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]	# [Hz]
dutylaser34=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]	# [#]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser34':laser34}))
laserROI34 = [ 'ff' ] * len(laser34)
collection_of_records['laser34'] = dict([])
collection_of_records['laser34']['path_files_laser'] = laser34
collection_of_records['laser34']['voltlaser'] = voltlaser34
collection_of_records['laser34']['freqlaser'] = freqlaser34
collection_of_records['laser34']['dutylaser'] = dutylaser34
collection_of_records['laser34']['laserROI'] = laserROI34
collection_of_records['laser34']['reference_clear'] = [vacuum6] * len(laser34)
collection_of_records['laser34']['power_interpolator'] = [power_interpolator1] * len(laser34)
collection_of_records['laser34']['focus_status'] = ['focused'] * len(laser34)
collection_of_records['laser34']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser34)
collection_of_records['laser34']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 25/10/2018 1ms 383Hz fully_defocused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser35=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000016','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000024']
voltlaser35=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser35=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser35=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser35':laser35}))
laserROI35 = [ 'ff' ] * len(laser35)
collection_of_records['laser35'] = dict([])
collection_of_records['laser35']['path_files_laser'] = laser35
collection_of_records['laser35']['voltlaser'] = voltlaser35
collection_of_records['laser35']['freqlaser'] = freqlaser35
collection_of_records['laser35']['dutylaser'] = dutylaser35
collection_of_records['laser35']['laserROI'] = laserROI35
collection_of_records['laser35']['reference_clear'] = [vacuum6] * len(laser35)
collection_of_records['laser35']['power_interpolator'] = [power_interpolator2] * len(laser35)
collection_of_records['laser35']['focus_status'] = ['fully_defocused'] * len(laser35)
collection_of_records['laser35']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser35)
collection_of_records['laser35']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 25/10/2018 1ms 994Hz  (width=320, height=92, xoffset=0, yoffset=82, invert (V flip) selected) fully_defocused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser36=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000033','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000036']
voltlaser36=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser36=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser36=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser36':laser36}))
laserROI36 = [ [[82,173],[0,319]] ] * len(laser36)
collection_of_records['laser36'] = dict([])
collection_of_records['laser36']['path_files_laser'] = laser36
collection_of_records['laser36']['voltlaser'] = voltlaser36
collection_of_records['laser36']['freqlaser'] = freqlaser36
collection_of_records['laser36']['dutylaser'] = dutylaser36
collection_of_records['laser36']['laserROI'] = laserROI36
collection_of_records['laser36']['reference_clear'] = [vacuum6] * len(laser36)
collection_of_records['laser36']['power_interpolator'] = [power_interpolator2] * len(laser36)
collection_of_records['laser36']['focus_status'] = ['fully_defocused'] * len(laser36)
collection_of_records['laser36']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser36)
collection_of_records['laser36']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 25/10/2018 0.5ms 1976Hz  (width=96, height=96, xoffset=160, yoffset=80, invert (V flip) selected) fully_defocused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser37=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000039','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000040','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000042','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000043','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000045','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000046','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000048']
voltlaser37=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser37=[0.2,0.5,1,3,5,10,30,60,90]
dutylaser37=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser37':laser37}))
laserROI37 = [ [[80,175],[160,255]] ] * len(laser37)
collection_of_records['laser37'] = dict([])
collection_of_records['laser37']['path_files_laser'] = laser37
collection_of_records['laser37']['voltlaser'] = voltlaser37
collection_of_records['laser37']['freqlaser'] = freqlaser37
collection_of_records['laser37']['dutylaser'] = dutylaser37
collection_of_records['laser37']['laserROI'] = laserROI37
collection_of_records['laser37']['reference_clear'] = [vacuum6] * len(laser37)
collection_of_records['laser37']['power_interpolator'] = [power_interpolator2] * len(laser37)
collection_of_records['laser37']['focus_status'] = ['fully_defocused'] * len(laser37)
collection_of_records['laser37']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser37)
collection_of_records['laser37']['scan_type'] = 'freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 25/10/2018 1ms 994Hz  (width=320, height=92, xoffset=0, yoffset=82, invert (V flip) selected) focused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser38=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000053','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000054','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000055','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000057','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000059','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000060','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000061','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000063','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000065']
voltlaser38=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser38=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser38=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser38':laser38}))
laserROI38 = [ [[82,173],[0,319]] ] * len(laser38)
collection_of_records['laser38'] = dict([])
collection_of_records['laser38']['path_files_laser'] = laser38
collection_of_records['laser38']['voltlaser'] = voltlaser38
collection_of_records['laser38']['freqlaser'] = freqlaser38
collection_of_records['laser38']['dutylaser'] = dutylaser38
collection_of_records['laser38']['laserROI'] = laserROI38
collection_of_records['laser38']['reference_clear'] = [vacuum6] * len(laser38)
collection_of_records['laser38']['power_interpolator'] = [power_interpolator1] * len(laser38)
collection_of_records['laser38']['focus_status'] = ['focused'] * len(laser38)
collection_of_records['laser38']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser38)
collection_of_records['laser38']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 25/10/2018 0.5ms 1976Hz  (width=96, height=96, xoffset=160, yoffset=80, invert (V flip) selected) focused laser
# NOTE THAT LASER VOLTAGE / POWER CORRELATION IS DIFFERENT FROM 03/2018 ONE ! ! !
# Use     reflaserpower1    and   reflaserfvoltage1
laser39=['/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000069','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000070','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000071','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000072','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000074','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000075','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000076','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000077','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000078','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000080','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000081','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000082','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000083','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000084','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000086','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000087','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000088','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000089','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000090','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000092','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000093','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000094','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000095','/home/ffederic/work/irvb/laser/Oct25_2018/irvb_full-000096']
voltlaser39=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser39=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90,150,200,250,300,350,400,450,500,550,600,700,800]
dutylaser39=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser39':laser39}))
laserROI39 = [ [[80,175],[160,255]] ] * len(laser39)
collection_of_records['laser39'] = dict([])
collection_of_records['laser39']['path_files_laser'] = laser39
collection_of_records['laser39']['voltlaser'] = voltlaser39
collection_of_records['laser39']['freqlaser'] = freqlaser39
collection_of_records['laser39']['dutylaser'] = dutylaser39
collection_of_records['laser39']['laserROI'] = laserROI39
collection_of_records['laser39']['reference_clear'] = [vacuum6] * len(laser39)
collection_of_records['laser39']['power_interpolator'] = [power_interpolator1] * len(laser39)
collection_of_records['laser39']['focus_status'] = ['focused'] * len(laser39)
collection_of_records['laser39']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser39)
collection_of_records['laser39']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'




# Voltage / power correlation of red laser from PPPL through the mirror 20/11/2018
reflaserpower2=[3.605,3.605,0,0]
reflaserpower2=np.multiply(0.001,np.flip(reflaserpower2,0)) # [mW]
reflaserfvoltage2=[10,0.5101,0.0107,0]
reflaserfvoltage2=np.flip(reflaserfvoltage2,0) # [V]

power_interpolator3 = interp1d([0,0.0079,0.503,10],[0,0,3.648*1e-3,3.648*1e-3])


# Laser experiments 19/11/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation.
# vacuum7=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000024','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000069','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000075','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000081','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000085','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000086','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000092','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000098','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000102','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000103','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000109','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000115','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000119']
vacuum7=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000001','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000007','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000013','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000017','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000018','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000030','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000034','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000035','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000041','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000047','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000051','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000052','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000058','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000064','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000068','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000069','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000075','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000081','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000085','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000086','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000092','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000098','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000102','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000103','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000109','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000115','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000119']
vacuumframerate7=[383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383]
vacuuminttime7=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# vacuumtime7=[655, 658, 660, 662, 687, 690, 692, 694, 701, 703, 706, 707, 715,718, 723, 724, 735, 738, 740, 742, 751, 754, 759, 760, 773, 776,777, 779] #[min]
vacuumtime7=[655, 658, 660, 662, 687, 692, 694, 701, 703, 706, 707, 715,718, 723, 724, 735, 738, 740, 742, 751, 754, 759, 760, 773, 776,777, 779] #[min]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum7':vacuum7}))
vacuumROI7 = [ 'ff' ] * len(vacuum7)


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H167, V121 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser41=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000002','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000003','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000004','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000005','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000006','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000008','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000009','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000010','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000011','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000012','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000014','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000015','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000016']
voltlaser41=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser41=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser41=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser41':laser41}))
laserROI41 = [ 'ff' ] * len(laser41)
collection_of_records['laser41'] = dict([])
collection_of_records['laser41']['path_files_laser'] = laser41
collection_of_records['laser41']['voltlaser'] = voltlaser41
collection_of_records['laser41']['freqlaser'] = freqlaser41
collection_of_records['laser41']['dutylaser'] = dutylaser41
collection_of_records['laser41']['laserROI'] = laserROI41
collection_of_records['laser41']['reference_clear'] = [vacuum7] * len(laser41)
collection_of_records['laser41']['power_interpolator'] = [power_interpolator3] * len(laser41)
collection_of_records['laser41']['focus_status'] = ['focused'] * len(laser41)
collection_of_records['laser41']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser41)
collection_of_records['laser41']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H167, V131 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser42=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000019','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000020','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000021','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000022','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000023','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000025','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000026','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000027','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000028','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000029','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000031','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000032','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000033']
voltlaser42=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser42=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser42=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser42':laser42}))
laserROI42 = [ 'ff' ] * len(laser42)
collection_of_records['laser42'] = dict([])
collection_of_records['laser42']['path_files_laser'] = laser42
collection_of_records['laser42']['voltlaser'] = voltlaser42
collection_of_records['laser42']['freqlaser'] = freqlaser42
collection_of_records['laser42']['dutylaser'] = dutylaser42
collection_of_records['laser42']['laserROI'] = laserROI42
collection_of_records['laser42']['reference_clear'] = [vacuum7] * len(laser42)
collection_of_records['laser42']['power_interpolator'] = [power_interpolator3] * len(laser42)
collection_of_records['laser42']['focus_status'] = ['focused'] * len(laser42)
collection_of_records['laser42']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser42)
collection_of_records['laser42']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H167, V141 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser43=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000036','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000037','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000038','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000039','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000040','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000042','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000043','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000044','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000045','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000046','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000048','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000049','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000050']
voltlaser43=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser43=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser43=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser43':laser43}))
laserROI43 = [ 'ff' ] * len(laser43)
collection_of_records['laser43'] = dict([])
collection_of_records['laser43']['path_files_laser'] = laser43
collection_of_records['laser43']['voltlaser'] = voltlaser43
collection_of_records['laser43']['freqlaser'] = freqlaser43
collection_of_records['laser43']['dutylaser'] = dutylaser43
collection_of_records['laser43']['laserROI'] = laserROI43
collection_of_records['laser43']['reference_clear'] = [vacuum7] * len(laser43)
collection_of_records['laser43']['power_interpolator'] = [power_interpolator3] * len(laser43)
collection_of_records['laser43']['focus_status'] = ['focused'] * len(laser43)
collection_of_records['laser43']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser43)
collection_of_records['laser43']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H167, V151 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser44=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000053','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000054','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000055','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000056','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000057','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000059','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000060','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000061','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000062','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000063','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000065','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000066','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000067']
voltlaser44=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser44=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser44=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser44':laser44}))
laserROI44 = [ 'ff' ] * len(laser44)
collection_of_records['laser44'] = dict([])
collection_of_records['laser44']['path_files_laser'] = laser44
collection_of_records['laser44']['voltlaser'] = voltlaser44
collection_of_records['laser44']['freqlaser'] = freqlaser44
collection_of_records['laser44']['dutylaser'] = dutylaser44
collection_of_records['laser44']['laserROI'] = laserROI44
collection_of_records['laser44']['reference_clear'] = [vacuum7] * len(laser44)
collection_of_records['laser44']['power_interpolator'] = [power_interpolator3] * len(laser44)
collection_of_records['laser44']['focus_status'] = ['focused'] * len(laser44)
collection_of_records['laser44']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser44)
collection_of_records['laser44']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H167, V161 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser45=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000070','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000071','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000072','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000073','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000074','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000076','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000077','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000078','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000079','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000080','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000082','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000083','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000084']
voltlaser45=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser45=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser45=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser45':laser45}))
laserROI45 = [ 'ff' ] * len(laser45)
collection_of_records['laser45'] = dict([])
collection_of_records['laser45']['path_files_laser'] = laser45
collection_of_records['laser45']['voltlaser'] = voltlaser45
collection_of_records['laser45']['freqlaser'] = freqlaser45
collection_of_records['laser45']['dutylaser'] = dutylaser45
collection_of_records['laser45']['laserROI'] = laserROI45
collection_of_records['laser45']['reference_clear'] = [vacuum7] * len(laser45)
collection_of_records['laser45']['power_interpolator'] = [power_interpolator3] * len(laser45)
collection_of_records['laser45']['focus_status'] = ['focused'] * len(laser45)
collection_of_records['laser45']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser45)
collection_of_records['laser45']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H167, V171 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser46=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000087','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000088','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000089','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000090','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000091','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000093','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000094','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000095','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000096','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000097','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000099','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000100','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000101']
voltlaser46=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser46=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser46=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser46':laser46}))
laserROI46 = [ 'ff' ] * len(laser46)
collection_of_records['laser46'] = dict([])
collection_of_records['laser46']['path_files_laser'] = laser46
collection_of_records['laser46']['voltlaser'] = voltlaser46
collection_of_records['laser46']['freqlaser'] = freqlaser46
collection_of_records['laser46']['dutylaser'] = dutylaser46
collection_of_records['laser46']['laserROI'] = laserROI46
collection_of_records['laser46']['reference_clear'] = [vacuum7] * len(laser46)
collection_of_records['laser46']['power_interpolator'] = [power_interpolator3] * len(laser46)
collection_of_records['laser46']['focus_status'] = ['focused'] * len(laser46)
collection_of_records['laser46']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser46)
collection_of_records['laser46']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Laser experiments 19/11/2018 1ms 383Hz focused laser, center of the laser spot located at H187, V121 on the full frame
# NOTE THAT LASER VOLTAGE / POWER CORRELATION MEASURED 20/11/2018
# Use     reflaserpower2    and   reflaserfvoltage2
laser47=['/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000104','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000105','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000106','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000107','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000108','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000110','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000111','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000112','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000113','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000114','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000116','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000117','/home/ffederic/work/irvb/laser/Nov19_2018/irvb_full-000118']
voltlaser47=[0.05,0.1,0.25,0.35,0.5,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
freqlaser47=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,1,3,10,30,60,90]
dutylaser47=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laser47':laser47}))
laserROI47 = [ 'ff' ] * len(laser47)
collection_of_records['laser47'] = dict([])
collection_of_records['laser47']['path_files_laser'] = laser47
collection_of_records['laser47']['voltlaser'] = voltlaser47
collection_of_records['laser47']['freqlaser'] = freqlaser47
collection_of_records['laser47']['dutylaser'] = dutylaser47
collection_of_records['laser47']['laserROI'] = laserROI47
collection_of_records['laser47']['reference_clear'] = [vacuum7] * len(laser47)
collection_of_records['laser47']['power_interpolator'] = [power_interpolator3] * len(laser47)
collection_of_records['laser47']['focus_status'] = ['focused'] * len(laser47)
collection_of_records['laser47']['foil_position_dict'] = [dict([('angle',-2),('foilcenter',[163,130]),('foilhorizwpixel',240)])] * len(laser47)
collection_of_records['laser47']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'


# Experiment 20/11/2018 1ms samples with no power on foil and IR camera shielded from any possible stray radiation to test background stability.
# Starting to record from camera switch on to the point that contrast stabilize
vacuum8=['/home/ffederic/work/irvb/vacuum_chamber_testing/Nov20_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Nov20_2018/irvb_sample-000003','/home/ffederic/work/irvb/vacuum_chamber_testing/Nov20_2018/irvb_sample-000004','/home/ffederic/work/irvb/vacuum_chamber_testing/Nov20_2018/irvb_sample-000005','/home/ffederic/work/irvb/vacuum_chamber_testing/Nov20_2018/irvb_sample-000007','/home/ffederic/work/irvb/vacuum_chamber_testing/Nov20_2018/irvb_sample-000009']
vacuumframerate8=[383,383,383,383,383,383]
vacuuminttime8=[1,1,1,1,1,1]
# vacuumtime8=[820, 830, 840, 850, 883, 955] #[min]
vacuumtime8=[820, 833, 844, 853, 885, 957] #[min]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum8':vacuum8}))


# Experiment 25/08/2022 2ms low frequency with the BB source getting closer to the camera to see what is the effect on counts. TBB=35degC
files50=['/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000001','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000002','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000003','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000004','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000005','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000006','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000007','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000008','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000009','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000010','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000011','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000012','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000013','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000014']
distance50=[65.7,60.7,55.7,50.7,45.7,40.7,35.7,30.7,25.7,20.7,15.7,10.7,5.7,3.7] #[cm] camera to source distance
collection_of_records['files50'] = dict([])
collection_of_records['files50']['path_files_laser'] = files50
collection_of_records['files50']['distance'] = distance50

# Experiment 25/08/2022 2ms low frequency with the BB source getting closer to the camera to see what is the effect on counts. TBB=50degC.
# WARNING! SATURATED!!
files51=['/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000015','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000016','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000017','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000018','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000019','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000020','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000021','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000022','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000023','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000024','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000025','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000026','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000027','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000028']
distance51=[65.7,60.7,55.7,50.7,45.7,40.7,35.7,30.7,25.7,20.7,15.7,10.7,5.7,3.7] #[cm] camera to source distance
collection_of_records['files51'] = dict([])
collection_of_records['files51']['path_files_laser'] = files51
collection_of_records['files51']['distance'] = distance51

# Experiment 25/08/2022 2ms low frequency with the NUC PLATE getting closer to the camera to see what is the effect on counts. TNUC ~ 27degC.
files52=['/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000029','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000030','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000031','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000032','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000033','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000034','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000035','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000036','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000037','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000038','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000039','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000040','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000041','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000042']
distance52=[65.7,60.7,55.7,50.7,45.7,40.7,35.7,30.7,25.7,20.7,15.7,10.7,5.7,4.7] #[cm] camera to source distance
collection_of_records['files52'] = dict([])
collection_of_records['files52']['path_files_laser'] = files52
collection_of_records['files52']['distance'] = distance52

# Experiment 25/08/2022 2ms low frequency with the BB source getting closer to the camera to see what is the effect on counts. TBB ~ 26.4degC.
# WARNING! SATURATED!!
files53=['/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000043','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000044','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000045','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000046','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000047','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000048','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000049','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000050','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000051','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000052','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000053','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000054','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000055','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000056','/home/ffederic/work/irvb/flatfield/Aug25_2022/flat_field-000057']
distance53=[65.7,60.7,55.7,50.7,45.7,40.7,35.7,30.7,25.7,20.7,15.7,9.7,10.7,5.7,3.7] #[cm] camera to source distance
collection_of_records['files53'] = dict([])
collection_of_records['files53']['path_files_laser'] = files53
collection_of_records['files53']['distance'] = distance53



# Experiment 26/10/2018 1ms samples with one of the other IR cameras from Andrew Thorton to see if the oscillation is still there
vacuum9=['/home/ffederic/work/irvb/vacuum_chamber_testing/Oct26_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct26_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct26_2018/irvb_sample-000003']
vacuumframerate9=[383,383,383]
vacuuminttime9=[1,1,1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum9':vacuum9}))


#24/10/2018 Record corrisponding to depressurizing to open the vacuum chamber and repressurizing
vacuum10=['/home/ffederic/work/irvb/vacuum_chamber_testing/Oct24_2018/irvb_sample-000001','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct24_2018/irvb_sample-000002','/home/ffederic/work/irvb/vacuum_chamber_testing/Oct24_2018/irvb_sample-000003']
vacuumframerate10=[50,50,50]
vacuuminttime10=[1,1,1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuum10':vacuum10}))



# Data given me by Glenn Wurden regarding the "calibration" of his coated windows

# Sapphire disk in air 2micron Gold
vacuumG01=['/home/ffederic/work/irvb/laser/Glen_Wurden_HTPD2024/Calibration from Glenn/test92']
vacuumframerateG01=[50]
vacuuminttimeG01=[1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuumG01':vacuumG01}))

power_interpolatorG01 = interp1d([0,0.,0.5,10],[0,0,0.017,0.017])
laserG01=['/home/ffederic/work/irvb/laser/Glen_Wurden_HTPD2024/Calibration from Glenn/Rec-0092']
voltlaserG01=[1]
freqlaserG01=[0.007788]
dutylaserG01=[0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laserG01':laserG01}))
laserROIG01 = [ 'ff' ] * len(laserG01)
collection_of_records['laserG01'] = dict([])
collection_of_records['laserG01']['path_files_laser'] = laserG01
collection_of_records['laserG01']['voltlaser'] = voltlaserG01
collection_of_records['laserG01']['freqlaser'] = freqlaserG01
collection_of_records['laserG01']['dutylaser'] = dutylaserG01
collection_of_records['laserG01']['laserROI'] = laserROIG01
collection_of_records['laserG01']['reference_clear'] = [vacuumG01] * len(laserG01)
collection_of_records['laserG01']['power_interpolator'] = [power_interpolatorG01] * len(laserG01)
collection_of_records['laserG01']['focus_status'] = ['focused'] * len(laserG01)
collection_of_records['laserG01']['foil_position_dict'] = [dict([('angle',0),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laserG01)
collection_of_records['laserG01']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'
collection_of_records['laserG01']['absorber_material'] = 'Au'	# name of the element from periodic table


# Sapphire disk in vacuum 2micron Gold
vacuumG02=['/home/ffederic/work/irvb/laser/Glen_Wurden_HTPD2024/Calibration from Glenn/test94']
vacuumframerateG02=[50]
vacuuminttimeG02=[1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuumG02':vacuumG02}))

power_interpolatorG02 = interp1d([0,0.,0.5,10],[0,0,0.010,0.010])
laserG02=['/home/ffederic/work/irvb/laser/Glen_Wurden_HTPD2024/Calibration from Glenn/Rec-0094']
voltlaserG02=[1]
freqlaserG02=[0.05]
dutylaserG02=[0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laserG02':laserG02}))
laserROIG02 = [ 'ff' ] * len(laserG02)
collection_of_records['laserG02'] = dict([])
collection_of_records['laserG02']['path_files_laser'] = laserG02
collection_of_records['laserG02']['voltlaser'] = voltlaserG02
collection_of_records['laserG02']['freqlaser'] = freqlaserG02
collection_of_records['laserG02']['dutylaser'] = dutylaserG02
collection_of_records['laserG02']['laserROI'] = laserROIG02
collection_of_records['laserG02']['reference_clear'] = [vacuumG02] * len(laserG02)
collection_of_records['laserG02']['power_interpolator'] = [power_interpolatorG02] * len(laserG02)
collection_of_records['laserG02']['focus_status'] = ['focused'] * len(laserG02)
collection_of_records['laserG02']['foil_position_dict'] = [dict([('angle',0),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laserG02)
collection_of_records['laserG02']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'
collection_of_records['laserG02']['absorber_material'] = 'Au'	# name of the element from periodic table


# 5micron no sapphire gold foil in vacuum
vacuumG03=['/home/ffederic/work/irvb/laser/Glen_Wurden_HTPD2024/Calibration from Glenn/test120']
vacuumframerateG03=[50]
vacuuminttimeG03=[1]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuumG03':vacuumG03}))

power_interpolatorG03 = interp1d([0,0.,0.5,10],[0,0,0.010,0.010])
laserG03=['/home/ffederic/work/irvb/laser/Glen_Wurden_HTPD2024/Calibration from Glenn/Rec-0120']
voltlaserG03=[1]
freqlaserG03=[0.02]
dutylaserG03=[0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laserG03':laserG03}))
laserROIG03 = [ 'ff' ] * len(laserG03)
collection_of_records['laserG03'] = dict([])
collection_of_records['laserG03']['path_files_laser'] = laserG03
collection_of_records['laserG03']['voltlaser'] = voltlaserG03
collection_of_records['laserG03']['freqlaser'] = freqlaserG03
collection_of_records['laserG03']['dutylaser'] = dutylaserG03
collection_of_records['laserG03']['laserROI'] = laserROIG03
collection_of_records['laserG03']['reference_clear'] = [vacuumG03] * len(laserG03)
collection_of_records['laserG03']['power_interpolator'] = [power_interpolatorG03] * len(laserG03)
collection_of_records['laserG03']['focus_status'] = ['focused'] * len(laserG03)
collection_of_records['laserG03']['foil_position_dict'] = [dict([('angle',0),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laserG03)
collection_of_records['laserG03']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'
collection_of_records['laserG03']['absorber_material'] = 'Au'	# name of the element from periodic table



# new data from the foil olympics in September 2024 in Greifswald

# no laser reference for G12
vacuumG12=['Rec-000059-001_06_34_24_097','Rec-000071-001_06_47_15_712','Rec-000079-001_06_55_58_114','Rec-000085-001_07_03_35_858']
for i in range(len(vacuumG12)):
	vacuumG12[i] = vacuumG12[i] + '/home/ffederic/work/irvb/laser/Sept17_2024/'
vacuumframerateG12=[383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383]
vacuuminttimeG12=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
vacuumtimeG12=[900,913,922,929] #[min]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuumG12':vacuumG12}))
vacuumROIG12 = [ 'ff' ] * len(vacuumG12)

# my 350C/315Pt/1000Ti/325Pt/350C CVD foil made summer 2024
power_interpolatorG12 = interp1d([0,0.,2.5,10],[0,0,0.043,0.043])
laserG12=['Rec-000060-001_06_36_06_485','Rec-000061-001_06_37_27_644','Rec-000062-001_06_38_27_675','Rec-000063-001_06_39_09_026','Rec-000064-001_06_39_49_437','Rec-000065-001_06_40_50_709','Rec-000066-001_06_42_07_374','Rec-000067-001_06_43_08_847','Rec-000068-001_06_43_50_064','Rec-000069-001_06_44_47_613','Rec-000070-001_06_46_30_538','Rec-000072-001_06_48_07_694','Rec-000073-001_06_48_50_319','Rec-000074-001_06_50_31_432','Rec-000075-001_06_51_31_128','Rec-000076-001_06_52_47_894','Rec-000077-001_06_53_48_595','Rec-000078-001_06_55_07_507','Rec-000080-001_06_56_47_447','Rec-000081-001_06_57_47_947','Rec-000082-001_06_58_47_240','Rec-000083-001_07_01_27_915','Rec-000084-001_07_02_46_324']
for i in range(len(laserG12)):
	laserG12[i] = '/home/ffederic/work/irvb/laser/Sept17_2024/' + laserG12[i]
voltlaserG12=[1]
freqlaserG12=[0.02]
dutylaserG12=[0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laserG12':laserG12}))
laserROIG12 = [ 'ff' ] * len(laserG12)
collection_of_records['laserG12'] = dict([])
collection_of_records['laserG12']['path_files_laser'] = laserG12
collection_of_records['laserG12']['voltlaser'] = voltlaserG12
collection_of_records['laserG12']['freqlaser'] = freqlaserG12
collection_of_records['laserG12']['dutylaser'] = dutylaserG12
collection_of_records['laserG12']['laserROI'] = laserROIG12
collection_of_records['laserG12']['reference_clear'] = [vacuumG12] * len(laserG12)
collection_of_records['laserG12']['power_interpolator'] = [power_interpolatorG12] * len(laserG12)
collection_of_records['laserG12']['focus_status'] = ['focused'] * len(laserG12)
collection_of_records['laserG12']['foil_position_dict'] = [dict([('angle',0),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laserG12)
collection_of_records['laserG12']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'
collection_of_records['laserG12']['absorber_material'] = 'Au'	# name of the element from periodic table



# no laser reference for G13
vacuumG13=['Rec-000001-001_00_52_23_745','Rec-000010-001_01_03_58_795','Rec-000018-001_01_11_20_509','Rec-000028-001_01_23_11_422','Rec-000035-001_01_30_09_424']
for i in range(len(vacuumG13)):
	vacuumG13[i] = vacuumG13[i] + '/home/ffederic/work/irvb/laser/Sept18_2024/'
vacuumframerateG13=[383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383,383]
vacuuminttimeG13=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
vacuumtimeG13=[900,913,922,929] #[min]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'vacuumG13':vacuumG13}))
vacuumROIG13 = [ 'ff' ] * len(vacuumG13)

# Japanese foil 2.5mum Pt CVD carbon both sides
power_interpolatorG13 = interp1d([0,0.,2.5,10],[0,0,0.043,0.043])
laserG13=['Rec-000002-001_00_52_53_090','Rec-000003-001_00_54_13_444','Rec-000004-001_00_57_11_960','Rec-000005-001_00_58_11_891','Rec-000006-001_00_58_53_611','Rec-000007-001_00_59_35_330','Rec-000008-001_01_01_32_676','Rec-000009-001_01_03_18_082','Rec-000011-001_01_05_13_314','Rec-000012-001_01_05_54_263','Rec-000013-001_01_06_52_516','Rec-000014-001_01_07_34_705','Rec-000015-001_01_08_33_328','Rec-000016-001_01_09_32_084','Rec-000017-001_01_10_33_524','Rec-000019-001_01_12_34_222','Rec-000020-001_01_13_32_442','Rec-000021-001_01_14_33_949','Rec-000022-001_01_15_53_815','Rec-000023-001_01_16_52_992','Rec-000024-001_01_17_34_410','Rec-000025-001_01_18_32_228','Rec-000026-001_01_19_36_417','Rec-000027-001_01_22_33_357','Rec-000029-001_01_24_33_419','Rec-000030-001_01_25_13_663','Rec-000031-001_01_26_39_719','Rec-000032-001_01_27_13_792','Rec-000033-001_01_28_11_945','Rec-000034-001_01_29_13_183']
for i in range(len(laserG13)):
	laserG13[i] = '/home/ffederic/work/irvb/laser/Sept18_2024/' + laserG13[i]
voltlaserG13=[1]
freqlaserG13=[0.02]
dutylaserG13=[0.5]
full_pathfile_index = full_pathfile_index.merge(xr.Dataset({'laserG13':laserG13}))
laserROIG13 = [ 'ff' ] * len(laserG13)
collection_of_records['laserG13'] = dict([])
collection_of_records['laserG13']['path_files_laser'] = laserG13
collection_of_records['laserG13']['voltlaser'] = voltlaserG13
collection_of_records['laserG13']['freqlaser'] = freqlaserG13
collection_of_records['laserG13']['dutylaser'] = dutylaserG13
collection_of_records['laserG13']['laserROI'] = laserROIG13
collection_of_records['laserG13']['reference_clear'] = [vacuumG13] * len(laserG13)
collection_of_records['laserG13']['power_interpolator'] = [power_interpolatorG13] * len(laserG13)
collection_of_records['laserG13']['focus_status'] = ['focused'] * len(laserG13)
collection_of_records['laserG13']['foil_position_dict'] = [dict([('angle',0),('foilcenter',[162,133]),('foilhorizwpixel',240)])] * len(laserG13)
collection_of_records['laserG13']['scan_type'] = 'power&freq'	# other: 'freq&duty' 'power&freq' 'freq'
collection_of_records['laserG13']['absorber_material'] = 'Au'	# name of the element from periodic table



##  IRVB FOIL PARAMETERS

foilemissivity=[0.87915237,0.84817978,0.76761355,0.73948514,0.81449329,0.86221733,0.81863479,0.83238318,0.75014256,0.82455539,0.87319584,0.84327053,0.8539331,0.83438179,0.82835241,0.83600738,0.8822808,0.84797073,0.75170152,0.87531842,0.86392473,0.89127889,0.89074805,0.86384333,0.82690635,0.88410513,0.84639702,0.78673421,0.84965535,0.8552314,0.82101809,0.81856018,0.80005979,0.89601878,0.83453686,0.91161225,0.88091159,0.81225521,0.87672616,0.8256423,0.85326844,0.85851042,0.83989414,0.84155748,0.84208017,0.8138739,0.87013654,0.83342386,0.84799913,0.94714049,0.86117643,0.87311947,0.86511982,0.94298577,0.94470884,0.88453373,0.84725634,0.84136077,0.91724797,0.88871259,0.84423068,0.84649853,0.93356454] #[au]
foilthickness=[1.49682E-06,1.37544E-06,1.6084E-06,1.63786E-06,1.41477E-06,0.000001314,1.33445E-06,1.43023E-06,0.000001259,1.06779E-06,1.1723E-06,1.36939E-06,1.31647E-06,1.43612E-06,1.53133E-06,1.39534E-06,0.000001335,1.48462E-06,1.65731E-06,1.35984E-06,1.36457E-06,1.29692E-06,0.000001389,1.39388E-06,1.47688E-06,1.0365E-06,1.01075E-06,1.18426E-06,1.18164E-06,1.28718E-06,1.35841E-06,1.41908E-06,1.48667E-06,1.27384E-06,1.43041E-06,1.27027E-06,0.000001233,1.43669E-06,1.36793E-06,1.46373E-06,1.35276E-06,1.19541E-06,1.2682E-06,0.000001148,1.04091E-06,1.01677E-06,1.03126E-06,1.09143E-06,1.2775E-06,1.20106E-06,1.35034E-06,1.42827E-06,1.3802E-06,1.21383E-06,1.17713E-06,0.000001358,1.28267E-06,1.11787E-06,1.02187E-06,1.01785E-06,1.07125E-06,1.00346E-06,8.67195E-07] #[m]


foilemissivity=np.reshape(foilemissivity,(7,9))
foilemissivity=np.flip(foilemissivity,0)
#foilemissivity=foilemissivity.transpose()
foilthickness=np.reshape(foilthickness,(7,9))
foilthickness=np.flip(foilthickness,0)
#foilthickness=foilthickness.transpose()

# size of the FLIR SC7500 camera sensor
max_ROI = [[0,255],[0,319]]


Ptthermalconductivity=71.6 #[W/(m·K)]
Ptspecificheat=133 #[J/(kg K)]
Ptdensity=21.45*1000 #[kg/m3]
Ptthermaldiffusivity=Ptthermalconductivity/(Ptspecificheat*Ptdensity)    #m2/s
Cthermalconductivity=7 #[W/(m·K)]	# assumed in the unfavorable direction of a graphite crystal
Cspecificheat=709 #[J/(kg K)]
Cdensity=0.684*1000 #[kg/m3]
Cthermaldiffusivity=Cthermalconductivity/(Cspecificheat*Cdensity)    #m2/s
Tithermalconductivity=17 #[W/(m·K)]
Tispecificheat=523 #[J/(kg K)]
Tidensity=4500 #[kg/m3]
Tithermaldiffusivity=Tithermalconductivity/(Tispecificheat*Tidensity)    #m2/s
Redensity = 21040 #[kg/m3]
Wdensity = 19350 #[kg/m3]
Osdensity = 22600 #[kg/m3]
Tadensity = 16650 #[kg/m3]
Hfdensity = 13310 #[kg/m3]
Aldensity = 2702 #[kg/m3]
Zrdensity = 6510 #[kg/m3]
Authermalconductivity=317 #[W/(m·K)]
Auspecificheat=129 #[J/(kg K)]
Audensity = 19320 #[kg/m3]
Authermaldiffusivity=Authermalconductivity/(Auspecificheat*Audensity)    #m2/s
Irdensity = 21040 #[kg/m3]

sigmaSB=5.6704e-08 #[W/(m2 K4)]
zeroC=273.15 #K / C


J_to_eV = 6.242e18	#1J = 1/J_to_eV eV
eV_to_K = 8.617333262145e-5	# eV/K
boltzmann_constant_J = 1.380649e-23	# J/K
hydrogen_mass = 1.008*1.660*1e-27	# kg
electron_mass = 9.10938356* 1e-31	# kg
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s


# Library of laser experiments characteristics
# To be compiled after any new experiment is added



###################################################################################################

def laser_shot_library(pathfiles,full_pathfile_index=full_pathfile_index,new_and_standard=False):


	import numpy as np
	import os, sys
	os.chdir("/home/ffederic/work/python_library/collect_and_eval")
	import collect_and_eval as coleval




	type = '.npy'
	filenames = coleval.all_file_names(pathfiles, type)[0]

	if bool((full_pathfile_index['laser12']==pathfiles).any()):
		framerate = 994
		datashort = np.load(os.path.join(pathfiles, filenames))
		data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
		data[:, :, 64:96, :] = datashort
		type_of_experiment = 'low duty cycle partially_defocused'
		poscentred = [[15, 80], [40, 75], [80, 85]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		inttime = 1
	elif ( bool((full_pathfile_index['laser15']==pathfiles).any()) or bool((full_pathfile_index['laser16']==pathfiles).any()) or bool((full_pathfile_index['laser21']==pathfiles).any()) or bool((full_pathfile_index['laser22']==pathfiles).any()) or bool((full_pathfile_index['laser23']==pathfiles).any()) ):
		framerate = 994
		datashort = np.load(os.path.join(pathfiles, filenames))
		data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
		data[:, :, 64:128, :] = datashort
		type_of_experiment = 'low duty cycle partially_defocused'
		poscentred = [[15, 80], [40, 75], [80, 85]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		inttime = 1
	elif ( bool((full_pathfile_index['laser24']==pathfiles).any()) or bool((full_pathfile_index['laser26']==pathfiles).any()) or bool((full_pathfile_index['laser27']==pathfiles).any()) or bool((full_pathfile_index['laser28']==pathfiles).any()) or bool((full_pathfile_index['laser29']==pathfiles).any()) or bool((full_pathfile_index['laser31']==pathfiles).any()) ):
		framerate = 1974
		datashort = np.load(os.path.join(pathfiles, filenames))
		data = np.multiply(3000, np.ones((1, np.shape(datashort)[1], 256, 320)))
		data[:, :, 64:128, 128:] = datashort
		type_of_experiment = 'low duty cycle partially_defocused'
		poscentred = [[15, 80], [40, 75], [80, 85]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/0.5ms383Hz/average'
		inttime = 0.5
	elif (bool((full_pathfile_index['vacuum1'][1] == pathfiles).any()) or bool(
			(full_pathfile_index['vacuum1'][3] == pathfiles).any()) ):
		framerate = 994
		datashort = np.load(os.path.join(pathfiles, filenames))
		data = np.multiply(6000, np.ones((1, np.shape(datashort)[1], 256, 320)))
		data[:, :, 64:96, :] = datashort
		type_of_experiment = 'low duty cycle partially_defocused'
		poscentred = [[15, 80], [40, 75], [80, 85]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		inttime = 1
	elif (bool((full_pathfile_index['laser11'] == pathfiles).any()) ):
		framerate = 383
		data = np.load(os.path.join(pathfiles, filenames))
		type_of_experiment = 'partially_defocused'
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		inttime = 1
	elif ( bool((full_pathfile_index['laser25']==pathfiles).any()) or bool((full_pathfile_index['laser30']==pathfiles).any()) ):
		framerate = 383
		data = np.load(os.path.join(pathfiles, filenames))
		type_of_experiment = 'focused'
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/2ms383Hz/average'
		inttime = 2
	elif new_and_standard == False:
		print('The path '+pathfiles)
		print('is not in the list of known files. Maybe there is something wrong going on.')
		exit()

	elif new_and_standard == True:
		framerate = 383
		data = np.load(os.path.join(pathfiles, filenames))
		type_of_experiment = 'focused'
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]
		pathparams = '/home/ffederic/work/irvb/2018-05-14_multiple_search_for_parameters/1ms383Hz/average'
		inttime = 1


	return framerate,data,type_of_experiment,poscentred,pathparams,inttime







###################################################################################################

def laser_shot_location_library(pathfiles,full_pathfile_index=full_pathfile_index):




	if (bool((full_pathfile_index['laser11'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser12'] == pathfiles).any())):
		powerzoom = 5 + addition
		laserspot = [41, 172]
	elif (bool((full_pathfile_index['laser13'] == pathfiles).any())):
		powerzoom = 3 + addition
		laserspot = [26, 196]
	elif (bool((full_pathfile_index['laser14'] == pathfiles).any())):
		powerzoom = 3 + addition
		laserspot = [27, 156]
	elif (bool((full_pathfile_index['laser15'] == pathfiles).any())):
		powerzoom = 2 + addition
		laserspot = [43, 188]
	elif (bool((full_pathfile_index['laser16'] == pathfiles).any())):
		powerzoom = 3 + addition
		laserspot = [41, 186]
	elif (bool((full_pathfile_index['laser17'] == pathfiles).any())):
		powerzoom = 2 + addition
		laserspot = [21, 216]
	elif (bool((full_pathfile_index['laser18'] == pathfiles).any())):
		powerzoom = 1 + addition
		laserspot = [22, 159]
	elif (bool((full_pathfile_index['laser19'] == pathfiles).any())):
		powerzoom = 1 + addition
		laserspot = [54, 187]
	elif (bool((full_pathfile_index['laser20'] == pathfiles).any())):
		powerzoom = 6 + addition
		laserspot = [42, 180]
	elif (bool((full_pathfile_index['laser21'] == pathfiles).any())):
		powerzoom = 5 + addition
		laserspot = [42, 180]
	elif (bool((full_pathfile_index['laser22'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser23'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser24'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser25'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser26'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser27'] == pathfiles).any())):
		powerzoom = 1 + addition
		laserspot = [42, 189]
	elif (bool((full_pathfile_index['laser28'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser29'] == pathfiles).any())):
		powerzoom = 3 + addition
		laserspot = [43, 188]
	elif (bool((full_pathfile_index['laser30'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser31'] == pathfiles).any())):
		powerzoom = 6 + addition
		laserspot = [42, 190]
	elif (bool((full_pathfile_index['laser32'] == pathfiles).any())):
		powerzoom = 2 + addition
		laserspot = [49, 214]
	elif (bool((full_pathfile_index['laser1'] == pathfiles).any()) or bool(
			(full_pathfile_index['laser2'] == pathfiles).any())):
		powerzoom = 3 + addition
		laserspot = [41, 172]

	else:
		print('The path '+pathfiles)
		print("doesn't have a laser spot location assigned. Maybe there is something wrong going on.")
		exit()


	return powerzoom,laserspot



###################################################################################################

noise_space_x = [1,2,3,4,5,6,10]
noise_space_y = [152,65,45.7,35.4,29.8,25.8,17.7]
noise_time_x = [1,2,3,4,5,6,8,10]
noise_time_y = [152,86,67,57,51,46.7,40.8,36.8]


def return_power_noise_level(spatial_averaging,time_averaging,noise_space_x=noise_space_x,noise_space_y=noise_space_y,noise_time_x=noise_time_x,noise_time_y=noise_time_y):

	# 2018-03-17 This is just to return the maximum expected level of total power noise for given averaging

	noise_space_x = np.array(noise_space_x)
	noise_space_y= np.array(noise_space_y)
	noise_time_x= np.array(noise_time_x)
	noise_time_y= np.array(noise_time_y)

	f_space = interp1d(noise_space_x, noise_space_y/(noise_space_y.max()))
	f_time = interp1d(noise_time_x, noise_time_y/(noise_time_y.max()))

	return noise_space_y[0]*f_space(spatial_averaging)*f_time(time_averaging)

####################################################################################################

# Points to show the silouette of the inner surfaces

_MASTU_CORE_GRID_POLYGON = np.array([
	(1.49, -0.0),
    (1.49, -1.007),
    (1.191, -1.007),
    (0.893, -1.304),
    (0.868, -1.334),
    (0.847, -1.368),
    (0.832, -1.404),
    (0.823, -1.442),
    (0.82, -1.481),
    (0.82, -1.49),
    (0.825, -1.522),
    (0.84, -1.551),
    (0.864, -1.573),
    (0.893, -1.587),
    (0.925, -1.59),
    (1.69, -1.552),
    (2.0, -1.56),
    (2.0, -2.169),
    (1.319, -2.169),
    (1.769, -1.719),
    (1.73, -1.68),
    (1.35, -2.06),
    (1.09, -2.06),
    (0.9, -1.87),
    (0.36, -1.33),
    (0.333, -1.303),
    (0.333, -1.1),
    (0.261, -0.5),
    (0.261, 0.0),
	(0.261, 0.2),     # point added just to cut the unnecessary voxels the 04/02/2018
    # (0.261, 0.5),
    # (0.333, 0.8),     # point added just to cut the unnecessary voxels	# Replaced 04/02/2018
    # (0.333, 1.1),
    # (0.333, 1.303),
    # (0.36, 1.33),
    # (0.9, 1.87),
    # (1.09, 2.06),
    # (1.35, 2.06),
    # (1.73, 1.68),
    # (1.769, 1.719),
    # (1.319, 2.169),
    # (2.0, 2.169),
    # (2.0, 1.56),
    # (1.69, 1.552),
    # (0.925, 1.59),
    # (0.893, 1.587),
    # (0.864, 1.573),
    # (0.84, 1.551),
    # (0.825, 1.522),
    # (0.82, 1.49),
    # (0.82, 1.481),
    # (0.823, 1.442),
    # (0.832, 1.404),
    # (0.847, 1.368),
    # (0.868, 1.334),
    # (0.893, 1.304),
    # (1.191, 1.007),
    # (1.49, 1.007)
    # (1.49, 0.8)     # point added just to cut the unnecessary voxels	# Replaced 04/02/2018
	(1.49, 0.75)  # point added just to cut the unnecessary voxels the 04/02/2018
])


################################################################################################################




# SOLPS simulations for MAST-U

mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
mdsnos_cd1H = [69548, 69549, 69485, 69550, 69551, 69552, 69553, 69554, 69555, 69556, 69557, 69558, 69559, 69560, 69561,69562, 69563, 69564, 69565, 69566]
mdsnos_cd15H = [69567, 69568, 69569, 69570, 69571, 69572, 69573, 69574, 69575, 69576, 69577, 69578, 69579, 69580,69581]
mdsnos_cd2H = [69582, 69583, 69584, 69585, 69586, 69587, 69588, 69589, 69590, 69591, 69592, 69593, 69594, 69595, 69596,69597, 69598, 69599, 69600, 69601]

mdsnos_sxd1L = [69602, 69603, 69604, 69605, 69606, 69607, 69608, 69609, 69610, 69611]
mdsnos_sxd2L = [69612, 69613, 69614, 69615, 69616, 69617, 69618, 69619, 69620, 69621, 69622, 69623, 69624, 69625, 69626,69627, 69628, 69629]
mdsnos_sxd1H = [69637, 69638, 69639, 69640, 69641, 69642, 69643, 69644, 69630, 69631, 69632, 69633, 69634, 69635,69636]
mdsnos_sxd15H = [69645, 69646, 69647, 69648, 69649, 69650, 69651, 69652, 69653, 69654, 69655, 69656, 69657, 69658,69659, 69660, 69661, 69662, 69663, 69664]
mdsnos_sxd2H = [69665, 69666, 69667, 69668, 69669, 69670, 69671, 69672, 69673, 69674, 69675, 69676, 69677, 69678, 69679,69680, 69681, 69682, 69683, 69684]


# LEGEND
#
# cd = conventional divertor
# sxd = super x
#
# 1 = core scope(initial power)
# 15 = core scope + extra nbi(1 and a half times power)
# 2 = core scope + extra nbi + cryo(predicted max power)
#
# H = h - mode
# L = l - mode
#
# MAST simulation that compared best against experimental data
# 39625


MASTU_shots_timing = dict([])
MASTU_shots_timing['44115'] = dict([('pulse_start',0.5)])	# s
MASTU_shots_timing['44116'] = dict([('pulse_start',0.5)])
MASTU_shots_timing['44117'] = dict([('pulse_start',1.5)])
MASTU_shots_timing['44118'] = dict([('pulse_start',1.5)])
MASTU_shots_timing['44119'] = dict([('pulse_start',1.5)])
MASTU_shots_timing['44120'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44121'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44122'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44123'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44124'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44125'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44126'] = dict([('pulse_start',2.0)])
MASTU_shots_timing['44127'] = dict([('pulse_start',1.6)])




# 2023/09/12 attebuation coefficients for Pt, Ti, C in terms of [µ/￼density] Total [cm2 g-1]
# from https://physics.nist.gov/PhysRefData/FFast/html/form.html
# more detailed explanation (ultimately unnecessary) at
# https://physics.nist.gov/PhysRefData/FFast/Text2000/sec08.html
# https://radiopaedia.org/articles/linear-attenuation-coefficient?lang=gb

energy_C_attenuation = np.array([0.006432 , 0.006528 , 0.01069 , 0.01142761 , 0.01221612 , 0.01305903 , 0.0139601 , 0.01492335 , 0.01595306 , 0.01705382 , 0.01823053 , 0.0191198 , 0.01941245 , 0.01948844 , 0.01949049 , 0.01960755 , 0.0199002 , 0.02083314 , 0.02227063 , 0.0238073 , 0.02545001 , 0.02720606 , 0.02908327 , 0.03109002 , 0.03323523 , 0.03552846 , 0.03797993 , 0.04060054 , 0.04340198 , 0.04639671 , 0.04959809 , 0.05302035 , 0.05667876 , 0.06058959 , 0.06477028 , 0.06923942 , 0.07401695 , 0.07912411 , 0.08458368 , 0.09041995 , 0.09665893 , 0.1033284 , 0.1104581 , 0.1180797 , 0.1262272 , 0.1349368 , 0.1442475 , 0.1542005 , 0.1648404 , 0.1762144 , 0.1883732 , 0.2013709 , 0.2152655 , 0.2301188 , 0.245997 , 0.2629708 , 0.278124 , 0.2811158 , 0.282381 , 0.2835162 , 0.285219 , 0.289476 , 0.3005128 , 0.3212482 , 0.3434143 , 0.3671099 , 0.3924405 , 0.4195189 , 0.4484657 , 0.4794098 , 0.5124891 , 0.5478508 , 0.5856525 , 0.6260625 , 0.6692609 , 0.7154399 , 0.7648052 , 0.8175768 , 0.8739896 , 0.9342948 , 0.9987612 , 1.067676 , 1.141345 , 1.220098 , 1.304285 , 1.394281 , 1.490486 , 1.593329 , 1.703269 , 1.820795 , 1.94643 , 2.080733 , 2.224304 , 2.377781 , 2.541848 , 2.717235 , 2.904724 , 3.10515 , 3.319406 , 3.548445 , 3.793288 , 4.055024 , 4.334821 , 4.633924 , 4.953664 , 5.295467 , 5.660855 , 6.051453 , 6.469004 , 6.915365 , 7.392525 , 7.902609 , 8.44789 , 9.030794 , 9.653919 , 10.32004 , 11.03212 , 11.79334 , 12.60708 , 13.47697 , 14.40688 , 15.40095 , 16.46362 , 17.59961 , 18.81398 , 20.11215 , 21.49988 , 22.98338 , 24.56923 , 26.2645 , 28.07676 , 30.01405 , 32.08502 , 34.29889 , 36.66551 , 39.19543 , 41.89992 , 44.79101 , 47.88159 , 51.18542 , 54.71721 , 58.4927 , 62.5287 , 66.84318 , 71.45536 , 76.38578 , 81.6564 , 87.29069 , 93.31374 , 99.75239 , 106.6353 , 113.9931 , 121.8587 , 130.2669 , 139.2553 , 148.864 , 159.1356 , 170.1159 , 181.8539 , 194.4018 , 207.8156 , 222.1548 , 237.4835 , 253.8699 , 271.3869 , 290.1126 , 310.1304 , 331.5294 , 354.4049 , 378.8588 , 405.0001 , 432.9451])*1e3	# eV
mass_coefficient_C_attenuation = np.array([1072900 , 1051800 , 569990 , 525300 , 483160 , 443300 , 405580 , 369920 , 336260 , 304590 , 274890 , 254910 , 248740 , 247180 , 247130 , 272340 , 267450 , 253120 , 234370 , 217230 , 201340 , 186430 , 172270 , 158790 , 145940 , 133690 , 122040 , 111000 , 100590 , 90829 , 81721 , 73276 , 65488 , 58347 , 51834 , 45922 , 40582 , 35777 , 31461 , 27596 , 24151 , 21091 , 18383 , 15996 , 13847 , 11956 , 10310 , 8880.5 , 7641.5 , 6569.1 , 5642.4 , 4842.5 , 4153.1 , 3559.6 , 3049 , 2610.4 , 2290.4 , 2233.9 , 2210.5 , 2189.9 , 48382 , 46932 , 43462 , 37872 , 32793 , 28213 , 24135 , 20555 , 17442 , 14756 , 12451 , 10483 , 8809.3 , 7390.2 , 6190.1 , 5177.7 , 4325.1 , 3608.4 , 3006.9 , 2502.7 , 2080.6 , 1726.6 , 1430.7 , 1177.1 , 968.5 , 796.88 , 655.69 , 539.55 , 444 , 365.39 , 300.72 , 247.52 , 203.74 , 167.73 , 138.09 , 113.71 , 93.646 , 76.739 , 62.598 , 51.075 , 41.686 , 34.035 , 27.799 , 22.717 , 18.574 , 15.197 , 12.384 , 10.096 , 8.2403 , 6.7348 , 5.513 , 4.5212 , 3.7161 , 3.0584 , 2.5142 , 2.0746 , 1.7194 , 1.432 , 1.1995 , 1.0112 , 0.85861 , 0.73485 , 0.63435 , 0.55263 , 0.4861 , 0.43181 , 0.38745 , 0.35109 , 0.3212 , 0.29654 , 0.2759 , 0.25871 , 0.24435 , 0.23229 , 0.22206 , 0.2133 , 0.20573 , 0.1991 , 0.19325 , 0.18799 , 0.18322 , 0.17883 , 0.17474 , 0.17091 , 0.16726 , 0.16376 , 0.16039 , 0.15711 , 0.1539 , 0.15076 , 0.14766 , 0.1446 , 0.14158 , 0.13859 , 0.13562 , 0.13268 , 0.12978 , 0.1269 , 0.12404 , 0.12122 , 0.11843 , 0.11567 , 0.11296 , 0.11027 , 0.10763 , 0.10503 , 0.10247 , 0.099959 , 0.097495 , 0.095081 , 0.092716 , 0.090403])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_C_attenuation_interpolator = interp1d(energy_C_attenuation,mass_coefficient_C_attenuation*Cdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Ti_attenuation = np.array([0.034773 , 0.035292 , 0.03552846 , 0.03797993 , 0.04060054 , 0.04340198 , 0.04639671 , 0.04959809 , 0.05302035 , 0.05667876 , 0.059094 , 0.0599985 , 0.0602397 , 0.06058959 , 0.0606015 , 0.061506 , 0.06477028 , 0.06923942 , 0.07401695 , 0.07912411 , 0.08458368 , 0.09041995 , 0.09665893 , 0.1033284 , 0.1104581 , 0.1180797 , 0.1262272 , 0.1349368 , 0.1442475 , 0.1542005 , 0.1648404 , 0.1762144 , 0.1883732 , 0.2013709 , 0.2152655 , 0.2301188 , 0.245997 , 0.2629708 , 0.2811158 , 0.3005128 , 0.3212482 , 0.3434143 , 0.3671099 , 0.3924405 , 0.4195189 , 0.44639 , 0.4484657 , 0.45227 , 0.4532225 , 0.4550445 , 0.4577775 , 0.4591925 , 0.4610385 , 0.4638075 , 0.46461 , 0.47073 , 0.4794098 , 0.5124891 , 0.5478508 , 0.552426 , 0.5608815 , 0.5631363 , 0.5665185 , 0.574974 , 0.5856525 , 0.6260625 , 0.6692609 , 0.7154399 , 0.7648052 , 0.8175768 , 0.8739896 , 0.9342948 , 0.9987612 , 1.067676 , 1.141345 , 1.220098 , 1.304285 , 1.394281 , 1.490486 , 1.593329 , 1.703269 , 1.820795 , 1.94643 , 2.080733 , 2.224304 , 2.377781 , 2.541848 , 2.717235 , 2.904724 , 3.10515 , 3.319406 , 3.548445 , 3.793288 , 4.055024 , 4.334821 , 4.633924 , 4.867072 , 4.941568 , 4.953664 , 4.961434 , 4.991232 , 5.065728 , 5.295467 , 5.660855 , 6.051453 , 6.469004 , 6.915365 , 7.392525 , 7.902609 , 8.44789 , 9.030794 , 9.653919 , 10.32004 , 11.03212 , 11.79334 , 12.60708 , 13.47697 , 14.40688 , 15.40095 , 16.46362 , 17.59961 , 18.81398 , 20.11215 , 21.49988 , 22.98338 , 24.56923 , 26.2645 , 28.07676 , 30.01405 , 32.08502 , 34.29889 , 36.66551 , 39.19543 , 41.89992 , 44.79101 , 47.88159 , 51.18542 , 54.71721 , 58.4927 , 62.5287 , 66.84318 , 71.45536 , 76.38578 , 81.6564 , 87.29069 , 93.31374 , 99.75239 , 106.6353 , 113.9931 , 121.8587 , 130.2669 , 139.2553 , 148.864 , 159.1356 , 170.1159 , 181.8539 , 194.4018 , 207.8156 , 222.1548 , 237.4835 , 253.8699 , 271.3869 , 290.1126 , 310.1304 , 331.5294 , 354.4049 , 378.8588 , 405.0001 , 432.9451])*1e3	# eV
mass_coefficient_Ti_attenuation = np.array([111400 , 98018 , 92523 , 51992 , 31199 , 21267 , 16102 , 13276 , 11718 , 10910 , 10661 , 10607 , 10596 , 15404 , 15403 , 15399 , 15493 , 15797 , 16205 , 16658 , 17114 , 17531 , 17872 , 18104 , 18200 , 18139 , 17912 , 17516 , 16956 , 16227 , 15361 , 14392 , 13357 , 12290 , 11220 , 10173 , 9166.9 , 8215.8 , 7328.7 , 6511.1 , 5765 , 5089.8 , 4482.9 , 3940.5 , 3458.3 , 3059 , 3031 , 2980.5 , 2968.1 , 2944.5 , 21326 , 21177 , 20986 , 29710 , 29593 , 28729 , 27565 , 23695 , 20320 , 19931 , 19236 , 19056 , 21354 , 20661 , 19828 , 17063 , 14655 , 12562 , 10759 , 9175.6 , 7812 , 6648.1 , 5654.1 , 4759 , 4003.9 , 3367.4 , 2832.2 , 2382.6 , 2005 , 1688 , 1421.9 , 1198.4 , 1003.7 , 841.11 , 705.79 , 590.72 , 494.44 , 414.29 , 347.49 , 289 , 238.46 , 197.02 , 162.98 , 134.99 , 111.96 , 92.975 , 81.168 , 77.838 , 77.316 , 76.983 , 697.78 , 668.93 , 591.07 , 496.29 , 418.5 , 353.27 , 296.29 , 248.22 , 207.81 , 173.94 , 145.43 , 121.17 , 100.59 , 83.249 , 68.894 , 57.028 , 47.22 , 39.109 , 32.402 , 26.854 , 22.264 , 18.468 , 15.325 , 12.725 , 10.508 , 8.6841 , 7.1826 , 5.9158 , 4.87 , 4.015 , 3.3158 , 2.7438 , 2.2758 , 1.8925 , 1.5786 , 1.3214 , 1.1105 , 0.93748 , 0.79538 , 0.67859 , 0.5825 , 0.50333 , 0.43803 , 0.38373 , 0.33818 , 0.30058 , 0.26943 , 0.24355 , 0.22195 , 0.20386 , 0.1886 , 0.17567 , 0.16463 , 0.15514 , 0.14691 , 0.1397 , 0.13335 , 0.12768 , 0.12257 , 0.11792 , 0.11366 , 0.10972 , 0.10603 , 0.10256 , 0.099272 , 0.096139 , 0.093138 , 0.09025 , 0.087461])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Ti_attenuation_interpolator = interp1d(energy_Ti_attenuation,mass_coefficient_Ti_attenuation*Tidensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Pt_attenuation = np.array([0.006156007 , 0.006247887 , 0.007291108 , 0.007402706 , 0.007432466 , 0.007477105 , 0.007588704 , 0.01069 , 0.01142761 , 0.01221612 , 0.01305903 , 0.0139601 , 0.01492335 , 0.01595306 , 0.01705382 , 0.01823053 , 0.01948844 , 0.02083314 , 0.02227063 , 0.0238073 , 0.02545001 , 0.02720606 , 0.02908327 , 0.03109002 , 0.03323523 , 0.03552846 , 0.03797993 , 0.04060054 , 0.04340198 , 0.04639671 , 0.04959809 , 0.050666 , 0.0514415 , 0.0516483 , 0.0519585 , 0.052734 , 0.05302035 , 0.05667876 , 0.06058959 , 0.063994 , 0.06477028 , 0.0649735 , 0.0652347 , 0.0656265 , 0.066606 , 0.06923942 , 0.069678 , 0.0707445 , 0.0710289 , 0.0714555 , 0.072522 , 0.072814 , 0.0739285 , 0.07401695 , 0.0742257 , 0.0746715 , 0.075786 , 0.07912411 , 0.08458368 , 0.09041995 , 0.09665893 , 0.099666 , 0.1011915 , 0.1015983 , 0.1022085 , 0.1033284 , 0.103734 , 0.1104581 , 0.1180797 , 0.1262272 , 0.1349368 , 0.1442475 , 0.1542005 , 0.1648404 , 0.1762144 , 0.1883732 , 0.2013709 , 0.2152655 , 0.2301188 , 0.245997 , 0.2629708 , 0.2811158 , 0.3005128 , 0.307034 , 0.3117335 , 0.3129867 , 0.3148665 , 0.319566 , 0.3212482 , 0.324184 , 0.329146 , 0.3304692 , 0.332454 , 0.337416 , 0.3434143 , 0.3671099 , 0.3924405 , 0.4195189 , 0.4484657 , 0.4794098 , 0.5 , 0.5025 , 0.5050125 , 0.50753756 , 0.51007525 , 0.51262563 , 0.51518875 , 0.5177647 , 0.51833567 , 0.51966431 , 0.52035352 , 0.52295529 , 0.52557007 , 0.52819792 , 0.53083891 , 0.5334931 , 0.53616057 , 0.53884137 , 0.54153558 , 0.54424325 , 0.54696447 , 0.54969929 , 0.55244779 , 0.55521003 , 0.55798608 , 0.56077601 , 0.56357989 , 0.56639779 , 0.56922978 , 0.57207593 , 0.5749363 , 0.57781099 , 0.58070004 , 0.58360354 , 0.58652156 , 0.58945417 , 0.59240144 , 0.59536345 , 0.59834026 , 0.60133196 , 0.60433862 , 0.60736032 , 0.60832884 , 0.61007116 , 0.61039712 , 0.6134491 , 0.61651635 , 0.61959893 , 0.62269693 , 0.62581041 , 0.62893946 , 0.63208416 , 0.63524458 , 0.6384208 , 0.64161291 , 0.64482097 , 0.64804508 , 0.6512853 , 0.65454173 , 0.65781444 , 0.66110351 , 0.66440903 , 0.66773107 , 0.67106973 , 0.67442508 , 0.6777972 , 0.68118619 , 0.68459212 , 0.68801508 , 0.69145515 , 0.69491243 , 0.69838699 , 0.70187893 , 0.70538832 , 0.70891526 , 0.71245984 , 0.71602214 , 0.71960225 , 0.72085202 , 0.72320026 , 0.72681626 , 0.73045034 , 0.7341026 , 0.73777311 , 0.74146197 , 0.74516928 , 0.74889513 , 0.75263961 , 0.7564028 , 0.76018482 , 0.76398574 , 0.76780567 , 0.7716447 , 0.77550292 , 0.77938044 , 0.78327734 , 0.78719373 , 0.79112969 , 0.79508534 , 0.79906077 , 0.80305607 , 0.80707135 , 0.81110671 , 0.81516224 , 0.81923806 , 0.82333425 , 0.82745092 , 0.83158817 , 0.83574611 , 0.83992484 , 0.84412447 , 0.84834509 , 0.85258682 , 0.85684975 , 0.861134 , 0.86543967 , 0.86976687 , 0.8741157 , 0.87848628 , 0.88287871 , 0.8872931 , 0.89172957 , 0.89618822 , 0.90066916 , 0.9051725 , 0.90969837 , 0.91424686 , 0.91881809 , 0.92341218 , 0.92802924 , 0.93266939 , 0.93733274 , 0.9420194 , 0.9467295 , 0.95146315 , 0.95622046 , 0.96100156 , 0.96580657 , 0.9706356 , 0.97548878 , 0.98036623 , 0.98526806 , 0.9901944 , 0.99514537 , 1.0001211 , 1.0051217 , 1.0101473 , 1.015198 , 1.020274 , 1.0253754 , 1.0305023 , 1.0356548 , 1.0408331 , 1.0460372 , 1.0512674 , 1.0565238 , 1.0618064 , 1.0671154 , 1.072451 , 1.0778132 , 1.0832023 , 1.0886183 , 1.0940614 , 1.0995317 , 1.1050294 , 1.1105545 , 1.1161073 , 1.1216878 , 1.1272963 , 1.1329328 , 1.1385974 , 1.1442904 , 1.1500119 , 1.1557619 , 1.1615407 , 1.1673484 , 1.1731852 , 1.1790511 , 1.1849464 , 1.1908711 , 1.1968254 , 1.2028096 , 1.2088236 , 1.2148677 , 1.2209421 , 1.2270468 , 1.233182 , 1.2393479 , 1.2455447 , 1.2517724 , 1.2580312 , 1.2643214 , 1.270643 , 1.2769962 , 1.2833812 , 1.2897981 , 1.2962471 , 1.3027283 , 1.309242 , 1.3157882 , 1.3223671 , 1.328979 , 1.3356239 , 1.342302 , 1.3490135 , 1.3557586 , 1.3625374 , 1.36935 , 1.3761968 , 1.3830778 , 1.3899932 , 1.3969431 , 1.4039278 , 1.4109475 , 1.4180022 , 1.4250922 , 1.4322177 , 1.4393788 , 1.4465757 , 1.4538086 , 1.4610776 , 1.468383 , 1.4757249 , 1.4831035 , 1.490519 , 1.4979716 , 1.5054615 , 1.5129888 , 1.5205537 , 1.5281565 , 1.5357973 , 1.5434763 , 1.5511937 , 1.5589496 , 1.5667444 , 1.5745781 , 1.582451 , 1.5903633 , 1.5983151 , 1.6063066 , 1.6143382 , 1.6224099 , 1.6305219 , 1.6386745 , 1.6468679 , 1.6551022 , 1.6633777 , 1.6716946 , 1.6800531 , 1.6884534 , 1.6968956 , 1.7053801 , 1.713907 , 1.7224766 , 1.7310889 , 1.7397444 , 1.7484431 , 1.7571853 , 1.7659712 , 1.7748011 , 1.7836751 , 1.7925935 , 1.8015565 , 1.8105642 , 1.8196171 , 1.8287151 , 1.8378587 , 1.847048 , 1.8562833 , 1.8655647 , 1.8748925 , 1.884267 , 1.8936883 , 1.9031567 , 1.9126725 , 1.9222359 , 1.9318471 , 1.9415063 , 1.9512138 , 1.9609699 , 1.9707747 , 1.9806286 , 1.9905318 , 2.0004844 , 2.0104868 , 2.0205393 , 2.030642 , 2.0407952 , 2.0509992 , 2.0612542 , 2.0715604 , 2.0819182 , 2.0923278 , 2.1027895 , 2.1133034 , 2.1210653 , 2.1221346 , 2.1238699 , 2.1344893 , 2.1451617 , 2.1558875 , 2.166667 , 2.1775003 , 2.1883878 , 2.1993297 , 2.2011954 , 2.2026046 , 2.2103264 , 2.221378 , 2.2324849 , 2.2436473 , 2.2548656 , 2.2661399 , 2.2774706 , 2.2888579 , 2.3003022 , 2.3118037 , 2.3233628 , 2.3349796 , 2.3466545 , 2.3583878 , 2.3701797 , 2.3820306 , 2.3939407 , 2.4059104 , 2.41794 , 2.4300297 , 2.4421798 , 2.4543907 , 2.4666627 , 2.478996 , 2.491391 , 2.5038479 , 2.5163672 , 2.528949 , 2.5415938 , 2.5543017 , 2.5670732 , 2.5799086 , 2.5928082 , 2.6057722 , 2.6188011 , 2.6318951 , 2.6429266 , 2.6450545 , 2.6478735 , 2.6582798 , 2.6715712 , 2.6849291 , 2.6983537 , 2.7118455 , 2.7254047 , 2.7390317 , 2.7527269 , 2.7664905 , 2.780323 , 2.7942246 , 2.8081957 , 2.8222367 , 2.8363479 , 2.8505296 , 2.8647823 , 2.8791062 , 2.8935017 , 2.9079692 , 2.9225091 , 2.9371216 , 2.9518072 , 2.9665662 , 2.9813991 , 2.9963061 , 3.0112876 , 3.0221418 , 3.026344 , 3.0308581 , 3.0414758 , 3.0566831 , 3.0719666 , 3.0873264 , 3.102763 , 3.1182768 , 3.1338682 , 3.1495376 , 3.1652853 , 3.1811117 , 3.1970172 , 3.2130023 , 3.2290673 , 3.2452127 , 3.2614387 , 3.2777459 , 3.2891773 , 3.2941347 , 3.3028227 , 3.3106053 , 3.3271584 , 3.3437941 , 3.3605131 , 3.3773157 , 3.3942023 , 3.4111733 , 3.4282291 , 3.4453703 , 3.4625971 , 3.4799101 , 3.4973097 , 3.5147962 , 3.5323702 , 3.5500321 , 3.5677822 , 3.5856211 , 3.6035492 , 3.621567 , 3.6396748 , 3.6578732 , 3.6761626 , 3.6945434 , 3.7130161 , 3.7315812 , 3.7502391 , 3.7689903 , 3.7878352 , 3.8067744 , 3.8258083 , 3.8449373 , 3.864162 , 3.8834828 , 3.9029002 , 3.9224147 , 3.9420268 , 3.9617369 , 3.9815456 , 4.0014533 , 4.0214606 , 4.0415679 , 4.0617757 , 4.0820846 , 4.102495 , 4.1230075 , 4.1436226 , 4.1643407 , 4.1851624 , 4.2060882 , 4.2271186 , 4.2482542 , 4.2694955 , 4.290843 , 4.3122972 , 4.3338587 , 4.355528 , 4.3773056 , 4.3991921 , 4.4211881 , 4.443294 , 4.4655105 , 4.4878381 , 4.5102772 , 4.5328286 , 4.5554928 , 4.5782702 , 4.6011616 , 4.6241674 , 4.6472882 , 4.6705247 , 4.6938773 , 4.7173467 , 4.7409334 , 4.7646381 , 4.7884613 , 4.8124036 , 4.8364656 , 4.8606479 , 4.8849512 , 4.9093759 , 4.9339228 , 4.9585924 , 4.9833854 , 5.0083023 , 5.0333438 , 5.0585105 , 5.0838031 , 5.1092221 , 5.1347682 , 5.1604421 , 5.1862443 , 5.2121755 , 5.2382364 , 5.2644276 , 5.2907497 , 5.3172034 , 5.3437895 , 5.3705084 , 5.3973609 , 5.4243477 , 5.4514695 , 5.4787268 , 5.5061205 , 5.5336511 , 5.5613193 , 5.5891259 , 5.6170716 , 5.6451569 , 5.6733827 , 5.7017496 , 5.7302584 , 5.7589096 , 5.7877042 , 5.8166427 , 5.8457259 , 5.8749546 , 5.9043293 , 5.933851 , 5.9635202 , 5.9933378 , 6.0233045 , 6.053421 , 6.0836882 , 6.1141066 , 6.1446771 , 6.1754005 , 6.2062775 , 6.2373089 , 6.2684954 , 6.2998379 , 6.3313371 , 6.3629938 , 6.3948088 , 6.4267828 , 6.4589167 , 6.4912113 , 6.5236674 , 6.5562857 , 6.5890671 , 6.6220125 , 6.6551225 , 6.6883981 , 6.7218401 , 6.7554493 , 6.7892266 , 6.8231727 , 6.8572886 , 6.891575 , 6.9260329 , 6.9606631 , 6.9954664 , 7.0304437 , 7.0655959 , 7.1009239 , 7.1364285 , 7.1721107 , 7.2079712 , 7.2440111 , 7.2802311 , 7.3166323 , 7.3532155 , 7.3899815 , 7.4269314 , 7.4640661 , 7.5013864 , 7.5388934 , 7.5765878 , 7.6144708 , 7.6525431 , 7.6908058 , 7.7292599 , 7.7679062 , 7.8067457 , 7.8457794 , 7.8850083 , 7.9244334 , 7.9640555 , 8.0038758 , 8.0438952 , 8.0841147 , 8.1245352 , 8.1651579 , 8.2059837 , 8.2470136 , 8.2882487 , 8.3296899 , 8.3713384 , 8.4131951 , 8.455261 , 8.4975373 , 8.540025 , 9.030794 , 9.653919 , 10.32004 , 11.03212 , 11.33243 , 11.50588 , 11.55214 , 11.62152 , 11.79334 , 11.79497 , 12.60708 , 13.00715 , 13.20624 , 13.25933 , 13.33896 , 13.47697 , 13.53805 , 13.6023 , 13.8105 , 13.86602 , 13.9493 , 14.1575 , 14.40688 , 15.40095 , 16.46362 , 17.59961 , 18.81398 , 20.11215 , 21.49988 , 22.98338 , 24.56923 , 26.2645 , 28.07676 , 30.01405 , 32.08502 , 34.29889 , 36.66551 , 39.19543 , 41.89992 , 44.79101 , 47.88159 , 51.18542 , 54.71721 , 58.4927 , 62.5287 , 66.84318 , 71.45536 , 76.38578 , 76.8269 , 78.00282 , 78.3164 , 78.78677 , 79.96269 , 81.6564 , 87.29069 , 93.31374 , 99.75239 , 106.6353 , 113.9931 , 121.8587 , 130.2669 , 139.2553 , 148.864 , 159.1356 , 170.1159 , 181.8539 , 194.4018 , 207.8156 , 222.1548 , 237.4835 , 253.8699 , 271.3869 , 290.1126 , 310.1304 , 331.5294 , 354.4049 , 378.8588 , 405.0001 , 432.9451])*1e3	# eV
mass_coefficient_Pt_attenuation = np.array([40474 , 39638 , 33428 , 33345 , 33339 , 60636 , 59765 , 74256 , 83433 , 93785 , 104840 , 115990 , 126540 , 135690 , 142680 , 146850 , 147750 , 145200 , 139350 , 130620 , 119650 , 107170 , 94012 , 80908 , 68426 , 57012 , 46920 , 38238 , 30932 , 24894 , 19973 , 18611 , 17696 , 17462 , 34762 , 33023 , 32410 , 25731 , 20414 , 16878 , 16187 , 16013 , 15792 , 21854 , 20937 , 18742 , 18411 , 17642 , 17446 , 19026 , 18234 , 18023 , 17241 , 17181 , 17039 , 17927 , 17148 , 15026 , 12274 , 10177 , 8631.6 , 8083.8 , 7847 , 7788.1 , 7888.6 , 7742.8 , 7693 , 7079.7 , 6793.3 , 6854.6 , 7224.6 , 7890.5 , 8824 , 9977.1 , 11294 , 12701 , 14109 , 15426 , 16562 , 17438 , 18000 , 18221 , 18088 , 17966 , 17858 , 17826 , 18165 , 18049 , 18004 , 17923 , 17775 , 17733 , 17948 , 17792 , 17592 , 16712 , 15679 , 14559 , 13404 , 12254 , 11546 , 11463 , 11381 , 11298 , 11217 , 11135 , 11054 , 10973 , 10955 , 11602 , 11580 , 11498 , 11415 , 11334 , 11252 , 11171 , 11090 , 11009 , 10929 , 10850 , 10770 , 10691 , 10613 , 10534 , 10456 , 10379 , 10302 , 10225 , 10149 , 10073 , 9997.7 , 9922.6 , 9847.9 , 9773.6 , 9699.7 , 9626.3 , 9553.2 , 9480.5 , 9408.2 , 9336.3 , 9264.9 , 9193.8 , 9171.2 , 9264.4 , 9256.8 , 9186.5 , 9116.7 , 9047.2 , 8978.1 , 8909.4 , 8841.1 , 8773.3 , 8705.8 , 8638.8 , 8572.1 , 8505.9 , 8440 , 8374.6 , 8309.6 , 8245 , 8180.7 , 8116.9 , 8053.5 , 7990.5 , 7927.9 , 7865.6 , 7803.8 , 7742.4 , 7681.3 , 7620.7 , 7560.4 , 7500.6 , 7441.1 , 7382 , 7323.3 , 7265 , 7207.1 , 7149.5 , 7129.6 , 7252.1 , 7194.8 , 7137.9 , 7081.3 , 7025.1 , 6969.3 , 6913.9 , 6858.9 , 6804.2 , 6749.9 , 6695.9 , 6642.4 , 6589.2 , 6536.3 , 6483.9 , 6431.7 , 6380 , 6328.6 , 6277.6 , 6226.9 , 6176.5 , 6126.6 , 6076.9 , 6027.6 , 5978.7 , 5930 , 5881.7 , 5833.8 , 5786.2 , 5738.9 , 5692 , 5645.4 , 5599.1 , 5553.1 , 5507.5 , 5462.2 , 5417.2 , 5372.6 , 5328.2 , 5284.2 , 5240.2 , 5196.6 , 5153.4 , 5110.4 , 5067.7 , 5025.4 , 4983.3 , 4941.6 , 4900.2 , 4859 , 4818.2 , 4777.7 , 4737.5 , 4697.6 , 4657.9 , 4618.6 , 4579.6 , 4540.9 , 4502.4 , 4464.3 , 4426.4 , 4388.8 , 4351.5 , 4314.4 , 4277.7 , 4241.1 , 4199.8 , 4158.9 , 4118.4 , 4078.2 , 4038.5 , 3999 , 3959.9 , 3921 , 3882.6 , 3844.5 , 3806.8 , 3769.5 , 3732.5 , 3695.9 , 3659.6 , 3623.7 , 3588.1 , 3552.9 , 3518.1 , 3483.5 , 3449.4 , 3415.5 , 3382 , 3348.8 , 3316 , 3283.5 , 3251.3 , 3219.4 , 3187.8 , 3156.6 , 3125.7 , 3095 , 3064.7 , 3034.7 , 3005.1 , 2975.7 , 2946.6 , 2917.8 , 2889.3 , 2861.1 , 2833 , 2805.3 , 2777.8 , 2750.6 , 2723.7 , 2695.8 , 2667.9 , 2640.3 , 2613 , 2585.9 , 2559.2 , 2532.8 , 2506.6 , 2480.8 , 2455.2 , 2430 , 2404.1 , 2378.3 , 2351.8 , 2325.6 , 2299.8 , 2274.2 , 2249 , 2224.1 , 2199.4 , 2175.1 , 2150.9 , 2126.4 , 2102.2 , 2078.3 , 2054.7 , 2031.4 , 2008.4 , 1985.7 , 1963.3 , 1941.2 , 1919.3 , 1897.8 , 1876.4 , 1855.4 , 1834.6 , 1814.1 , 1793.9 , 1773.9 , 1754.1 , 1734.6 , 1715.3 , 1696.3 , 1677.5 , 1659 , 1640.6 , 1622.5 , 1604.7 , 1587 , 1569.6 , 1552.3 , 1535.3 , 1518.5 , 1501.9 , 1485.5 , 1469.4 , 1453.4 , 1437.6 , 1422 , 1406.5 , 1391.3 , 1376.3 , 1361.4 , 1346.7 , 1332.2 , 1317.9 , 1303.7 , 1289.8 , 1276 , 1262.3 , 1248.8 , 1235.5 , 1222.3 , 1209.3 , 1196.5 , 1183.8 , 1171.2 , 1158.8 , 1146.6 , 1134.5 , 1122.5 , 1110.7 , 1099 , 1087.5 , 1076.1 , 1064.8 , 1053.7 , 1042.7 , 1031.8 , 1021 , 1010.4 , 999.88 , 989.5 , 979.24 , 969.1 , 959.07 , 949.17 , 939.38 , 929.7 , 920.13 , 910.68 , 901.22 , 891.79 , 882.47 , 873.25 , 866.55 , 2625.6 , 2620.4 , 2589 , 2558 , 2527.3 , 2497.1 , 2467.2 , 2437.7 , 2408.5 , 2403.6 , 3522 , 3491.9 , 3449.5 , 3407.7 , 3366.4 , 3325.6 , 3285.3 , 3245.4 , 3206.1 , 3167.3 , 3129 , 3091.1 , 3053.7 , 3016.8 , 2980.3 , 2944.3 , 2908.7 , 2873.6 , 2838.9 , 2804.6 , 2770.8 , 2737.3 , 2704.2 , 2671.5 , 2639.2 , 2607.4 , 2575.9 , 2544.8 , 2514.1 , 2483.7 , 2453.8 , 2424.2 , 2395 , 2366.1 , 2337.6 , 2309.4 , 2281.6 , 2258.6 , 2254.2 , 2628.4 , 2603.5 , 2572.1 , 2541.2 , 2510.6 , 2480.4 , 2450.5 , 2421 , 2391.9 , 2362.8 , 2334.1 , 2306.5 , 2279.4 , 2252.6 , 2226.2 , 2200.2 , 2174.5 , 2148.8 , 2123.5 , 2098.4 , 2073.7 , 2049.3 , 2025.2 , 2001.4 , 1977.8 , 1954.6 , 1930.3 , 1912.7 , 1905.9 , 2024.8 , 2006.7 , 1981.2 , 1956.1 , 1931.2 , 1906.7 , 1882.4 , 1858.5 , 1834.9 , 1811.5 , 1788.7 , 1766.3 , 1744.3 , 1722.6 , 1701.2 , 1680.1 , 1659.2 , 1644.8 , 1638.7 , 1699.8 , 1690 , 1669.5 , 1649.2 , 1629.3 , 1609.6 , 1590.1 , 1570.9 , 1552 , 1533.2 , 1514.6 , 1496.3 , 1478.2 , 1460.3 , 1442.7 , 1425.3 , 1408.2 , 1391.2 , 1374.5 , 1358 , 1341.8 , 1325.7 , 1309.8 , 1294.2 , 1278.8 , 1263.5 , 1248.4 , 1233.6 , 1218.9 , 1204.4 , 1190.1 , 1176 , 1162.1 , 1148.3 , 1134.7 , 1121.3 , 1108 , 1094.8 , 1081.7 , 1068.8 , 1056 , 1043.4 , 1031 , 1018.7 , 1006.6 , 994.59 , 982.75 , 971.07 , 959.52 , 948.12 , 936.86 , 925.64 , 914.24 , 902.99 , 891.85 , 880.85 , 870 , 859.29 , 848.72 , 838.08 , 827.54 , 817.13 , 806.86 , 796.73 , 786.74 , 776.88 , 767.14 , 757.54 , 748.07 , 738.72 , 729.49 , 720.39 , 711.41 , 702.55 , 693.8 , 685.17 , 676.66 , 668.26 , 659.97 , 651.78 , 643.71 , 635.74 , 627.88 , 620.13 , 612.47 , 604.92 , 597.46 , 590.1 , 582.84 , 575.68 , 568.61 , 561.63 , 554.75 , 547.96 , 541.25 , 534.63 , 527.94 , 521.34 , 514.82 , 508.4 , 502.05 , 495.79 , 489.62 , 483.53 , 477.51 , 471.58 , 465.73 , 459.95 , 454.25 , 448.62 , 443.07 , 437.6 , 432.19 , 426.86 , 421.6 , 416.4 , 411.28 , 406.2 , 401.17 , 396.22 , 391.33 , 386.5 , 381.74 , 376.99 , 372.31 , 367.68 , 363.12 , 358.62 , 354.16 , 349.75 , 345.41 , 341.12 , 336.89 , 332.72 , 328.6 , 324.53 , 320.52 , 316.56 , 312.66 , 308.8 , 304.98 , 301.21 , 297.48 , 293.81 , 290.19 , 286.61 , 283.08 , 279.59 , 276.15 , 272.76 , 269.41 , 266.11 , 262.84 , 259.62 , 256.45 , 253.31 , 250.22 , 247.16 , 244.15 , 241.17 , 238.23 , 235.33 , 232.47 , 229.65 , 226.86 , 224.11 , 221.39 , 218.71 , 216.07 , 213.46 , 210.88 , 208.33 , 205.82 , 203.34 , 200.89 , 198.47 , 196.09 , 193.73 , 191.41 , 189.11 , 186.85 , 184.61 , 182.4 , 180.22 , 178.07 , 175.95 , 173.85 , 171.78 , 169.74 , 167.72 , 165.72 , 163.76 , 142.93 , 120.03 , 100.73 , 84.743 , 79.101 , 76.07 , 75.275 , 190.52 , 183.1 , 183.03 , 152.89 , 140.51 , 134.84 , 133.38 , 182.5 , 177.5 , 175.35 , 173.12 , 166.16 , 164.38 , 186.94 , 180.05 , 172.27 , 145.34 , 122.51 , 103.27 , 86.941 , 73.215 , 61.696 , 52.028 , 43.843 , 36.978 , 30.964 , 25.846 , 21.601 , 18.074 , 15.143 , 12.704 , 10.672 , 8.9773 , 7.5616 , 6.3736 , 5.373 , 4.5347 , 3.8211 , 3.2203 , 2.7185 , 2.2988 , 2.266 , 2.1819 , 2.1603 , 9.0422 , 8.7057 , 8.2833 , 7.0399 , 5.9483 , 4.9898 , 4.1923 , 3.5323 , 2.9792 , 2.5156 , 2.1267 , 1.8005 , 1.526 , 1.2934 , 1.0985 , 0.93506 , 0.79788 , 0.6827 , 0.5859 , 0.50449 , 0.43593 , 0.37814 , 0.32935 , 0.28811 , 0.25407 , 0.22546 , 0.20093 , 0.17985])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Pt_attenuation_interpolator = interp1d(energy_Pt_attenuation,mass_coefficient_Pt_attenuation*Ptdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Re_attenuation = np.array([0.005235171,0.005313308,0.005941418,0.006032358,0.006056608,0.006092984,0.006183924,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.03109002,0.03323523,0.033908,0.034427,0.0345654,0.034773,0.035292,0.03552846,0.03797993,0.039788,0.040397,0.0405594,0.04060054,0.040803,0.041412,0.04340198,0.044688,0.045372,0.0455544,0.045828,0.04639671,0.046512,0.04959809,0.05302035,0.05667876,0.06058959,0.06477028,0.06923942,0.07401695,0.07912411,0.081144,0.082386,0.0827172,0.083214,0.084456,0.08458368,0.09041995,0.09665893,0.1033284,0.1104581,0.1180797,0.1262272,0.1349368,0.1442475,0.1542005,0.1648404,0.1762144,0.1883732,0.2013709,0.2152655,0.2301188,0.245997,0.254996,0.258899,0.2599398,0.261501,0.2629708,0.265404,0.268226,0.2723315,0.2734263,0.2750685,0.279174,0.2811158,0.3005128,0.3212482,0.3434143,0.3671099,0.3924405,0.4195189,0.435512,0.442178,0.4439556,0.446622,0.4484657,0.453288,0.4794098,0.5,0.5025,0.5050125,0.50753756,0.51007525,0.51262563,0.51518875,0.51717493,0.5177647,0.51862505,0.52035352,0.52295529,0.52557007,0.52819792,0.53083891,0.5334931,0.53616057,0.53884137,0.54153558,0.54424325,0.54696447,0.54969929,0.55244779,0.55521003,0.55798608,0.56077601,0.56357989,0.56639779,0.56922978,0.57207593,0.5749363,0.57781099,0.58070004,0.58360354,0.58652156,0.58945417,0.59240144,0.59536345,0.59834026,0.60133196,0.60433862,0.60736032,0.61039712,0.6134491,0.61651635,0.61959893,0.62269693,0.62404375,0.62581041,0.62595625,0.62893946,0.63208416,0.63524458,0.6384208,0.64161291,0.64482097,0.64804508,0.6512853,0.65454173,0.65781444,0.66110351,0.66440903,0.66773107,0.67106973,0.67442508,0.6777972,0.68118619,0.68459212,0.68801508,0.69145515,0.69491243,0.69838699,0.70187893,0.70538832,0.70891526,0.71245984,0.71602214,0.71960225,0.72320026,0.72681626,0.73045034,0.7341026,0.73777311,0.74146197,0.74516928,0.74889513,0.75263961,0.7564028,0.76018482,0.76398574,0.76780567,0.7716447,0.77550292,0.77938044,0.78327734,0.78719373,0.79112969,0.79508534,0.79906077,0.80305607,0.80707135,0.81110671,0.81516224,0.81923806,0.82333425,0.82745092,0.83158817,0.83574611,0.83992484,0.84412447,0.84834509,0.85258682,0.85684975,0.861134,0.86543967,0.86976687,0.8741157,0.87848628,0.88287871,0.8872931,0.89172957,0.89618822,0.90066916,0.9051725,0.90969837,0.91424686,0.91881809,0.92341218,0.92802924,0.93266939,0.93733274,0.9420194,0.9467295,0.95146315,0.95622046,0.96100156,0.96580657,0.9706356,0.97548878,0.98036623,0.98526806,0.9901944,0.99514537,1.0001211,1.0051217,1.0101473,1.015198,1.020274,1.0253754,1.0305023,1.0356548,1.0408331,1.0460372,1.0512674,1.0565238,1.0618064,1.0671154,1.072451,1.0778132,1.0832023,1.0886183,1.0940614,1.0995317,1.1050294,1.1105545,1.1161073,1.1216878,1.1272963,1.1329328,1.1385974,1.1442904,1.1500119,1.1557619,1.1615407,1.1673484,1.1731852,1.1790511,1.1849464,1.1908711,1.1968254,1.2028096,1.2088236,1.2148677,1.2209421,1.2270468,1.233182,1.2393479,1.2455447,1.2517724,1.2580312,1.2643214,1.270643,1.2769962,1.2833812,1.2897981,1.2962471,1.3027283,1.309242,1.3157882,1.3223671,1.328979,1.3356239,1.342302,1.3490135,1.3557586,1.3625374,1.36935,1.3761968,1.3830778,1.3899932,1.3969431,1.4039278,1.4109475,1.4180022,1.4250922,1.4322177,1.4393788,1.4465757,1.4538086,1.4610776,1.468383,1.4757249,1.4831035,1.490519,1.4979716,1.5054615,1.5129888,1.5205537,1.5281565,1.5357973,1.5434763,1.5511937,1.5589496,1.5667444,1.5745781,1.582451,1.5903633,1.5983151,1.6063066,1.6143382,1.6224099,1.6305219,1.6386745,1.6468679,1.6551022,1.6633777,1.6716946,1.6800531,1.6884534,1.6968956,1.7053801,1.713907,1.7224766,1.7310889,1.7397444,1.7484431,1.7571853,1.7659712,1.7748011,1.7836751,1.7925935,1.8015565,1.8105642,1.8196171,1.8287151,1.8378587,1.847048,1.8562833,1.8655647,1.8748925,1.8825065,1.8832935,1.884267,1.8936883,1.9031567,1.9126725,1.9222359,1.9318471,1.9415063,1.9481516,1.9496484,1.9512138,1.9609699,1.9707747,1.9806286,1.9905318,2.0004844,2.0104868,2.0205393,2.030642,2.0407952,2.0509992,2.0612542,2.0715604,2.0819182,2.0923278,2.1027895,2.1133034,2.1238699,2.1344893,2.1451617,2.1558875,2.166667,2.1775003,2.1883878,2.1993297,2.2103264,2.221378,2.2324849,2.2436473,2.2548656,2.2661399,2.2774706,2.2888579,2.3003022,2.3118037,2.3233628,2.3349796,2.3466545,2.3583878,2.3649801,2.36962,2.3701797,2.3820306,2.3939407,2.4059104,2.41794,2.4300297,2.4421798,2.4543907,2.4666627,2.478996,2.491391,2.5038479,2.5163672,2.528949,2.5415938,2.5543017,2.5670732,2.5799086,2.5928082,2.6057722,2.6188011,2.6318951,2.6450545,2.6582798,2.6715712,2.6780604,2.6849291,2.6851398,2.6983537,2.7118455,2.7254047,2.7390317,2.7527269,2.7664905,2.780323,2.7942246,2.8081957,2.8222367,2.8363479,2.8505296,2.8647823,2.8791062,2.8935017,2.9079692,2.9225091,2.9258366,2.9371216,2.9375634,2.9518072,2.9665662,2.9813991,2.9963061,3.0112876,3.026344,3.0414758,3.0566831,3.0719666,3.0873264,3.102763,3.1182768,3.1338682,3.1495376,3.1652853,3.1811117,3.1970172,3.2130023,3.2290673,3.2452127,3.2614387,3.2777459,3.2941347,3.3106053,3.3271584,3.3437941,3.3605131,3.3773157,3.3942023,3.4111733,3.4282291,3.4453703,3.4625971,3.4799101,3.4973097,3.5147962,3.5323702,3.5500321,3.5677822,3.5856211,3.6035492,3.621567,3.6396748,3.6578732,3.6761626,3.6945434,3.7130161,3.7315812,3.7502391,3.7689903,3.7878352,3.8067744,3.8258083,3.8449373,3.864162,3.8834828,3.9029002,3.9224147,3.9420268,3.9617369,3.9815456,4.0014533,4.0214606,4.0415679,4.0617757,4.0820846,4.102495,4.1230075,4.1436226,4.1643407,4.1851624,4.2060882,4.2271186,4.2482542,4.2694955,4.290843,4.3122972,4.3338587,4.355528,4.3773056,4.3991921,4.4211881,4.443294,4.4655105,4.4878381,4.5102772,4.5328286,4.5554928,4.5782702,4.6011616,4.6241674,4.6472882,4.6705247,4.6938773,4.7173467,4.7409334,4.7646381,4.7884613,4.8124036,4.8364656,4.8606479,4.8849512,4.9093759,4.9339228,4.9585924,4.9833854,5.0083023,5.0333438,5.0585105,5.0838031,5.1092221,5.1347682,5.1604421,5.1862443,5.2121755,5.2382364,5.2644276,5.2907497,5.3172034,5.3437895,5.3705084,5.3973609,5.4243477,5.4514695,5.4787268,5.5061205,5.5336511,5.5613193,5.5891259,5.6170716,5.6451569,5.6733827,5.7017496,5.7302584,5.7589096,5.7877042,5.8166427,5.8457259,5.8749546,5.9043293,5.933851,5.9635202,5.9933378,6.0233045,6.053421,6.0836882,6.1141066,6.1446771,6.1754005,6.2062775,6.2373089,6.2684954,6.2998379,6.3313371,6.3629938,6.3948088,6.4267828,6.4589167,6.4912113,6.5236674,6.5562857,6.5890671,6.6220125,6.6551225,6.6883981,6.7218401,6.7554493,6.7892266,6.8231727,6.8572886,6.891575,6.9260329,6.9606631,6.9954664,7.0304437,7.0655959,7.1009239,7.1364285,7.1721107,7.2079712,7.2440111,7.2802311,7.3166323,7.3532155,7.3899815,7.4269314,7.4640661,7.5013864,7.5388934,7.5765878,7.6144708,7.6525431,7.6908058,7.7292599,7.7679062,7.8067457,7.8457794,7.8850083,7.9244334,7.9640555,8.0038758,8.0438952,8.0841147,8.1245352,8.1651579,8.2059837,8.2470136,8.2882487,8.3296899,8.3713384,8.4131951,8.455261,8.4975373,8.540025,9.030794,9.653919,10.32004,10.32459,10.48262,10.52476,10.58798,10.74601,11.03212,11.71953,11.79334,11.89891,11.94674,12.01849,12.19787,12.27617,12.46407,12.51417,12.58933,12.60708,12.77723,13.47697,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,70.24287,71.31802,71.45536,71.60472,72.03478,73.10993,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Re_attenuation = np.array([12574,12380,11100,11011,10992,50183,49391,72464,79935,87210,93764,99067,102640,104140,103380,100380,95333,88613,80669,71997,63066,54267,45970,38422,31754,29930,28604,28261,56187,54035,53089,44443,39056,37365,36924,41471,40847,39026,33631,30574,29084,28704,38917,37638,37387,31553,26528,22170,18547,15680,13540,12064,11178,10981,10896,10878,11042,10999,10996,11076,11536,12320,13356,14587,15935,17316,18652,19872,20907,21643,22030,22073,21801,21255,20487,19984,19745,19680,20269,20177,20023,19843,19578,19507,19903,19632,19503,18220,16914,15626,14382,13197,12084,11492,11260,11199,11866,11803,11639,10814,10230,10162,10095,10028,9961.9,9896,9830.3,9780,9765.1,9900.4,9857.2,9792.7,9728.6,9664.8,9601.3,9538.3,9475.5,9413.2,9351.1,9289.5,9228.1,9167.1,9106.5,9046.1,8986.1,8926.5,8867.1,8808.1,8749.4,8691.1,8633,8575.3,8517.9,8460.7,8403.9,8347.4,8291.2,8235.3,8179.7,8124.4,8069.4,8014.7,7960.2,7906.1,7852.2,7798.7,7745.4,7722.4,7884.9,7882.4,7831.5,7778.4,7725.7,7673.2,7620.9,7569,7517.3,7465.8,7414.7,7363.7,7313.1,7262.7,7212.6,7162.7,7113.1,7063.8,7014.7,6965.9,6917.3,6869,6821,6773.2,6725.6,6678.3,6631.2,6584.4,6537.8,6491.5,6445.4,6399.5,6353.9,6308.5,6263.4,6218.5,6173.8,6129.4,6085.1,6040.8,5996.8,5950.1,5899.7,5849.6,5799.9,5750.5,5701.5,5652.7,5604.4,5556.3,5508.6,5461.2,5414.2,5367.5,5321.1,5275.1,5229.4,5184,5138.9,5094.2,5049.8,5005.8,4962,4918.6,4875.6,4832.8,4790.4,4748.3,4706.5,4664.9,4623.7,4582.7,4542.1,4501.8,4461.8,4422.1,4382.8,4343.7,4305,4266.5,4228.4,4190.5,4153,4115.8,4078.9,4042.3,4005.9,3969.9,3934.2,3898.8,3863.6,3828.8,3794.3,3760,3726,3692.2,3653.7,3615.6,3578,3540.7,3503.9,3467.4,3431.4,3395.8,3358.2,3321.1,3284.4,3248.1,3212.2,3176.8,3141.7,3107.2,3073,3039.3,3004.7,2970.4,2936.5,2903,2870,2837.4,2805.2,2773.4,2742,2711,2680.4,2650.1,2620.3,2590.8,2561.7,2533,2504.6,2476.5,2448.9,2421.5,2394.5,2367.9,2341.5,2315.5,2289.9,2264.5,2239.4,2214.7,2190.3,2166.1,2142.3,2118.7,2095.5,2072.5,2049.8,2027.4,2005.2,1983.4,1961.8,1940.4,1919.3,1898.5,1877.9,1857.6,1837.5,1817.7,1798.1,1778.7,1759.6,1740.7,1722,1703.6,1685.3,1667.3,1649.5,1632,1614.6,1597.4,1580.5,1563.7,1547.1,1530.8,1514.6,1498.6,1482.8,1467.2,1451.8,1436.6,1421.5,1406.6,1391.9,1377.3,1363,1348.8,1334.7,1320.8,1307.1,1293.6,1280.2,1266.9,1253.8,1240.9,1228.1,1215.4,1202.9,1190.6,1178.4,1166.3,1154.4,1142.6,1130.9,1119.4,1108,1096.7,1085.6,1074.6,1063.7,1052.8,1041.9,1031.1,1020.5,1010,999.58,989.3,979.14,969.09,959.16,949.34,941.44,3077.3,3073.5,3036.8,3000.7,2964.9,2929.6,2894.7,2860.3,2836.9,4191.7,4183.5,4133.1,4083.2,4034,3985.4,3937.4,3890,3843.2,3797,3751.3,3706.2,3661.6,3617.6,3574,3530.9,3488.4,3446.4,3405,3364,3323.5,3283.5,3244.1,3205.1,3166.6,3128.5,3090.9,3053.8,3017.2,2981,2945.2,2909.9,2875,2840.5,2806.5,2772.9,2739.6,2706.8,2674.4,2642.4,2624.7,3062.3,3060.5,3023,2985.9,2949.3,2913.2,2877.5,2842.3,2807.1,2772,2737.3,2704,2671.7,2640,2608.7,2577.5,2546.6,2516.2,2486.2,2456.6,2427.4,2398.6,2370.2,2342.1,2314.5,2287.1,2274,2415.7,2415.2,2386.6,2357.8,2329.4,2301.4,2273.7,2246.4,2219.4,2192.7,2166.4,2141,2116.4,2092.2,2068.4,2044.9,2021.7,1998.8,1976.2,1971.1,2040.6,2039.9,2017.7,1995.2,1972.8,1950.8,1927.8,1904.8,1882.1,1859.6,1837.5,1815.4,1793.6,1772,1750.7,1729.6,1708.8,1688.3,1668,1647.9,1628.1,1608.5,1589.1,1570,1551.1,1532.4,1513.9,1495.7,1477.7,1459.9,1442.3,1424.9,1407.7,1390.7,1374,1357.4,1341,1324.8,1308.8,1293,1277.4,1262,1246.7,1231.7,1216.8,1202.2,1187.7,1173.4,1159.2,1145.3,1131.5,1117.8,1103.9,1090.2,1076.6,1063.3,1050.1,1037.1,1024.2,1011.2,998.4,985.78,973.33,961.05,948.94,937,925.21,913.58,902.12,890.8,879.64,868.63,857.77,847.06,836.49,826.06,815.77,805.62,795.61,785.73,775.98,766.37,756.88,747.52,738.29,729.17,720.18,711.31,702.56,693.93,685.41,677,668.7,660.52,652.44,644.47,636.61,628.79,620.91,613.15,605.49,597.93,590.47,583.11,575.8,568.59,561.48,554.46,547.53,540.7,533.96,527.31,520.75,514.28,507.86,501.53,495.29,489.13,483.05,477.05,471.13,465.29,459.51,453.73,448.03,442.4,436.85,431.38,425.98,420.65,415.39,410.2,405.08,400.03,395.04,390.12,385.27,380.48,375.76,371.09,366.48,361.91,357.4,352.95,348.56,344.23,339.95,335.73,331.56,327.46,323.4,319.4,315.45,311.55,307.7,303.91,300.16,296.46,292.82,289.21,285.66,282.15,278.69,275.27,271.9,268.57,265.29,262.05,258.85,255.69,252.57,249.49,246.45,243.46,240.5,237.58,234.69,231.85,229.04,226.26,223.53,220.82,218.16,215.52,212.92,210.36,207.83,205.32,202.86,200.42,198.01,195.64,193.29,190.98,188.69,186.44,184.21,182.01,179.84,177.7,175.58,173.49,171.43,169.39,167.38,165.39,163.43,161.46,159.52,157.6,155.71,153.83,151.98,150.14,148.32,146.51,127.47,106.87,89.759,89.653,86.112,85.202,219.41,210.78,196.33,166.64,163.81,159.88,158.15,215.82,207.37,203.83,195.66,193.57,220.15,219.38,212.07,185.27,156.12,131.52,110.67,93.095,78.355,65.997,55.597,46.812,39.438,33.207,27.764,23.181,19.38,16.224,13.599,11.414,9.5927,8.0731,6.7935,5.7221,4.819,4.0506,3.4104,2.8754,2.5362,2.4411,2.4293,2.4166,10.467,10.094,9.0518,7.6181,6.4196,5.4207,4.5586,3.837,3.2327,2.7267,2.3027,1.9473,1.6465,1.393,1.1809,1.0033,0.85453,0.72982,0.62519,0.53733,0.46349,0.40135,0.34899,0.30479,0.26873,0.23799,0.2117,0.18916,0.16979])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Re_attenuation_interpolator = interp1d(energy_Re_attenuation,mass_coefficient_Re_attenuation*Redensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV



energy_W_attenuation = np.array([0.0061305,0.006222,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.03109002,0.032928,0.03323523,0.033432,0.0335664,0.033768,0.034272,0.034888,0.035422,0.03552846,0.0355644,0.03577,0.035778,0.0363175,0.0364635,0.0366825,0.03723,0.03797993,0.04060054,0.04340198,0.045864,0.04639671,0.046566,0.0467532,0.047034,0.047736,0.04959809,0.05302035,0.05667876,0.06058959,0.06477028,0.06923942,0.07401695,0.075558,0.0767145,0.0770229,0.0774855,0.078642,0.07912411,0.08458368,0.09041995,0.09665893,0.1,0.1005,0.1010025,0.10150751,0.10201505,0.10252513,0.10303775,0.10355294,0.1040707,0.10459106,0.10511401,0.10563958,0.10616778,0.10669862,0.10723211,0.10776827,0.10830712,0.10884865,0.10939289,0.10993986,0.11048956,0.11104201,0.11159722,0.1121552,0.11271598,0.11327956,0.11384596,0.11441519,0.11498726,0.1155622,0.11614001,0.11672071,0.11730431,0.11789083,0.11848029,0.11907269,0.11966805,0.12026639,0.12086772,0.12147206,0.12207942,0.12268982,0.12330327,0.12391979,0.12453939,0.12516208,0.12578789,0.12641683,0.12704892,0.12768416,0.12832258,0.12896419,0.12960902,0.13025706,0.13090835,0.13156289,0.1322207,0.13288181,0.13354621,0.13421395,0.13488502,0.13555944,0.13623724,0.13691842,0.13760302,0.13829103,0.13898249,0.1396774,0.14037579,0.14107766,0.14178305,0.14249197,0.14320443,0.14392045,0.14464005,0.14536325,0.14609007,0.14682052,0.14755462,0.14829239,0.14903386,0.14977903,0.15052792,0.15128056,0.15203696,0.15279715,0.15356113,0.15432894,0.15510058,0.15587609,0.15665547,0.15743875,0.15822594,0.15901707,0.15981215,0.16061121,0.16141427,0.16222134,0.16303245,0.16384761,0.16466685,0.16549018,0.16631763,0.16714922,0.16798497,0.16882489,0.16966902,0.17051736,0.17136995,0.1722268,0.17308793,0.17395337,0.17482314,0.17569726,0.17657574,0.17745862,0.17834591,0.17923764,0.18013383,0.1810345,0.18193967,0.18284937,0.18376362,0.18468244,0.18560585,0.18653388,0.18746655,0.18840388,0.1893459,0.19029263,0.19124409,0.19220031,0.19316131,0.19412712,0.19509776,0.19607325,0.19705361,0.19803888,0.19902907,0.20002422,0.20102434,0.20202946,0.20303961,0.20405481,0.20507508,0.20610046,0.20713096,0.20816661,0.20920745,0.21025348,0.21130475,0.21236128,0.21342308,0.2144902,0.21556265,0.21664046,0.21772366,0.21881228,0.21990634,0.22100588,0.2221109,0.22322146,0.22433757,0.22545925,0.22658655,0.22771948,0.22885808,0.23000237,0.23115238,0.23230814,0.23346969,0.23463703,0.23581022,0.23698927,0.23817422,0.23936509,0.24056191,0.24176472,0.24297355,0.24418841,0.24521055,0.24540936,0.24558945,0.2466364,0.24786959,0.24910893,0.25035448,0.25160625,0.25286428,0.2541286,0.25539925,0.25667624,0.25795962,0.25859581,0.25900419,0.25924942,0.26054567,0.2618484,0.26315764,0.26447343,0.26579579,0.26712477,0.2684604,0.2698027,0.27115171,0.27250747,0.27387001,0.27523936,0.27661556,0.27799863,0.27938863,0.28078557,0.2821895,0.28360044,0.28501845,0.28644354,0.28787576,0.28931514,0.29076171,0.29221552,0.2936766,0.29514498,0.29662071,0.29810381,0.29959433,0.3010923,0.30259776,0.30411075,0.3056313,0.30715946,0.30869526,0.31023873,0.31178993,0.31334888,0.31491562,0.3164902,0.31807265,0.31966301,0.32126133,0.32286764,0.32448197,0.32610438,0.32773491,0.32937358,0.33102045,0.33267555,0.33433893,0.33601062,0.33769068,0.33937913,0.34107602,0.3427814,0.34449531,0.34621779,0.34794888,0.34968862,0.35143706,0.35319425,0.35496022,0.35673502,0.3585187,0.36031129,0.36211285,0.36392341,0.36574303,0.36757174,0.3694096,0.37125665,0.37311293,0.3749785,0.37685339,0.37873766,0.38063135,0.3825345,0.38444718,0.38636941,0.38830126,0.39024276,0.39219398,0.39415495,0.39612572,0.39810635,0.40009688,0.40209737,0.40410785,0.40612839,0.40815904,0.41019983,0.41225083,0.41431208,0.41638364,0.41846556,0.42055789,0.42266068,0.42477398,0.42477688,0.42582312,0.42689785,0.42903234,0.4311775,0.43333339,0.43550006,0.43767756,0.43986595,0.44206528,0.4442756,0.44649698,0.44872947,0.45097311,0.45322798,0.45549412,0.45777159,0.46006045,0.46236075,0.46467255,0.46699592,0.4693309,0.47167755,0.47403594,0.47640612,0.47878815,0.48118209,0.483588,0.48600594,0.48843597,0.49087815,0.4909216,0.49227841,0.49333254,0.4957992,0.4982782,0.50076959,0.50327344,0.5057898,0.50831875,0.51086035,0.51341465,0.51598172,0.51856163,0.52115444,0.52376021,0.52637901,0.52901091,0.53165596,0.53431424,0.53698581,0.53967074,0.5423691,0.54508094,0.54780635,0.55054538,0.5532981,0.5560646,0.55884492,0.56163914,0.56444734,0.56726958,0.57010592,0.57295645,0.57582123,0.57870034,0.58159384,0.58450181,0.58742432,0.59036144,0.59331325,0.59410753,0.59589253,0.59627982,0.59926122,0.60225752,0.60526881,0.60829515,0.61133663,0.61439331,0.61746528,0.6205526,0.62365537,0.62677364,0.62990751,0.63305705,0.63622234,0.63940345,0.64260046,0.64581347,0.64904253,0.65228775,0.65554919,0.65882693,0.66212107,0.66543167,0.66875883,0.67210262,0.67546314,0.67884045,0.68223466,0.68564583,0.68907406,0.69251943,0.69598202,0.69946194,0.70295924,0.70647404,0.71000641,0.71355644,0.71712423,0.72070985,0.7243134,0.72793496,0.73157464,0.73523251,0.73890867,0.74260322,0.74631623,0.75004781,0.75379805,0.75756704,0.76135488,0.76516165,0.76898746,0.7728324,0.77669656,0.78058004,0.78448294,0.78840536,0.79234738,0.79630912,0.80029067,0.80429212,0.80831358,0.81235515,0.81641693,0.82049901,0.8246015,0.82872451,0.83286813,0.83703248,0.84121764,0.84542373,0.84965084,0.8538991,0.85816859,0.86245944,0.86677173,0.87110559,0.87546112,0.87983843,0.88423762,0.88865881,0.8931021,0.89756761,0.90205545,0.90656573,0.91109856,0.91565405,0.92023232,0.92483348,0.92945765,0.93410494,0.93877546,0.94346934,0.94818668,0.95292762,0.95769226,0.96248072,0.96729312,0.97212959,0.97699023,0.98187519,0.98678456,0.99171848,0.99667708,1.0016605,1.0066688,1.0117021,1.0167606,1.0218444,1.0269536,1.0320884,1.0372489,1.0424351,1.0476473,1.0528855,1.0581499,1.0634407,1.0687579,1.0741017,1.0794722,1.0848695,1.0902939,1.0957454,1.1012241,1.1067302,1.1122639,1.1178252,1.1234143,1.1290314,1.1346765,1.1403499,1.1460517,1.1517819,1.1575408,1.1633285,1.1691452,1.1749909,1.1808659,1.1867702,1.192704,1.1986676,1.2046609,1.2106842,1.2167376,1.2228213,1.2289354,1.2350801,1.2412555,1.2474618,1.2536991,1.2599676,1.2662674,1.2725988,1.2789618,1.2853566,1.2917833,1.2982423,1.3047335,1.3112571,1.3178134,1.3244025,1.3310245,1.3376796,1.344368,1.3510899,1.3578453,1.3646345,1.3714577,1.378315,1.3852066,1.3921326,1.3990933,1.4060887,1.4131192,1.4201848,1.4272857,1.4344221,1.4415942,1.4488022,1.4560462,1.4633265,1.4706431,1.4779963,1.4853863,1.4928132,1.5002773,1.5077787,1.5153176,1.5228942,1.5305086,1.5381612,1.545852,1.5535812,1.5613491,1.5691559,1.5770017,1.5848867,1.5928111,1.6007752,1.608779,1.6168229,1.624907,1.6330316,1.6411967,1.6494027,1.6576497,1.665938,1.6742677,1.682639,1.6910522,1.6995075,1.708005,1.716545,1.7251278,1.7337534,1.7424222,1.7511343,1.7598899,1.7686894,1.7775328,1.7864205,1.7953526,1.8043294,1.8088491,1.809551,1.813351,1.8224178,1.8315299,1.8406875,1.8498909,1.8591404,1.8684361,1.8709394,1.8722607,1.8777783,1.8871672,1.896603,1.906086,1.9156165,1.9251945,1.9348205,1.9444946,1.9542171,1.9639882,1.9738081,1.9836772,1.9935955,2.0035635,2.0135813,2.0236492,2.0337675,2.0439363,2.054156,2.0644268,2.0747489,2.0851227,2.0955483,2.106026,2.1165562,2.1271389,2.1377746,2.1484635,2.1592058,2.1700018,2.1808519,2.1917561,2.2027149,2.2137285,2.2247971,2.2359211,2.2471007,2.2583362,2.2696279,2.2786505,2.280976,2.2833493,2.2923809,2.3038428,2.315362,2.3269388,2.3385735,2.3502664,2.3620177,2.3738278,2.385697,2.3976254,2.4096136,2.4216616,2.4337699,2.4459388,2.4581685,2.4704593,2.4828116,2.4952257,2.5077018,2.5202403,2.5328415,2.5455057,2.5582333,2.5710244,2.5716555,2.5781443,2.5838796,2.596799,2.6097829,2.6228319,2.635946,2.6491257,2.6623714,2.6756832,2.6890617,2.702507,2.7160195,2.7295996,2.7432476,2.7569638,2.7707486,2.7846024,2.7985254,2.812518,2.8140455,2.8251547,2.8265806,2.8407135,2.8549171,2.8691917,2.8835376,2.8979553,2.9124451,2.9270073,2.9416424,2.9563506,2.9711323,2.985988,3.0009179,3.0159225,3.0310021,3.0461571,3.0613879,3.0766949,3.0920783,3.1075387,3.1230764,3.1386918,3.1543853,3.1701572,3.186008,3.201938,3.2179477,3.2340374,3.2502076,3.2664587,3.282791,3.2992049,3.3157009,3.3322794,3.3489408,3.3656856,3.382514,3.3994265,3.4164237,3.4335058,3.4506733,3.4679267,3.4852663,3.5026927,3.5202061,3.5378072,3.5554962,3.5732737,3.59114,3.6090957,3.6271412,3.6452769,3.6635033,3.6818208,3.7002299,3.7187311,3.7373247,3.7560114,3.7747914,3.7936654,3.8126337,3.8316969,3.8508554,3.8701096,3.8894602,3.9089075,3.928452,3.9480943,3.9678347,3.9876739,4.055024,4.334821,4.633924,4.953664,5.295467,5.660855,6.051453,6.469004,6.915365,7.392525,7.902609,8.44789,9.030794,9.653919,10.00266,10.15577,10.19659,10.25783,10.32004,10.41094,11.03212,11.31312,11.48628,11.53246,11.60172,11.77488,11.79334,11.8578,12.0393,12.0877,12.1603,12.3418,12.60708,13.47697,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,68.1345,69.17738,69.45548,69.87263,70.9155,71.45536,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_W_attenuation = np.array([47191,46237,60149,66397,72486,77928,82236,84987,85881,84786,81755,77013,70918,63904,56419,48864,41553,34775,28709,24120,23434,23006,22719,25354,24172,22813,21707,21494,21422,53789,53753,51413,50802,51821,49573,46708,38241,30999,25971,25028,24738,24424,36197,34788,31454,26559,22375,18932,16286,14416,13255,13025,12892,12862,13002,12920,12894,12877,13288,14041,14529,14605,14683,14762,14842,14923,15004,15087,15171,15256,15342,15428,15516,15604,15693,15782,15873,15964,16055,16147,16240,16333,16427,16521,16616,16711,16806,16902,16999,17096,17194,17292,17390,17488,17586,17684,17782,17880,17978,18076,18174,18272,18369,18467,18564,18661,18757,18853,18949,19044,19139,19234,19328,19421,19514,19607,19698,19790,19880,19970,20059,20147,20234,20320,20404,20488,20570,20651,20731,20810,20887,20964,21038,21112,21184,21255,21325,21392,21459,21523,21586,21648,21707,21765,21821,21876,21929,21980,22029,22076,22122,22166,22208,22249,22287,22324,22359,22393,22424,22454,22482,22508,22533,22555,22576,22595,22613,22628,22642,22654,22665,22673,22680,22685,22689,22691,22691,22689,22686,22681,22675,22667,22657,22646,22633,22618,22602,22584,22565,22545,22523,22499,22474,22447,22419,22390,22359,22327,22293,22258,22222,22185,22146,22106,22064,22022,21978,21933,21887,21839,21791,21741,21690,21638,21586,21532,21477,21421,21364,21306,21247,21187,21126,21065,21002,20939,20875,20810,20744,20677,20610,20542,20473,20404,20334,20263,20191,20119,20046,19973,19911,20881,20869,20794,20702,20610,20518,20424,20330,20236,20141,20045,19949,19901,20577,20557,20453,20348,20243,20137,20031,19925,19819,19712,19605,19498,19391,19284,19178,19071,18965,18858,18752,18646,18539,18434,18328,18223,18118,18014,17910,17806,17703,17600,17497,17395,17293,17192,17091,16990,16889,16789,16689,16590,16491,16392,16294,16196,16098,16001,15904,15807,15711,15615,15520,15425,15330,15236,15142,15049,14956,14864,14771,14680,14588,14497,14407,14317,14227,14138,14049,13961,13873,13786,13699,13612,13526,13440,13355,13270,13186,13102,13019,12936,12853,12771,12690,12609,12528,12448,12368,12289,12210,12132,12054,11977,11900,11823,11747,11671,11596,11522,11447,11374,11300,11300,12037,11999,11925,11851,11777,11704,11632,11560,11488,11417,11346,11275,11205,11136,11067,10998,10930,10862,10794,10727,10660,10594,10528,10463,10397,10333,10268,10204,10141,10077,10076,10206,10179,10116,10054,9992.1,9930.5,9869.3,9808.4,9747.9,9687.6,9627.8,9568.2,9509,9450.1,9391.5,9333.2,9275.2,9217.5,9160.2,9103.1,9046.4,8989.9,8933.8,8877.9,8822.3,8767,8712,8657.2,8602.8,8548.6,8494.6,8441,8387.6,8334.4,8281.6,8229,8176.6,8124.5,8072.7,8058.8,8232.4,8225.6,8173.5,8121.7,8070.2,8018.9,7967.8,7917,7866.4,7816.1,7766,7716.1,7666.5,7612.9,7557.4,7502.3,7447.5,7393,7338.9,7285,7231.5,7178.2,7125.3,7072.7,7020.3,6968.3,6916.5,6865,6813.8,6762.7,6708.2,6654.1,6600.4,6547,6493.9,6441.2,6388.9,6336.8,6285.2,6233.6,6182.2,6131.2,6080.5,6030.1,5980.1,5930.3,5881,5831.9,5783.2,5734.8,5686.8,5639,5591.6,5544.5,5497.8,5451.3,5405.2,5359.4,5314,5268.8,5224,5179.5,5135.3,5091.4,5047.9,5004.6,4961.7,4919,4876.7,4834.5,4792.7,4751.2,4710,4669.1,4628.5,4588.1,4548.1,4508.4,4469,4429.9,4391.1,4352.5,4314.3,4276.3,4238.7,4201.3,4164.2,4127.5,4091,4054.7,4018.8,3983.1,3947.8,3912.7,3877.9,3843.3,3809.1,3775.1,3741.4,3707.9,3674.8,3641.9,3609.3,3576.9,3544.8,3510.5,3471.5,3432.8,3394.7,3357,3319.8,3283,3246,3208.6,3171.6,3135.1,3099,3063.5,3028.3,2993.7,2959.4,2925.6,2892.3,2859.3,2826.8,2794.7,2763,2731.7,2700.8,2670.2,2640.1,2610.4,2581,2552,2523.3,2495,2467.1,2439.5,2412.3,2385.4,2358.8,2332.6,2306.7,2281.1,2255.8,2230.9,2206.2,2181.9,2157.8,2134.1,2110.6,2087.5,2064.6,2042,2019.7,1997.6,1975.8,1954.3,1933.1,1912.1,1891.3,1870.8,1850.6,1830.6,1810.9,1791.3,1772.1,1753,1734.2,1715.6,1697.3,1679.1,1661.2,1643.5,1626,1608.7,1591.6,1574.7,1558,1541.5,1525.2,1509.1,1493.2,1477.5,1462,1446.6,1431.4,1416.4,1401.6,1387,1372.5,1358.2,1344,1330.1,1316.3,1302.6,1289.1,1275.8,1262.6,1249.6,1236.7,1224,1211.4,1199,1186.7,1174.6,1162.5,1150.7,1138.9,1127.3,1115.9,1104.5,1093.1,1081.8,1070.6,1059.5,1048.5,1037.7,1027,1016.4,1006,995.61,985.39,975.28,970.25,3252,3235.5,3196.8,3158.5,3120.7,3083.4,3046.5,3010.1,3000.4,4448.1,4416.2,4362.7,4309.9,4257.7,4206.2,4155.3,4105.1,4055.4,4006.4,3958,3910.1,3862.8,3816,3769.8,3724.2,3679.2,3634.7,3590.8,3547.4,3504.5,3462.2,3420.3,3379.1,3338.3,3298,3258.2,3218.9,3180.1,3141.8,3103.9,3066.5,3029.6,2993.1,2957.1,2921.5,2886.3,2851.6,2817.3,2783.4,2756.8,2750,3227.1,3194.6,3153.9,3113.8,3074.2,3035.1,2996.5,2957.9,2919.6,2881.7,2845.2,2810.7,2776.8,2743.3,2710,2677.2,2644.9,2613,2581.7,2550.8,2520.3,2490.3,2460.7,2431.6,2402.8,2401.4,2555.1,2541.1,2510,2479.3,2449.1,2419.3,2389.9,2360.9,2332.2,2303.9,2276.1,2249.8,2223.9,2198.4,2173.3,2148.5,2124.1,2100,2076.2,2073.7,2147.4,2145,2121.1,2097.5,2074.1,2051.1,2028.3,2005.7,1983.5,1961.4,1939.6,1918.1,1896.9,1875.8,1853.2,1830.9,1808.9,1787.1,1765.6,1744.3,1723.3,1702.5,1682,1661.8,1641.8,1622,1602.4,1583.1,1564,1545.2,1526.5,1508.1,1490,1472,1454.2,1436.7,1419.3,1402.1,1385.2,1368.4,1351.9,1335.5,1319.4,1303.4,1287.7,1272.1,1256.8,1241.6,1226.7,1211.9,1197.3,1182.6,1167.9,1153.4,1139.1,1124.9,1111,1097.2,1083.4,1069.7,1056.1,1042.8,1029.6,1016.6,1003.8,991.18,978.71,966.41,954.28,942.32,930.51,892.04,754.92,639.56,540.34,456.66,386,326.52,276.66,234.81,199.64,169.96,144.32,122.28,102.53,93.374,89.669,88.717,234.78,230.28,223.89,187.71,174.87,167.65,165.8,226.62,217.71,216.79,213.63,205.1,202.9,231,222.6,211.12,178.63,150.64,126.91,106.71,89.718,75.468,63.534,53.466,45.023,37.899,31.9,26.671,22.27,18.621,15.589,13.067,10.969,9.2195,7.757,6.5249,5.4934,4.6175,3.8819,3.2685,2.7569,2.6266,2.5279,2.5025,10.964,10.571,10.376,8.7689,7.3878,6.2249,5.2412,4.4076,3.7099,3.1258,2.6366,2.2268,1.8832,1.5903,1.3455,1.1408,0.96937,0.82584,0.70555,0.60466,0.51997,0.44879,0.38891,0.33845,0.2965,0.26163,0.23188,0.20643,0.18463,0.16588])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_W_attenuation_interpolator = interp1d(energy_W_attenuation,mass_coefficient_W_attenuation*Wdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Os_attenuation = np.array([0.006058082,0.006148501,0.006911593,0.007017383,0.007045593,0.007087909,0.007193699,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.03109002,0.03323523,0.03552846,0.03797993,0.04060054,0.04340198,0.044492,0.045173,0.045374,0.045627,0.0460685,0.0462537,0.046308,0.04639671,0.0465315,0.047226,0.04959809,0.05302035,0.05667876,0.05684,0.05771,0.057942,0.05829,0.05916,0.06058959,0.06477028,0.06923942,0.07401695,0.07912411,0.082026,0.0832815,0.0836163,0.0841185,0.08458368,0.085374,0.09041995,0.09665893,0.1033284,0.1104581,0.1180797,0.1262272,0.1349368,0.1442475,0.1542005,0.1648404,0.1762144,0.1883732,0.2013709,0.2152655,0.2301188,0.245997,0.2629708,0.267344,0.271436,0.2725272,0.274164,0.278256,0.2811158,0.283612,0.287953,0.2891106,0.290847,0.295188,0.3005128,0.3212482,0.3434143,0.3671099,0.3924405,0.4195189,0.4484657,0.458836,0.465859,0.4677318,0.470541,0.477564,0.4794098,0.5,0.5025,0.5050125,0.50753756,0.51007525,0.51262563,0.51518875,0.5177647,0.52035352,0.52295529,0.52557007,0.52819792,0.53083891,0.5334931,0.53616057,0.53884137,0.54153558,0.54424325,0.54571853,0.54696447,0.54728152,0.54969929,0.55244779,0.55521003,0.55798608,0.56077601,0.56357989,0.56639779,0.56922978,0.57207593,0.5749363,0.57781099,0.58070004,0.58360354,0.58652156,0.58945417,0.59240144,0.59536345,0.59834026,0.60133196,0.60433862,0.60736032,0.61039712,0.6134491,0.61651635,0.61959893,0.62269693,0.62581041,0.62893946,0.63208416,0.63524458,0.6384208,0.64161291,0.64482097,0.64804508,0.6512853,0.65327272,0.65454173,0.65532723,0.65781444,0.66110351,0.66440903,0.66773107,0.67106973,0.67442508,0.6777972,0.68118619,0.68459212,0.68801508,0.69145515,0.69491243,0.69838699,0.70187893,0.70538832,0.70891526,0.71245984,0.71602214,0.71960225,0.72320026,0.72681626,0.73045034,0.7341026,0.73777311,0.74146197,0.74516928,0.74889513,0.75263961,0.7564028,0.76018482,0.76398574,0.76780567,0.7716447,0.77550292,0.77938044,0.78327734,0.78719373,0.79112969,0.79508534,0.79906077,0.80305607,0.80707135,0.81110671,0.81516224,0.81923806,0.82333425,0.82745092,0.83158817,0.83574611,0.83992484,0.84412447,0.84834509,0.85258682,0.85684975,0.861134,0.86543967,0.86976687,0.8741157,0.87848628,0.88287871,0.8872931,0.89172957,0.89618822,0.90066916,0.9051725,0.90969837,0.91424686,0.91881809,0.92341218,0.92802924,0.93266939,0.93733274,0.9420194,0.9467295,0.95146315,0.95622046,0.96100156,0.96580657,0.9706356,0.97548878,0.98036623,0.98526806,0.9901944,0.99514537,1.0001211,1.0051217,1.0101473,1.015198,1.020274,1.0253754,1.0305023,1.0356548,1.0408331,1.0460372,1.0512674,1.0565238,1.0618064,1.0671154,1.072451,1.0778132,1.0832023,1.0886183,1.0940614,1.0995317,1.1050294,1.1105545,1.1161073,1.1216878,1.1272963,1.1329328,1.1385974,1.1442904,1.1500119,1.1557619,1.1615407,1.1673484,1.1731852,1.1790511,1.1849464,1.1908711,1.1968254,1.2028096,1.2088236,1.2148677,1.2209421,1.2270468,1.233182,1.2393479,1.2455447,1.2517724,1.2580312,1.2643214,1.270643,1.2769962,1.2833812,1.2897981,1.2962471,1.3027283,1.309242,1.3157882,1.3223671,1.328979,1.3356239,1.342302,1.3490135,1.3557586,1.3625374,1.36935,1.3761968,1.3830778,1.3899932,1.3969431,1.4039278,1.4109475,1.4180022,1.4250922,1.4322177,1.4393788,1.4465757,1.4538086,1.4610776,1.468383,1.4757249,1.4831035,1.490519,1.4979716,1.5054615,1.5129888,1.5205537,1.5281565,1.5357973,1.5434763,1.5511937,1.5589496,1.5667444,1.5745781,1.582451,1.5903633,1.5983151,1.6063066,1.6143382,1.6224099,1.6305219,1.6386745,1.6468679,1.6551022,1.6633777,1.6716946,1.6800531,1.6884534,1.6968956,1.7053801,1.713907,1.7224766,1.7310889,1.7397444,1.7484431,1.7571853,1.7659712,1.7748011,1.7836751,1.7925935,1.8015565,1.8105642,1.8196171,1.8287151,1.8378587,1.847048,1.8562833,1.8655647,1.8748925,1.884267,1.8936883,1.9031567,1.9126725,1.9222359,1.9318471,1.9415063,1.9512138,1.959659,1.9605411,1.9609699,1.9707747,1.9806286,1.9905318,2.0004844,2.0104868,2.0205393,2.0299512,2.030642,2.031649,2.0407952,2.0509992,2.0612542,2.0715604,2.0819182,2.0923278,2.1027895,2.1133034,2.1238699,2.1344893,2.1451617,2.1558875,2.166667,2.1775003,2.1883878,2.1993297,2.2103264,2.221378,2.2324849,2.2436473,2.2548656,2.2661399,2.2774706,2.2888579,2.3003022,2.3118037,2.3233628,2.3349796,2.3466545,2.3583878,2.3701797,2.3820306,2.3939407,2.4059104,2.41794,2.4300297,2.4421798,2.4543907,2.454905,2.4594951,2.4666627,2.478996,2.491391,2.5038479,2.5163672,2.528949,2.5415938,2.5543017,2.5670732,2.5799086,2.5928082,2.6057722,2.6188011,2.6318951,2.6450545,2.6582798,2.6715712,2.6849291,2.6983537,2.7118455,2.7254047,2.7390317,2.7527269,2.7664905,2.780323,2.7883189,2.7942246,2.7960812,2.8081957,2.8222367,2.8363479,2.8505296,2.8647823,2.8791062,2.8935017,2.9079692,2.9225091,2.9371216,2.9518072,2.9665662,2.9813991,2.9963061,3.0112876,3.026344,3.0414758,3.0422811,3.054719,3.0566831,3.0719666,3.0873264,3.102763,3.1182768,3.1338682,3.1495376,3.1652853,3.1811117,3.1970172,3.2130023,3.2290673,3.2452127,3.2614387,3.2777459,3.2941347,3.3106053,3.3271584,3.3437941,3.3605131,3.3773157,3.3942023,3.4111733,3.4282291,3.4453703,3.4625971,3.4799101,3.4973097,3.5147962,3.5323702,3.5500321,3.5677822,3.5856211,3.6035492,3.621567,3.6396748,3.6578732,3.6761626,3.6945434,3.7130161,3.7315812,3.7502391,3.7689903,3.7878352,3.8067744,3.8258083,3.8449373,3.864162,3.8834828,3.9029002,3.9224147,3.9420268,3.9617369,3.9815456,4.0014533,4.0214606,4.0415679,4.0617757,4.0820846,4.102495,4.1230075,4.1436226,4.1643407,4.1851624,4.2060882,4.2271186,4.2482542,4.2694955,4.290843,4.3122972,4.3338587,4.355528,4.3773056,4.3991921,4.4211881,4.443294,4.4655105,4.4878381,4.5102772,4.5328286,4.5554928,4.5782702,4.6011616,4.6241674,4.6472882,4.6705247,4.6938773,4.7173467,4.7409334,4.7646381,4.7884613,4.8124036,4.8364656,4.8606479,4.8849512,4.9093759,4.9339228,4.9585924,4.9833854,5.0083023,5.0333438,5.0585105,5.0838031,5.1092221,5.1347682,5.1604421,5.1862443,5.2121755,5.2382364,5.2644276,5.2907497,5.3172034,5.3437895,5.3705084,5.3973609,5.4243477,5.4514695,5.4787268,5.5061205,5.5336511,5.5613193,5.5891259,5.6170716,5.6451569,5.6733827,5.7017496,5.7302584,5.7589096,5.7877042,5.8166427,5.8457259,5.8749546,5.9043293,5.933851,5.9635202,5.9933378,6.0233045,6.053421,6.0836882,6.1141066,6.1446771,6.1754005,6.2062775,6.2373089,6.2684954,6.2998379,6.3313371,6.3629938,6.3948088,6.4267828,6.4589167,6.4912113,6.5236674,6.5562857,6.5890671,6.6220125,6.6551225,6.6883981,6.7218401,6.7554493,6.7892266,6.8231727,6.8572886,6.891575,6.9260329,6.9606631,6.9954664,7.0304437,7.0655959,7.1009239,7.1364285,7.1721107,7.2079712,7.2440111,7.2802311,7.3166323,7.3532155,7.3899815,7.4269314,7.4640661,7.5013864,7.5388934,7.5765878,7.6144708,7.6525431,7.6908058,7.7292599,7.7679062,7.8067457,7.8457794,7.8850083,7.9244334,7.9640555,8.0038758,8.0438952,8.0841147,8.1245352,8.1651579,8.2059837,8.2470136,8.2882487,8.3296899,8.3713384,8.4131951,8.455261,8.4975373,8.540025,9.030794,9.653919,10.32004,10.65348,10.81655,10.86003,10.92525,11.03212,11.08832,11.79334,12.1373,12.32308,12.37262,12.44693,12.60708,12.6327,12.70864,12.90316,12.95503,13.03284,13.22736,13.47697,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,71.45536,72.39338,73.50144,73.79693,74.24015,75.34821,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451,3,3.02,3.03,3.05,3.06,3.08,3.09,3.11,3.12,3.14,3.15,3.17,3.19,3.2,3.22,3.23,3.25,3.27,3.28,3.3,3.32,3.33,3.35,3.37,3.38,3.4,3.42,3.43,3.45,3.47,3.49,3.5,3.52,3.54,3.56,3.57,3.59,3.61,3.63,3.65,3.66,3.68,3.7,3.72,3.74,3.76,3.77,3.79,3.81,3.83,3.85,3.87,3.89,3.91,3.93,3.95,3.97,3.99,4.06,4.33,4.63,4.95,5.3,5.66,6.05,6.47,6.92,7.39,7.9,8.45,9.03,9.65,10,10.2,10.2,10.3,10.3,10.4,11,11.3,11.5,11.5,11.6,11.8,11.8,11.9,12,12.1,12.2,12.3,12.6,13.5,14.4,15.4,16.5,17.6,18.8,20.1,21.5,23,24.6,26.3,28.1,30,32.1,34.3,36.7,39.2,41.9,44.8,47.9,51.2,54.7,58.5,62.5,66.8,68.1,69.2,69.5,69.9,70.9,71.5,76.4,81.7,87.3,93.3,99.8,107,114,122,130,139,149,159,170,182,194,208,222,237,254,271,290,310,332,354,379,405,433])*1e3	# eV
mass_coefficient_Os_attenuation = np.array([20049,19575,16452,16262,16222,48882,47807,59114,67564,77132,87344,97565,107030,114910,120440,123030,122350,118430,111600,102460,91742,80230,68657,57586,47467,38564,30967,24644,19486,17837,16892,16625,36179,35022,34551,38679,38427,38048,36171,30593,24331,19455,19275,18345,18109,25321,24325,22831,19311,16406,14038,12246,11519,11261,11197,11311,11231,11105,10550,10354,10586,11183,12086,13223,14517,15894,17258,18529,19633,20502,21036,21203,21025,20546,19809,19583,19362,19302,19746,19525,19368,19229,18984,18918,19205,18959,18655,17463,16228,14993,13787,12630,11538,11180,10947,10885,11525,11296,11238,10612,10540,10468,10397,10326,10255,10185,10115,10046,9976.9,9908.2,9839.9,9771.9,9704.4,9637.2,9570.4,9504,9437.9,9402.2,9520.2,9512.6,9454.9,9389.9,9325.3,9261.1,9197.2,9133.7,9070.5,9007.7,8945.3,8883.1,8821.4,8760,8698.9,8638.2,8577.8,8517.7,8458,8398.7,8339.6,8281,8222.6,8164.5,8106.8,8049.4,7992.4,7935.6,7879.2,7823,7767.2,7711.7,7656.6,7601.7,7547.1,7492.9,7438.9,7406.1,7566.2,7553.2,7512.3,7458.6,7405.3,7352.3,7299.6,7247.2,7195,7143.2,7091.7,7040.4,6989.5,6938.8,6888.4,6838.4,6788.6,6739.1,6689.9,6640.9,6592.3,6543.9,6495.8,6448,6400.5,6353.2,6306.2,6259.4,6213,6166.8,6120.8,6075.2,6029.8,5984.6,5939.7,5895.1,5850.8,5806.7,5762.9,5719.3,5675.7,5632.4,5589.3,5546.5,5503.9,5461.6,5419.5,5377.7,5336.1,5294.8,5253.8,5213,5172.4,5132.1,5092,5052.2,5012.6,4973.2,4933.8,4890.1,4846.8,4803.8,4761.1,4718.8,4676.8,4635.1,4593.7,4552.6,4511.9,4471.5,4431.4,4391.5,4352,4312.7,4273.8,4235.2,4197,4159,4121.4,4084,4047,4010.3,3973.9,3937.8,3902,3866.5,3831.2,3791.2,3751.5,3712.3,3673.6,3635.2,3597.3,3559.8,3522.7,3486,3449.7,3413.8,3378.3,3343.3,3308.6,3274.3,3240.4,3206.9,3172.7,3137.8,3103.3,3069.2,3035.5,3002.2,2969.3,2936.8,2904.7,2872.9,2841.6,2810.7,2779.2,2747.8,2716.7,2686,2655.7,2625.8,2596.3,2567.1,2538.3,2509.8,2481.8,2454,2426.6,2399.5,2372.8,2346.4,2320.3,2294.6,2269.2,2244.1,2219.2,2194.7,2170.5,2146.6,2123,2099.7,2076.6,2053.9,2031.4,2009.2,1987.3,1965.6,1944.2,1923,1902.1,1881.5,1861.1,1840.9,1821,1801.4,1781.9,1762.8,1743.8,1725.1,1706.5,1688.2,1670.2,1652.3,1634.7,1617.2,1600,1583,1566.2,1549.5,1533.1,1516.9,1500.8,1485,1469.3,1453.9,1438.6,1423.4,1408.5,1393.7,1379.1,1364.7,1350.4,1336.3,1322.4,1308.6,1295,1281.5,1268.2,1255.1,1242.1,1229.3,1216.6,1204,1191.6,1179.4,1167.3,1155.3,1143.4,1131.7,1120.2,1108.7,1097.4,1086.2,1075.2,1064.2,1053.4,1042.8,1032.2,1021.8,1011.4,1001.2,991.09,980.88,970.79,960.82,950.96,941.21,931.57,922.03,912.61,904.54,2890.2,2888.6,2854.3,2820.3,2786.8,2753.7,2721,2688.7,2658.9,2656.7,3920.1,3877.5,3830.8,3784.6,3739.1,3694.1,3649.6,3605.7,3562.3,3519.5,3477.2,3435.4,3394.1,3353.3,3313.1,3273.3,3233.9,3195,3156.5,3118.6,3081.1,3044.1,3007.5,2971.4,2935.7,2900.5,2865.7,2831.3,2797.3,2763.8,2730.6,2697.9,2665.6,2633.7,2602.1,2571,2540.2,2509.8,2479.8,2478.5,2889.5,2869,2834.2,2799.8,2765.8,2732.3,2699.1,2666.4,2633.7,2601.1,2569,2538.5,2508.5,2478.8,2449.6,2420.7,2391.8,2363.4,2335.4,2307.7,2280.4,2253.4,2226.8,2200.6,2174.7,2149.1,2134.5,2268.9,2265.2,2241.6,2214.7,2188.2,2161.9,2136,2110.5,2085.2,2060.2,2035.6,2011.8,1988.9,1966.3,1944.1,1922.1,1899.1,1876,1853.2,1852,1914.3,1911.4,1888.5,1866,1843.7,1821.7,1799.9,1778.4,1757.2,1736.2,1715.4,1694.7,1674.3,1654.1,1634.1,1614.4,1594.9,1575.6,1556.6,1537.8,1519.2,1500.9,1482.7,1464.8,1447.1,1429.6,1412.3,1395.3,1378.4,1361.7,1345.3,1329,1313,1297.1,1281.5,1266,1250.7,1235.7,1220.8,1206.1,1191.6,1177.2,1163.1,1149.1,1135.3,1121.7,1108.3,1095,1081.9,1068.9,1056.1,1043.5,1030.5,1017.8,1005.2,992.73,980.44,968.31,956.34,944.24,932.3,920.53,908.92,897.46,886.16,875.01,864.02,853.17,842.47,831.91,821.49,811.22,801.08,791.08,781.22,771.48,761.88,752.41,743.06,733.84,724.74,715.76,706.9,698.17,689.54,681.04,672.64,664.36,656.19,648.12,640.17,632.31,624.57,616.92,609.38,601.93,594.59,587.18,579.84,572.59,565.44,558.39,551.43,544.56,537.79,531.08,524.44,517.89,511.43,505.05,498.76,492.55,486.42,480.38,474.42,468.53,462.73,456.98,451.31,445.7,440.18,434.72,429.33,423.94,418.62,413.37,408.19,403.08,398.04,393.06,388.15,383.31,378.53,373.81,369.16,364.57,360.04,355.57,351.15,346.8,342.49,338.23,334.03,329.88,325.78,321.74,317.75,313.82,309.93,306.1,302.32,298.59,294.9,291.27,287.68,284.14,280.65,277.2,273.8,270.44,267.12,263.85,260.62,257.44,254.29,251.19,248.12,245.1,242.11,239.17,236.26,233.39,230.55,227.76,224.99,222.27,219.58,216.92,214.3,211.72,209.16,206.64,204.15,201.7,199.27,196.88,194.51,192.18,189.87,187.6,185.36,183.14,180.95,178.79,176.66,174.55,172.47,170.42,168.39,166.39,164.42,162.47,160.54,158.64,156.76,154.91,153.08,151.24,131.67,110.35,92.72,85.388,82.028,81.166,207.88,202.5,199.75,169.15,156.49,150.15,148.51,202.81,195.94,194.87,191.75,184.06,182.09,207.11,199.49,190.28,160.52,135.25,113.93,95.865,80.688,67.957,57.276,48.236,40.65,34.253,28.637,23.909,19.987,16.729,14.021,11.767,9.8888,8.3211,7.0082,5.902,4.9757,4.188,3.5252,2.9721,2.5099,2.4291,2.3383,2.3149,9.914,9.56,9.2447,7.7995,6.5674,5.5422,4.6714,3.932,3.3129,2.7943,2.3596,1.9952,1.689,1.4289,1.2112,1.0288,0.87601,0.74788,0.64035,0.55004,0.47412,0.41022,0.35636,0.31091,0.27324,0.24177,0.21487,0.1918,0.17197,1875.8,1853.2,1830.9,1808.9,1787.1,1765.6,1744.3,1723.3,1702.5,1682,1661.8,1641.8,1622,1602.4,1583.1,1564,1545.2,1526.5,1508.1,1490,1472,1454.2,1436.7,1419.3,1402.1,1385.2,1368.4,1351.9,1335.5,1319.4,1303.4,1287.7,1272.1,1256.8,1241.6,1226.7,1211.9,1197.3,1182.6,1167.9,1153.4,1139.1,1124.9,1111,1097.2,1083.4,1069.7,1056.1,1042.8,1029.6,1016.6,1003.8,991.18,978.71,966.41,954.28,942.32,930.51,892.04,754.92,639.56,540.34,456.66,386,326.52,276.66,234.81,199.64,169.96,144.32,122.28,102.53,93.374,89.669,88.717,234.78,230.28,223.89,187.71,174.87,167.65,165.8,226.62,217.71,216.79,213.63,205.1,202.9,231,222.6,211.12,178.63,150.64,126.91,106.71,89.718,75.468,63.534,53.466,45.023,37.899,31.9,26.671,22.27,18.621,15.589,13.067,10.969,9.2195,7.757,6.5249,5.4934,4.6175,3.8819,3.2685,2.7569,2.6266,2.5279,2.5025,10.964,10.571,10.376,8.7689,7.3878,6.2249,5.2412,4.4076,3.7099,3.1258,2.6366,2.2268,1.8832,1.5903,1.3455,1.1408,0.96937,0.82584,0.70555,0.60466,0.51997,0.44879,0.38891,0.33845,0.2965,0.26163,0.23188,0.20643,0.18463,0.16588])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Os_attenuation_interpolator = interp1d(energy_Os_attenuation,mass_coefficient_Os_attenuation*Osdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Ta_attenuation = np.array([0.0057285,0.005814,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.0245,0.024875,0.024975,0.025125,0.02545001,0.0255,0.02720606,0.02908327,0.03109002,0.03323523,0.03552846,0.035672,0.036218,0.0363636,0.036582,0.037128,0.03797993,0.04060054,0.04340198,0.044002,0.0446755,0.0448551,0.0451245,0.045798,0.04639671,0.04959809,0.05302035,0.05667876,0.06058959,0.06477028,0.06923942,0.069678,0.0707445,0.0710289,0.0714555,0.072522,0.07401695,0.07912411,0.08458368,0.09041995,0.09665893,0.1,0.1005,0.1010025,0.10150751,0.10201505,0.10252513,0.10303775,0.10355294,0.1040707,0.10459106,0.10511401,0.10563958,0.10616778,0.10669862,0.10723211,0.10776827,0.10830712,0.10884865,0.10939289,0.10993986,0.11048956,0.11104201,0.11159722,0.1121552,0.11271598,0.11327956,0.11384596,0.11441519,0.11498726,0.1155622,0.11614001,0.11672071,0.11730431,0.11789083,0.11848029,0.11907269,0.11966805,0.12026639,0.12086772,0.12147206,0.12207942,0.12268982,0.12330327,0.12391979,0.12453939,0.12516208,0.12578789,0.12641683,0.12704892,0.12768416,0.12832258,0.12896419,0.12960902,0.13025706,0.13090835,0.13156289,0.1322207,0.13288181,0.13354621,0.13421395,0.13488502,0.13555944,0.13623724,0.13691842,0.13760302,0.13829103,0.13898249,0.1396774,0.14037579,0.14107766,0.14178305,0.14249197,0.14320443,0.14392045,0.14464005,0.14536325,0.14609007,0.14682052,0.14755462,0.14829239,0.14903386,0.14977903,0.15052792,0.15128056,0.15203696,0.15279715,0.15356113,0.15432894,0.15510058,0.15587609,0.15665547,0.15743875,0.15822594,0.15901707,0.15981215,0.16061121,0.16141427,0.16222134,0.16303245,0.16384761,0.16466685,0.16549018,0.16631763,0.16714922,0.16798497,0.16882489,0.16966902,0.17051736,0.17136995,0.1722268,0.17308793,0.17395337,0.17482314,0.17569726,0.17657574,0.17745862,0.17834591,0.17923764,0.18013383,0.1810345,0.18193967,0.18284937,0.18376362,0.18468244,0.18560585,0.18653388,0.18746655,0.18840388,0.1893459,0.19029263,0.19124409,0.19220031,0.19316131,0.19412712,0.19509776,0.19607325,0.19705361,0.19803888,0.19902907,0.20002422,0.20102434,0.20202946,0.20303961,0.20405481,0.20507508,0.20610046,0.20713096,0.20816661,0.20920745,0.21025348,0.21130475,0.21236128,0.21342308,0.2144902,0.21556265,0.21664046,0.21772366,0.21881228,0.21990634,0.22100588,0.2221109,0.22322146,0.22433757,0.22545925,0.22658655,0.22771948,0.22885808,0.22912551,0.2294745,0.23000237,0.23115238,0.23230814,0.23346969,0.23463703,0.23581022,0.23698927,0.23817422,0.23936509,0.24056191,0.24111348,0.24148653,0.24176472,0.24297355,0.24418841,0.24540936,0.2466364,0.24786959,0.24910893,0.25035448,0.25160625,0.25286428,0.2541286,0.25539925,0.25667624,0.25795962,0.25924942,0.26054567,0.2618484,0.26315764,0.26447343,0.26579579,0.26712477,0.2684604,0.2698027,0.27115171,0.27250747,0.27387001,0.27523936,0.27661556,0.27799863,0.27938863,0.28078557,0.2821895,0.28360044,0.28501845,0.28644354,0.28787576,0.28931514,0.29076171,0.29221552,0.2936766,0.29514498,0.29662071,0.29810381,0.29959433,0.3010923,0.30259776,0.30411075,0.3056313,0.30715946,0.30869526,0.31023873,0.31178993,0.31334888,0.31491562,0.3164902,0.31807265,0.31966301,0.32126133,0.32286764,0.32448197,0.32610438,0.32773491,0.32937358,0.33102045,0.33267555,0.33433893,0.33601062,0.33769068,0.33937913,0.34107602,0.3427814,0.34449531,0.34621779,0.34794888,0.34968862,0.35143706,0.35319425,0.35496022,0.35673502,0.3585187,0.36031129,0.36211285,0.36392341,0.36574303,0.36757174,0.3694096,0.37125665,0.37311293,0.3749785,0.37685339,0.37873766,0.38063135,0.3825345,0.38444718,0.38636941,0.38830126,0.39024276,0.39219398,0.39415495,0.39612572,0.39810635,0.40009688,0.40209737,0.4039782,0.40410785,0.40502181,0.40612839,0.40815904,0.41019983,0.41225083,0.41431208,0.41638364,0.41846556,0.42055789,0.42266068,0.42477398,0.42689785,0.42903234,0.4311775,0.43333339,0.43550006,0.43767756,0.43986595,0.44206528,0.4442756,0.44649698,0.44872947,0.45097311,0.45322798,0.45549412,0.45777159,0.46006045,0.46236075,0.46412139,0.46467255,0.46547861,0.46699592,0.4693309,0.47167755,0.47403594,0.47640612,0.47878815,0.48118209,0.483588,0.48600594,0.48843597,0.49087815,0.49333254,0.4957992,0.4982782,0.50076959,0.50327344,0.5057898,0.50831875,0.51086035,0.51341465,0.51598172,0.51856163,0.52115444,0.52376021,0.52637901,0.52901091,0.53165596,0.53431424,0.53698581,0.53967074,0.5423691,0.54508094,0.54780635,0.55054538,0.5532981,0.5560646,0.55884492,0.56163914,0.56444734,0.56457826,0.56642179,0.56726958,0.57010592,0.57295645,0.57582123,0.57870034,0.58159384,0.58450181,0.58742432,0.59036144,0.59331325,0.59627982,0.59926122,0.60225752,0.60526881,0.60829515,0.61133663,0.61439331,0.61746528,0.6205526,0.62365537,0.62677364,0.62990751,0.63305705,0.63622234,0.63940345,0.64260046,0.64581347,0.64904253,0.65228775,0.65554919,0.65882693,0.66212107,0.66543167,0.66875883,0.67210262,0.67546314,0.67884045,0.68223466,0.68564583,0.68907406,0.69251943,0.69598202,0.69946194,0.70295924,0.70647404,0.71000641,0.71355644,0.71712423,0.72070985,0.7243134,0.72793496,0.73157464,0.73523251,0.73890867,0.74260322,0.74631623,0.75004781,0.75379805,0.75756704,0.76135488,0.76516165,0.76898746,0.7728324,0.77669656,0.78058004,0.78448294,0.78840536,0.79234738,0.79630912,0.80029067,0.80429212,0.80831358,0.81235515,0.81641693,0.82049901,0.8246015,0.82872451,0.83286813,0.83703248,0.84121764,0.84542373,0.84965084,0.8538991,0.85816859,0.86245944,0.86677173,0.87110559,0.87546112,0.87983843,0.88423762,0.88865881,0.8931021,0.89756761,0.90205545,0.90656573,0.91109856,0.91565405,0.92023232,0.92483348,0.92945765,0.93410494,0.93877546,0.94346934,0.94818668,0.95292762,0.95769226,0.96248072,0.96729312,0.97212959,0.97699023,0.98187519,0.98678456,0.99171848,0.99667708,1.0016605,1.0066688,1.0117021,1.0167606,1.0218444,1.0269536,1.0320884,1.0372489,1.0424351,1.0476473,1.0528855,1.0581499,1.0634407,1.0687579,1.0741017,1.0794722,1.0848695,1.0902939,1.0957454,1.1012241,1.1067302,1.1122639,1.1178252,1.1234143,1.1290314,1.1346765,1.1403499,1.1460517,1.1517819,1.1575408,1.1633285,1.1691452,1.1749909,1.1808659,1.1867702,1.192704,1.1986676,1.2046609,1.2106842,1.2167376,1.2228213,1.2289354,1.2350801,1.2412555,1.2474618,1.2536991,1.2599676,1.2662674,1.2725988,1.2789618,1.2853566,1.2917833,1.2982423,1.3047335,1.3112571,1.3178134,1.3244025,1.3310245,1.3376796,1.344368,1.3510899,1.3578453,1.3646345,1.3714577,1.378315,1.3852066,1.3921326,1.3990933,1.4060887,1.4131192,1.4201848,1.4272857,1.4344221,1.4415942,1.4488022,1.4560462,1.4633265,1.4706431,1.4779963,1.4853863,1.4928132,1.5002773,1.5077787,1.5153176,1.5228942,1.5305086,1.5381612,1.545852,1.5535812,1.5613491,1.5691559,1.5770017,1.5848867,1.5928111,1.6007752,1.608779,1.6168229,1.624907,1.6330316,1.6411967,1.6494027,1.6576497,1.665938,1.6742677,1.682639,1.6910522,1.6995075,1.708005,1.716545,1.7251278,1.7337534,1.7347877,1.7354123,1.7424222,1.7511343,1.7598899,1.7686894,1.7775328,1.7864205,1.7926172,1.7937828,1.7953526,1.8043294,1.813351,1.8224178,1.8315299,1.8406875,1.8498909,1.8591404,1.8684361,1.8777783,1.8871672,1.896603,1.906086,1.9156165,1.9251945,1.9348205,1.9444946,1.9542171,1.9639882,1.9738081,1.9836772,1.9935955,2.0035635,2.0135813,2.0236492,2.0337675,2.0439363,2.054156,2.0644268,2.0747489,2.0851227,2.0955483,2.106026,2.1165562,2.1271389,2.1377746,2.1484635,2.1592058,2.1700018,2.1808519,2.1917561,2.1963695,2.2027149,2.2137285,2.2247971,2.2359211,2.2471007,2.2583362,2.2696279,2.280976,2.2923809,2.3038428,2.315362,2.3269388,2.3385735,2.3502664,2.3620177,2.3738278,2.385697,2.3976254,2.4096136,2.4216616,2.4337699,2.4459388,2.4581685,2.4657375,2.4704593,2.4716624,2.4828116,2.4952257,2.5077018,2.5202403,2.5328415,2.5455057,2.5582333,2.5710244,2.5838796,2.596799,2.6097829,2.6228319,2.635946,2.6491257,2.6623714,2.6756832,2.6890617,2.702507,2.7027735,2.7132264,2.7160195,2.7295996,2.7432476,2.7569638,2.7707486,2.7846024,2.7985254,2.812518,2.8265806,2.8407135,2.8549171,2.8691917,2.8835376,2.8979553,2.9124451,2.9270073,2.9416424,2.9563506,2.9711323,2.985988,3.0009179,3.0159225,3.0310021,3.0461571,3.0613879,3.0766949,3.0920783,3.1075387,3.1230764,3.1386918,3.1543853,3.1701572,3.186008,3.201938,3.2179477,3.2340374,3.2502076,3.2664587,3.282791,3.2992049,3.3157009,3.3322794,3.3489408,3.3656856,3.382514,3.3994265,3.4164237,3.4335058,3.4506733,3.4679267,3.4852663,3.5026927,3.5202061,3.5378072,3.5554962,3.5732737,3.59114,3.6090957,3.6271412,3.6452769,3.6635033,3.6818208,3.7002299,3.7187311,3.7373247,3.7560114,3.7747914,3.7936654,3.8126337,3.8316969,3.8508554,3.8701096,3.8894602,3.9089075,3.928452,3.9480943,3.9678347,3.9876739,4.055024,4.334821,4.633924,4.953664,5.295467,5.660855,6.051453,6.469004,6.915365,7.392525,7.902609,8.44789,9.030794,9.653919,9.683478,9.831694,9.871219,9.930505,10.07872,10.32004,10.91338,11.03212,11.08042,11.12496,11.19178,11.35882,11.44787,11.62309,11.66982,11.73991,11.79334,11.91513,12.60708,13.47697,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.06807,66.84318,67.07932,67.34898,67.75348,68.76473,71.45536,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Ta_attenuation = np.array([47371,46396,55425,58905,61782,63767,64634,64246,62577,59707,55814,51143,45972,40582,35215,32961,31788,31481,37323,36071,35883,30020,24845,20628,17211,14483,14338,13811,13677,51572,49299,46036,37794,30936,29681,28358,28020,42452,40830,39472,33413,28442,24373,21288,19149,17850,17766,17588,17547,17668,17550,17436,17407,17804,18508,19411,19917,19992,20068,20143,20217,20292,20366,20440,20513,20586,20658,20730,20802,20872,20943,21013,21083,21152,21221,21288,21355,21421,21486,21551,21614,21677,21738,21799,21859,21918,21975,22032,22088,22143,22196,22249,22300,22351,22400,22448,22495,22541,22585,22629,22671,22712,22752,22791,22828,22864,22899,22933,22965,22996,23026,23055,23082,23108,23133,23156,23179,23200,23219,23238,23255,23270,23285,23298,23310,23321,23330,23338,23345,23350,23355,23358,23359,23360,23359,23357,23354,23349,23344,23337,23329,23319,23309,23297,23284,23270,23255,23239,23221,23203,23183,23162,23140,23117,23093,23068,23041,23014,22986,22956,22926,22894,22862,22828,22794,22759,22722,22685,22647,22608,22568,22527,22485,22442,22399,22355,22309,22264,22217,22169,22121,22072,22022,21972,21920,21868,21816,21762,21708,21654,21598,21542,21486,21429,21371,21312,21254,21194,21134,21074,21013,20951,20889,20826,20763,20700,20636,20572,20507,20442,20376,20310,20244,20177,20110,20043,19975,19907,19839,19770,19702,19632,19563,19547,21080,21035,20937,20840,20743,20647,20550,20454,20359,20263,20168,20124,21188,21161,21045,20930,20815,20701,20587,20475,20363,20251,20139,20021,19904,19788,19673,19559,19447,19335,19218,19101,18985,18871,18758,18646,18535,18425,18317,18209,18103,17997,17893,17789,17686,17584,17482,17382,17282,17183,17085,16987,16890,16794,16698,16603,16509,16415,16322,16229,16137,16045,15954,15864,15774,15685,15596,15508,15420,15333,15246,15160,15075,14990,14905,14821,14737,14654,14571,14489,14407,14326,14245,14165,14085,14006,13927,13849,13771,13693,13616,13540,13464,13388,13313,13238,13164,13090,13017,12944,12872,12800,12728,12657,12586,12516,12446,12377,12308,12239,12171,12104,12036,11969,11903,11837,11775,11771,12535,12499,12432,12366,12300,12235,12169,12105,12040,11976,11913,11850,11787,11724,11662,11601,11539,11478,11418,11357,11297,11238,11178,11119,11061,11002,10944,10887,10843,10829,10983,10946,10889,10811,10732,10654,10576,10499,10422,10346,10271,10196,10121,10047,9973.5,9900.5,9827.9,9755.9,9684.3,9613.2,9542.5,9472.4,9402.7,9333.5,9264.7,9196.4,9128.6,9061.1,8994.2,8927.7,8861.6,8795.9,8730.7,8665.9,8601.5,8537.5,8474,8410.8,8348.1,8285.8,8282.9,8460.3,8441.6,8379.3,8317.4,8255.9,8194.8,8134.1,8073.7,8013.8,7954.2,7895,7836.2,7777.7,7719.6,7661.8,7604.4,7547.4,7490.7,7434.4,7378.4,7322.8,7267.5,7212.5,7157.9,7103.6,7049.6,6996,6942.7,6889.7,6837,6784.6,6732.6,6680.8,6629.4,6578.3,6527.5,6477,6426.8,6376.9,6327.3,6277.9,6228.9,6180.2,6131.7,6083.6,6035.7,5988.1,5940.9,5893.9,5847.2,5800.7,5754.6,5708.8,5663.2,5617.9,5572.9,5528.2,5483.8,5439.6,5395.7,5352.1,5308.8,5265.7,5223,5180.4,5138.2,5096.2,5054.4,5012.9,4971.6,4930.5,4889.7,4849.2,4809,4769,4729.3,4689.8,4650.6,4611.6,4573,4534.5,4496.3,4458.4,4420.8,4383.3,4346.2,4309.3,4272.6,4236.2,4200.1,4164.2,4128.5,4093.1,4058,4023,3988.4,3954,3919.8,3885.9,3852.2,3818.7,3785.5,3752.5,3719.8,3687.2,3655,3622.9,3591.1,3559.5,3528.1,3497,3466.1,3435.5,3405.2,3375,3342.2,3303.7,3265.7,3228.2,3191.1,3154.5,3118.4,3082.8,3047.6,3012.8,2978.5,2944.6,2911.2,2878.1,2845.5,2813.3,2781.5,2750.1,2719.1,2688.4,2658.2,2628.3,2598.8,2569.7,2540.9,2512.5,2484.4,2456.7,2429.3,2402.3,2375.6,2349.2,2323.2,2297.4,2272,2246.9,2222.1,2197.6,2173.4,2149.5,2125.9,2102.6,2079.6,2056.8,2034.3,2012.1,1990.2,1968.5,1947.1,1925.9,1905,1884.4,1864,1843.8,1823.9,1804.3,1784.8,1765.6,1746.6,1727.9,1709.3,1691,1672.9,1655,1637.3,1619.9,1602.6,1585.6,1568.7,1552.1,1535.6,1519.3,1503.3,1487.4,1471.7,1456.2,1440.8,1425.7,1410.7,1395.9,1381.3,1366.8,1352.5,1338.4,1324.4,1310.6,1297,1283.5,1270.2,1257,1244,1231.1,1218.4,1205.8,1193.4,1181.1,1169,1156.8,1144.7,1132.7,1120.9,1109.3,1097.7,1086.3,1075.1,1063.9,1052.9,1042,1031.3,1020.6,1010.1,1008.9,3464.4,3430.5,3389.1,3348.2,3307.9,3268,3228.6,3201.6,4760.1,4749.8,4691.9,4634.6,4578.1,4522.3,4467.2,4412.8,4359,4305.8,4253.3,4201.4,4150.2,4099.5,4049.6,4000.2,3951.5,3903.3,3855.8,3808.8,3762.5,3716.7,3671.5,3626.8,3582.7,3539.2,3496.1,3453.7,3411.7,3370.3,3329.4,3288.9,3249,3209.6,3170.7,3132.2,3094.3,3056.8,3019.7,2983.2,2947,2911.4,3399.4,3374.6,3332.1,3290.2,3248.8,3207.9,3167.4,3126.9,3086.8,3047.3,3008.6,2972.2,2936.3,2900.7,2865.5,2830.8,2796.6,2762.9,2729.7,2697,2664.8,2633,2601.6,2570.7,2551.9,2719.2,2715.9,2685.8,2652.9,2620.5,2588.4,2556.9,2525.7,2495,2464.6,2434.7,2405.8,2377.8,2350.3,2323.2,2296.6,2270.3,2244.3,2218.7,2193.5,2193,2271.9,2266.6,2241.2,2216.2,2191.5,2167,2142.9,2119.1,2095.5,2072.2,2049.1,2026.4,2003.8,1981.5,1959.4,1937.6,1916,1894.7,1873.6,1852.7,1832.1,1811.6,1789.8,1768.2,1746.9,1725.8,1705,1684.4,1664.1,1644,1624.2,1604.6,1585.2,1566.1,1547.2,1528.5,1510,1491.7,1473.6,1455.8,1438.2,1420.8,1403.6,1386.6,1369.9,1353.3,1337,1320.9,1305,1289.2,1273.7,1257.9,1242.3,1226.9,1211.6,1196.6,1181.7,1167,1152.2,1137.6,1123.2,1109,1095,1081.2,1067.6,1054.1,1040.9,1027.8,1014.9,1002.2,989.62,977.24,965.02,952.96,941.07,929.34,917.76,906.34,895.08,858.36,727.29,614.45,519.39,438.66,370.85,313.91,266.17,226.05,192.3,163.33,138.66,117.52,98.456,97.651,93.759,92.759,246.83,235.31,218.05,185.42,179.91,177.74,175.79,240,230.59,225.81,216.83,214.54,244.24,241.49,235.39,204.59,172.72,145.56,122.42,102.88,86.49,72.761,61.242,51.526,43.374,36.481,30.707,25.676,21.442,17.929,15.012,12.586,10.565,8.8817,7.4661,6.2808,5.282,4.4341,3.7285,3.1397,2.7288,2.6494,2.626,2.5995,11.521,11.109,10.101,8.5192,7.1777,6.0451,5.0802,4.272,3.5958,3.0298,2.5557,2.1586,1.8231,1.5396,1.3026,1.1045,0.93878,0.80001,0.68375,0.58627,0.50444,0.4357,0.37786,0.32913,0.28923,0.25541,0.22655,0.20189,0.18074,0.16256])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Ta_attenuation_interpolator = interp1d(energy_Ta_attenuation,mass_coefficient_Ta_attenuation*Tadensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Hf_attenuation = np.array([0.005025,0.0051,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.016758,0.0170145,0.01705382,0.0170829,0.0171855,0.017442,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.029988,0.030447,0.0305694,0.030753,0.03109002,0.031212,0.03323523,0.03552846,0.037338,0.0379095,0.03797993,0.0380619,0.0382905,0.038862,0.04060054,0.04340198,0.04639671,0.04959809,0.05302035,0.05667876,0.06058959,0.063602,0.0645755,0.06477028,0.0648351,0.0652245,0.066198,0.06923942,0.07401695,0.07912411,0.08458368,0.09041995,0.09665893,0.1,0.1005,0.1010025,0.10150751,0.10201505,0.10252513,0.10303775,0.10355294,0.1040707,0.10459106,0.10511401,0.10563958,0.10616778,0.10669862,0.10723211,0.10776827,0.10830712,0.10884865,0.10939289,0.10993986,0.11048956,0.11104201,0.11159722,0.1121552,0.11271598,0.11327956,0.11384596,0.11441519,0.11498726,0.1155622,0.11614001,0.11672071,0.11730431,0.11789083,0.11848029,0.11907269,0.11966805,0.12026639,0.12086772,0.12147206,0.12207942,0.12268982,0.12330327,0.12391979,0.12453939,0.12516208,0.12578789,0.12641683,0.12704892,0.12768416,0.12832258,0.12896419,0.12960902,0.13025706,0.13090835,0.13156289,0.1322207,0.13288181,0.13354621,0.13421395,0.13488502,0.13555944,0.13623724,0.13691842,0.13760302,0.13829103,0.13898249,0.1396774,0.14037579,0.14107766,0.14178305,0.14249197,0.14320443,0.14392045,0.14464005,0.14536325,0.14609007,0.14682052,0.14755462,0.14829239,0.14903386,0.14977903,0.15052792,0.15128056,0.15203696,0.15279715,0.15356113,0.15432894,0.15510058,0.15587609,0.15665547,0.15743875,0.15822594,0.15901707,0.15981215,0.16061121,0.16141427,0.16222134,0.16303245,0.16384761,0.16466685,0.16549018,0.16631763,0.16714922,0.16798497,0.16882489,0.16966902,0.17051736,0.17136995,0.1722268,0.17308793,0.17395337,0.17482314,0.17569726,0.17657574,0.17745862,0.17834591,0.17923764,0.18013383,0.1810345,0.18193967,0.18284937,0.18376362,0.18468244,0.18560585,0.18653388,0.18746655,0.18840388,0.1893459,0.19029263,0.19124409,0.19220031,0.19316131,0.19412712,0.19509776,0.19607325,0.19705361,0.19803888,0.19902907,0.20002422,0.20102434,0.20202946,0.20303961,0.20405481,0.20507508,0.20610046,0.20713096,0.20816661,0.20920745,0.21025348,0.21130475,0.21236128,0.21342308,0.21354015,0.21385984,0.2144902,0.21556265,0.21664046,0.21772366,0.21881228,0.21990634,0.22100588,0.2221109,0.22322146,0.22361559,0.22398441,0.22433757,0.22545925,0.22658655,0.22771948,0.22885808,0.23000237,0.23115238,0.23230814,0.23346969,0.23463703,0.23581022,0.23698927,0.23817422,0.23936509,0.24056191,0.24176472,0.24297355,0.24418841,0.24540936,0.2466364,0.24786959,0.24910893,0.25035448,0.25160625,0.25286428,0.2541286,0.25539925,0.25667624,0.25795962,0.25924942,0.26054567,0.2618484,0.26315764,0.26447343,0.26579579,0.26712477,0.2684604,0.2698027,0.27115171,0.27250747,0.27387001,0.27523936,0.27661556,0.27799863,0.27938863,0.28078557,0.2821895,0.28360044,0.28501845,0.28644354,0.28787576,0.28931514,0.29076171,0.29221552,0.2936766,0.29514498,0.29662071,0.29810381,0.29959433,0.3010923,0.30259776,0.30411075,0.3056313,0.30715946,0.30869526,0.31023873,0.31178993,0.31334888,0.31491562,0.3164902,0.31807265,0.31966301,0.32126133,0.32286764,0.32448197,0.32610438,0.32773491,0.32937358,0.33102045,0.33267555,0.33433893,0.33601062,0.33769068,0.33937913,0.34107602,0.3427814,0.34449531,0.34621779,0.34794888,0.34968862,0.35143706,0.35319425,0.35496022,0.35673502,0.3585187,0.36031129,0.36211285,0.36392341,0.36574303,0.36757174,0.3694096,0.37125665,0.37311293,0.3749785,0.37685339,0.37873766,0.3799207,0.38063135,0.38087931,0.3825345,0.38444718,0.38636941,0.38830126,0.39024276,0.39219398,0.39415495,0.39612572,0.39810635,0.40009688,0.40209737,0.40410785,0.40612839,0.40815904,0.41019983,0.41225083,0.41431208,0.41638364,0.41846556,0.42055789,0.42266068,0.42477398,0.42689785,0.42903234,0.4311775,0.43333339,0.43550006,0.4363751,0.43762492,0.43767756,0.43986595,0.44206528,0.4442756,0.44649698,0.44872947,0.45097311,0.45322798,0.45549412,0.45777159,0.46006045,0.46236075,0.46467255,0.46699592,0.4693309,0.47167755,0.47403594,0.47640612,0.47878815,0.48118209,0.483588,0.48600594,0.48843597,0.49087815,0.49333254,0.4957992,0.4982782,0.50076959,0.50327344,0.5057898,0.50831875,0.51086035,0.51341465,0.51598172,0.51856163,0.52115444,0.52376021,0.52637901,0.52901091,0.53165596,0.53431424,0.53698581,0.53724443,0.53895558,0.53967074,0.5423691,0.54508094,0.54780635,0.55054538,0.5532981,0.5560646,0.55884492,0.56163914,0.56444734,0.56726958,0.57010592,0.57295645,0.57582123,0.57870034,0.58159384,0.58450181,0.58742432,0.59036144,0.59331325,0.59627982,0.59926122,0.60225752,0.60526881,0.60829515,0.61133663,0.61439331,0.61746528,0.6205526,0.62365537,0.62677364,0.62990751,0.63305705,0.63622234,0.63940345,0.64260046,0.64581347,0.64904253,0.65228775,0.65554919,0.65882693,0.66212107,0.66543167,0.66875883,0.67210262,0.67546314,0.67884045,0.68223466,0.68564583,0.68907406,0.69251943,0.69598202,0.69946194,0.70295924,0.70647404,0.71000641,0.71355644,0.71712423,0.72070985,0.7243134,0.72793496,0.73157464,0.73523251,0.73890867,0.74260322,0.74631623,0.75004781,0.75379805,0.75756704,0.76135488,0.76516165,0.76898746,0.7728324,0.77669656,0.78058004,0.78448294,0.78840536,0.79234738,0.79630912,0.80029067,0.80429212,0.80831358,0.81235515,0.81641693,0.82049901,0.8246015,0.82872451,0.83286813,0.83703248,0.84121764,0.84542373,0.84965084,0.8538991,0.85816859,0.86245944,0.86677173,0.87110559,0.87546112,0.87983843,0.88423762,0.88865881,0.8931021,0.89756761,0.90205545,0.90656573,0.91109856,0.91565405,0.92023232,0.92483348,0.92945765,0.93410494,0.93877546,0.94346934,0.94818668,0.95292762,0.95769226,0.96248072,0.96729312,0.97212959,0.97699023,0.98187519,0.98678456,0.99171848,0.99667708,1.0016605,1.0066688,1.0117021,1.0167606,1.0218444,1.0269536,1.0320884,1.0372489,1.0424351,1.0476473,1.0528855,1.0581499,1.0634407,1.0687579,1.0741017,1.0794722,1.0848695,1.0902939,1.0957454,1.1012241,1.1067302,1.1122639,1.1178252,1.1234143,1.1290314,1.1346765,1.1403499,1.1460517,1.1517819,1.1575408,1.1633285,1.1691452,1.1749909,1.1808659,1.1867702,1.192704,1.1986676,1.2046609,1.2106842,1.2167376,1.2228213,1.2289354,1.2350801,1.2412555,1.2474618,1.2536991,1.2599676,1.2662674,1.2725988,1.2789618,1.2853566,1.2917833,1.2982423,1.3047335,1.3112571,1.3178134,1.3244025,1.3310245,1.3376796,1.344368,1.3510899,1.3578453,1.3646345,1.3714577,1.378315,1.3852066,1.3921326,1.3990933,1.4060887,1.4131192,1.4201848,1.4272857,1.4344221,1.4415942,1.4488022,1.4560462,1.4633265,1.4706431,1.4779963,1.4853863,1.4928132,1.5002773,1.5077787,1.5153176,1.5228942,1.5305086,1.5381612,1.545852,1.5535812,1.5613491,1.5691559,1.5770017,1.5848867,1.5928111,1.6007752,1.608779,1.6168229,1.624907,1.6330316,1.6411967,1.6494027,1.6576497,1.6614142,1.6619858,1.665938,1.6742677,1.682639,1.6910522,1.6995075,1.708005,1.7158508,1.716545,1.7169493,1.7251278,1.7337534,1.7424222,1.7511343,1.7598899,1.7686894,1.7775328,1.7864205,1.7953526,1.8043294,1.813351,1.8224178,1.8315299,1.8406875,1.8498909,1.8591404,1.8684361,1.8777783,1.8871672,1.896603,1.906086,1.9156165,1.9251945,1.9348205,1.9444946,1.9542171,1.9639882,1.9738081,1.9836772,1.9935955,2.0035635,2.0135813,2.0236492,2.0337675,2.0439363,2.054156,2.0644268,2.0747489,2.0851227,2.0955483,2.1052605,2.106026,2.1099394,2.1165562,2.1271389,2.1377746,2.1484635,2.1592058,2.1700018,2.1808519,2.1917561,2.2027149,2.2137285,2.2247971,2.2359211,2.2471007,2.2583362,2.2696279,2.280976,2.2923809,2.3038428,2.315362,2.3269388,2.3385735,2.3502664,2.3620177,2.3625852,2.3682149,2.3738278,2.385697,2.3976254,2.4096136,2.4216616,2.4337699,2.4459388,2.4581685,2.4704593,2.4828116,2.4952257,2.5077018,2.5202403,2.5328415,2.5455057,2.5582333,2.5710244,2.5838796,2.5957502,2.596799,2.6060497,2.6097829,2.6228319,2.635946,2.6491257,2.6623714,2.6756832,2.6890617,2.702507,2.7160195,2.7295996,2.7432476,2.7569638,2.7707486,2.7846024,2.7985254,2.812518,2.8265806,2.8407135,2.8549171,2.8691917,2.8835376,2.8979553,2.9124451,2.9270073,2.9416424,2.9563506,2.9711323,2.985988,3.0009179,3.0159225,3.0310021,3.0461571,3.0613879,3.0766949,3.0920783,3.1075387,3.1230764,3.1386918,3.1543853,3.1701572,3.186008,3.201938,3.2179477,3.2340374,3.2502076,3.2664587,3.282791,3.2992049,3.3157009,3.3322794,3.3489408,3.3656856,3.382514,3.3994265,3.4164237,3.4335058,3.4506733,3.4679267,3.4852663,3.5026927,3.5202061,3.5378072,3.5554962,3.5732737,3.59114,3.6090957,3.6271412,3.6452769,3.6635033,3.6818208,3.7002299,3.7187311,3.7373247,3.7560114,3.7747914,3.7936654,3.8126337,3.8316969,3.8508554,3.8701096,3.8894602,3.9089075,3.928452,3.9480943,3.9678347,3.9876739,4.055024,4.334821,4.633924,4.953664,5.295467,5.660855,6.051453,6.469004,6.915365,7.392525,7.902609,8.44789,9.030794,9.369486,9.512897,9.55114,9.608504,9.653919,9.751914,10.32004,10.52461,10.6857,10.72866,10.7931,10.95419,11.03212,11.04529,11.21435,11.25943,11.32705,11.49611,11.79334,12.60708,13.47697,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,64.04378,65.02405,65.28545,65.67755,66.65782,66.84318,71.45536,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Hf_attenuation = np.array([47938,46879,42624,42740,42314,41310,39738,37645,35117,33036,32365,32263,32186,39787,38711,35577,31132,27222,23752,20691,18035,15782,13926,13203,12873,12790,68213,66178,65464,55158,45911,39752,38026,37821,37584,58709,56445,50347,42569,36054,30862,26982,24283,22579,21849,21696,21669,21661,21788,21690,21550,21649,21972,22403,22853,23259,23435,23459,23481,23503,23525,23545,23565,23585,23603,23621,23639,23655,23671,23686,23700,23713,23726,23738,23749,23760,23770,23778,23786,23794,23800,23806,23811,23815,23818,23821,23823,23823,23824,23823,23821,23819,23816,23812,23807,23802,23795,23788,23780,23772,23762,23752,23741,23729,23717,23703,23689,23674,23659,23642,23625,23607,23588,23569,23549,23528,23506,23484,23461,23437,23413,23388,23362,23335,23308,23280,23252,23223,23193,23162,23131,23099,23067,23034,23000,22966,22931,22896,22860,22823,22786,22748,22710,22671,22632,22592,22551,22511,22469,22427,22385,22342,22298,22254,22210,22165,22120,22074,22028,21981,21934,21887,21839,21791,21742,21693,21644,21594,21544,21494,21443,21392,21341,21289,21237,21184,21132,21079,21025,20972,20918,20864,20810,20755,20700,20645,20590,20534,20478,20422,20366,20310,20253,20197,20140,20083,20025,19968,19911,19853,19795,19737,19679,19621,19562,19504,19445,19387,19328,19322,21975,21893,21757,21623,21491,21361,21233,21107,20983,20861,20818,22605,22549,22375,22205,22037,21873,21712,21554,21399,21247,21098,20951,20809,20670,20535,20403,20274,20148,20026,19906,19788,19674,19563,19455,19350,19248,19148,19051,18956,18863,18772,18683,18595,18510,18426,18344,18263,18184,18105,18029,17953,17879,17806,17734,17662,17592,17523,17455,17388,17322,17256,17184,17108,17033,16959,16885,16812,16740,16668,16597,16527,16457,16388,16319,16245,16169,16093,16018,15943,15869,15796,15722,15650,15573,15457,15341,15226,15112,14999,14886,14775,14664,14554,14444,14336,14228,14121,14014,13909,13804,13700,13597,13494,13393,13292,13192,13092,12993,12895,12798,12702,12606,12511,12417,12323,12230,12138,12081,12870,12857,12777,12685,12594,12504,12414,12325,12237,12150,12063,11977,11891,11806,11722,11638,11556,11473,11392,11311,11231,11151,11072,10994,10916,10839,10762,10686,10611,10580,10721,10719,10645,10571,10498,10426,10354,10282,10211,10141,10071,10001,9932.6,9864.2,9796.3,9729,9662.1,9595.7,9529.8,9464.4,9399.4,9334.9,9270.9,9207.3,9144.2,9081.4,9019.2,8957.3,8895.9,8834.9,8774.3,8714.1,8654.3,8594.9,8535.9,8477.3,8419,8361.2,8303.7,8246.6,8189.8,8133.4,8077.3,8071.9,8266.6,8251.7,8195.6,8139.8,8084.3,8029.2,7974.5,7920,7865.9,7812.1,7758.6,7705.4,7652.5,7599.9,7547.6,7495.6,7443.8,7392.4,7341.2,7290.3,7239.7,7189.4,7139.3,7089.5,7040,6990.7,6941.7,6892.9,6844.4,6796.1,6748.1,6700.3,6652.8,6605.5,6558.4,6511.6,6465.1,6418.7,6372.6,6326.8,6281.1,6235.7,6190.5,6145.6,6100.9,6056.4,6012.1,5968.1,5924.3,5880.7,5837.3,5794.1,5751.2,5708.5,5666,5623.7,5581.7,5539.8,5498.2,5456.8,5415.6,5374.6,5333.9,5293.3,5253,5212.7,5172.7,5132.8,5093.2,5053.8,5014.5,4975.5,4936.8,4898.2,4859.8,4821.6,4783.7,4746,4708.4,4671.1,4634,4597.1,4560.4,4523.9,4487.7,4451.6,4415.8,4380.1,4344.7,4309.5,4274.5,4239.7,4205.1,4170.7,4136.5,4102.5,4068.8,4035.2,4001.8,3968.6,3935.6,3902.9,3870.3,3837.9,3805.8,3773.8,3742.1,3710.4,3679,3647.8,3616.8,3586.1,3555.5,3525.2,3495.1,3465.2,3435.5,3406.1,3376.9,3347.9,3319.1,3290.6,3262.3,3234.2,3206.4,3175.7,3139.3,3103.4,3067.9,3032.9,2998.3,2964.1,2930.4,2897.1,2864.2,2831.7,2799.7,2768,2736.8,2705.9,2675.4,2645.3,2615.6,2586.2,2557.2,2528.6,2500.3,2472.4,2444.8,2417.5,2390.6,2364,2337.8,2311.9,2286.2,2260.9,2236,2211.3,2186.9,2162.8,2139,2115.5,2092.3,2069.4,2046.7,2024.4,2002.2,1980.4,1958.8,1937.5,1916.4,1895.6,1875,1854.7,1834.6,1814.8,1795.2,1775.9,1756.7,1737.8,1719.1,1700.7,1682.5,1664.4,1646.6,1629.1,1611.7,1594.5,1577.5,1560.7,1544.2,1527.8,1511.6,1495.6,1479.8,1464.2,1448.7,1433.5,1418.4,1403.5,1388.8,1374.2,1359.8,1345.6,1331.5,1317.6,1303.9,1290.3,1276.9,1263.6,1250.3,1237.1,1224,1211.2,1198.4,1185.8,1173.4,1161.1,1148.9,1136.9,1125,1113.3,1101.7,1090.2,1078.9,1067.7,1056.6,1051.6,3704.1,3682.5,3637.5,3593,3549.1,3505.8,3463,3424.2,5104.7,5101.7,5041.7,4979.5,4918.1,4857.4,4797.4,4738.1,4679.6,4621.8,4564.7,4508.4,4452.8,4397.9,4343.6,4290.1,4237.2,4185,4133.4,4082.5,4032.2,3982.6,3933.6,3885.2,3837.4,3790.2,3743.6,3697.6,3652.2,3607.3,3563,3519.2,3476,3433.4,3391.2,3349.7,3308.6,3268,3228,3188.4,3149.4,3110.8,3075.5,3072.7,3588.4,3559.6,3514.4,3469.7,3425.5,3382,3338.5,3295.5,3253,3211.1,3170.2,3131.6,3093.4,3055.6,3018.4,2981.7,2945.6,2910,2874.9,2840.4,2806.3,2772.8,2739.7,2707,2705.5,2877.6,2860.8,2825.9,2791.4,2757.4,2723.9,2690.8,2658.2,2626,2594.2,2562.9,2533.1,2503.8,2474.8,2446.3,2418.2,2390.5,2363.2,2336.2,2311.8,2309.6,2395.5,2387.8,2361,2334.5,2308.4,2282.6,2257.1,2231.9,2207,2182.4,2158.1,2134.1,2110.2,2086.6,2063.2,2040.1,2017.3,1994.8,1972.5,1950.5,1928.7,1907.2,1885.9,1864.8,1844,1823.4,1803.1,1782.9,1763,1743.2,1722.2,1701.4,1680.9,1660.7,1640.6,1620.7,1601.1,1581.7,1562.5,1543.6,1524.9,1506.5,1488.2,1470.3,1452.5,1434.9,1417.6,1400.5,1383.6,1366.9,1350.2,1333.4,1316.9,1300.5,1284.3,1268.4,1252.6,1236.7,1221.1,1205.6,1190.4,1175.4,1160.6,1145.9,1131.5,1117.3,1103.2,1089.4,1075.7,1062.2,1048.9,1035.8,1022.9,1010.1,997.51,985.08,972.82,960.72,948.79,937.01,925.39,913.93,902.62,891.46,880.45,869.58,858.86,823.78,696.66,588.53,497.12,420.12,355.32,301.01,255.41,217.07,184.47,156.39,132.8,112.58,101.98,97.899,96.849,259.08,255.17,247.01,207.71,196.44,188.27,186.19,256.3,245.42,240.41,239.57,229.25,226.61,257.87,248.24,232.7,196.77,166.16,140.11,117.83,99.026,83.239,70.013,58.858,49.514,41.634,35.006,29.466,24.642,20.581,17.212,14.414,12.085,10.147,8.5288,7.1665,6.0257,5.0574,4.2463,3.5703,3.0076,2.8295,2.7224,2.6949,12.086,11.65,11.571,9.7814,8.2445,6.9465,5.8424,4.9087,4.1278,3.4744,2.9276,2.4698,2.0864,1.7592,1.4857,1.2571,1.0661,0.90637,0.77266,0.66065,0.56676,0.48798,0.42179,0.36613,0.31976,0.28133,0.24863,0.22074,0.19691,0.17647,0.15891])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Hf_attenuation_interpolator = interp1d(energy_Hf_attenuation,mass_coefficient_Hf_attenuation*Hfdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV



energy_Al_attenuation = np.array([0.008417543,0.008543178,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.03109002,0.03323523,0.03552846,0.03797993,0.04060054,0.04340198,0.04639671,0.04959809,0.05302035,0.05667876,0.06058959,0.06477028,0.06923942,0.071638,0.0727345,0.0730269,0.0734655,0.07401695,0.074562,0.07912411,0.08458368,0.09041995,0.09665893,0.1033284,0.1104581,0.115346,0.1171115,0.1175823,0.1180797,0.1182885,0.120054,0.1262272,0.1349368,0.1442475,0.1542005,0.1648404,0.1762144,0.1883732,0.2013709,0.2152655,0.2301188,0.245997,0.2629708,0.2811158,0.3005128,0.3212482,0.3434143,0.3671099,0.3924405,0.4195189,0.4484657,0.4794098,0.5124891,0.5478508,0.5856525,0.6260625,0.6692609,0.7154399,0.7648052,0.8175768,0.8739896,0.9342948,0.9987612,1.067676,1.141345,1.220098,1.304285,1.394281,1.490486,1.528408,1.551802,1.55804,1.567398,1.590792,1.593329,1.703269,1.820795,1.94643,2.080733,2.224304,2.377781,2.541848,2.717235,2.904724,3.10515,3.319406,3.548445,3.793288,4.055024,4.334821,4.633924,4.953664,5.295467,5.660855,6.051453,6.469004,6.915365,7.392525,7.902609,8.44789,9.030794,9.653919,10.32004,11.03212,11.79334,12.60708,13.47697,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,71.45536,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Al_attenuation = np.array([19119,16513,2677.3,2054,1744.8,1609.9,1586.4,1645.1,1772.8,1963.6,2215,2524.7,2888.8,3300.5,3748.8,4218.6,4691,5144.2,5555.8,5904.6,6162.5,6305.1,6334.2,6259.1,6095.1,5853.5,5538.7,5180.1,4795.6,4400.6,4008.2,3812.3,3726.2,3703.7,83378,84589,85797,96319,108400,116930,121370,121850,118890,115460,114020,113620,124430,124250,122650,116650,107640,97963,88119,78493,69358,60884,53155,46199,40005,34535,29735,25546,21904,18742,16007,13648,11620,9879.7,8389.8,7116.7,6029.8,5103.3,4314.8,3645.2,3077.1,2596.2,2189.4,1845.9,1555.9,1311.5,1105.8,920.86,767.42,640.19,534.58,446.24,371.36,346.67,332.58,328.96,4038,3957,3948.4,3485.8,2943,2447.6,2035.5,1706,1432.1,1203.6,1011.4,849.33,710.68,589.18,488.26,404.68,335.43,278.07,230.55,191.18,158.55,131.51,109.1,90.526,74.937,61.744,50.887,41.953,34.536,28.257,23.131,18.945,15.527,12.734,10.454,8.5895,7.0658,5.8199,4.801,3.9673,3.2851,2.7264,2.2689,1.8939,1.5865,1.3296,1.1184,0.94583,0.80374,0.68672,0.59119,0.51309,0.44915,0.39671,0.35359,0.31803,0.28863,0.26423,0.24389,0.22686,0.21251,0.20034,0.18995,0.181,0.17321,0.16636,0.16028,0.15482,0.14987,0.14532,0.1411,0.13716,0.13342,0.12986,0.12644,0.12314,0.11993,0.1168,0.11374,0.11073,0.10778,0.10486,0.10199,0.099153,0.096355,0.093592,0.090866])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Al_attenuation_interpolator = interp1d(energy_Al_attenuation,mass_coefficient_Al_attenuation*Aldensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Zr_attenuation = np.array([0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.028126,0.0285565,0.0286713,0.0288435,0.02908327,0.029274,0.03109002,0.03323523,0.03552846,0.03797993,0.04060054,0.04340198,0.04639671,0.04959809,0.050274,0.0510435,0.0512487,0.0515565,0.052326,0.05302035,0.05667876,0.06058959,0.06477028,0.06923942,0.07401695,0.07912411,0.08458368,0.09041995,0.09665893,0.1033284,0.1104581,0.1180797,0.1262272,0.1349368,0.1442475,0.1542005,0.1648404,0.1762144,0.1764,0.178752,0.1791,0.17982,0.1809,0.181488,0.1822176,0.183312,0.1836,0.186048,0.1883732,0.2013709,0.2152655,0.2301188,0.245997,0.2629708,0.2811158,0.3005128,0.3212482,0.32389,0.3288475,0.3301695,0.3321525,0.337316,0.342479,0.3434143,0.3438558,0.345921,0.351084,0.3671099,0.3924405,0.4195189,0.421694,0.4281485,0.4298697,0.4324515,0.438906,0.4484657,0.4794098,0.5124891,0.5478508,0.5856525,0.6260625,0.6692609,0.7154399,0.7648052,0.8175768,0.8739896,0.9342948,0.9987612,1.067676,1.141345,1.220098,1.304285,1.394281,1.490486,1.593329,1.703269,1.820795,1.94643,2.080733,2.177854,2.211189,2.220078,2.224304,2.233412,2.260566,2.266746,2.295166,2.304393,2.318233,2.352834,2.377781,2.480968,2.518942,2.529068,2.541848,2.544258,2.582232,2.717235,2.904724,3.10515,3.319406,3.548445,3.793288,4.055024,4.334821,4.633924,4.953664,5.295467,5.660855,6.051453,6.469004,6.915365,7.392525,7.902609,8.44789,9.030794,9.653919,10.32004,11.03212,11.79334,12.60708,13.47697,14.40688,15.40095,16.46362,17.59961,17.63765,17.90761,17.9796,18.08759,18.35755,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,71.45536,76.38578,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Zr_attenuation = np.array([162020,156300,145190,129980,112320,93901,76157,60092,46295,34973,26017,19137,13972,10161,7384.1,6302.3,5864,5753.6,126350,121920,118530,91486,68632,51432,38671,29287,22413,17374,13666,13041,12384,12217,12989,12433,11970,10007,8545.9,7438,6585,5916.6,5388.9,4960.9,4603.9,4299.8,4035.1,3800.7,3589.3,3396,3216.6,3048,2888,2734.9,2587.6,2585.4,2556.7,2552.5,2543.9,5907.4,5975.4,6061.8,8462.6,8522.7,9057.1,9605.9,13507,18340,23004,26798,29276,30308,30024,28704,28487,28058,27939,30637,30144,29634,29540,29496,30488,29958,28281,25640,22976,22772,22177,22021,22765,22182,21349,18891,16633,14588,12751,11118,9674.2,8402.8,7282.6,6299.1,5440.3,4693.3,4045.5,3433.4,2915.8,2478.8,2109.9,1789.4,1514.4,1283.8,1090,925.75,787.13,670.22,600.95,579.59,574.08,2356.5,2328.3,2247,2229.1,2148.9,2123.8,2924.8,2803.7,2723,2430.4,2336.5,2312.5,2627.3,2621.1,2526.9,2231.6,1899.4,1604.5,1349.4,1134.2,953.9,803.03,676.65,568.35,475.36,397.13,331.74,277.4,232.09,194.24,162.74,136.51,114.61,96.089,79.996,66.411,55.15,45.671,37.876,31.456,26.162,21.793,18.181,15.192,15.105,14.503,14.349,95.056,91.064,84.814,70.963,59.752,50.383,42.32,35.475,29.623,24.716,20.629,17.225,14.362,11.917,9.8949,8.2215,6.8362,5.6887,4.7379,3.9484,3.2949,2.7536,2.3052,1.9336,1.6203,1.3578,1.1413,0.96272,0.81529,0.69347,0.59272,0.50929,0.44014,0.3827,0.33493,0.2951,0.26182,0.23392,0.21047,0.19067,0.1739,0.1596,0.14735,0.1368,0.12764,0.11964,0.11259,0.10634,0.10074,0.095691])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Zr_attenuation_interpolator = interp1d(energy_Zr_attenuation,mass_coefficient_Zr_attenuation*Zrdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Au_attenuation = np.array([0.006824268,0.006926123,0.008142217,0.008266843,0.008300076,0.008349926,0.008474552,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.03109002,0.03323523,0.03552846,0.03797993,0.04060054,0.04340198,0.04639671,0.04959809,0.052626,0.05302035,0.0534315,0.0536463,0.0539685,0.054774,0.05667876,0.06058959,0.06477028,0.06923942,0.070266,0.0713415,0.0716283,0.0720585,0.073134,0.07401695,0.07912411,0.081144,0.082386,0.0827172,0.083214,0.084456,0.08458368,0.084672,0.085968,0.0863136,0.086832,0.088128,0.09041995,0.09665893,0.1033284,0.105644,0.107261,0.1076922,0.108339,0.109956,0.1104581,0.1180797,0.1262272,0.1349368,0.1442475,0.1542005,0.1648404,0.1762144,0.1883732,0.2013709,0.2152655,0.2301188,0.245997,0.2629708,0.2811158,0.3005128,0.3212482,0.327222,0.3322305,0.3335661,0.3355695,0.340578,0.3434143,0.34496,0.35024,0.351648,0.35376,0.35904,0.3671099,0.3924405,0.4195189,0.4484657,0.4794098,0.5,0.5025,0.5050125,0.50753756,0.51007525,0.51262563,0.51518875,0.5177647,0.52035352,0.52295529,0.52557007,0.52819792,0.53083891,0.5334931,0.53616057,0.53884137,0.54153558,0.54424325,0.54469646,0.54610359,0.54696447,0.54969929,0.55244779,0.55521003,0.55798608,0.56077601,0.56357989,0.56639779,0.56922978,0.57207593,0.5749363,0.57781099,0.58070004,0.58360354,0.58652156,0.58945417,0.59240144,0.59536345,0.59834026,0.60133196,0.60433862,0.60736032,0.61039712,0.6134491,0.61651635,0.61959893,0.62269693,0.62581041,0.62893946,0.63208416,0.63524458,0.6384208,0.64161291,0.64278595,0.64461406,0.64482097,0.64804508,0.6512853,0.65454173,0.65781444,0.66110351,0.66440903,0.66773107,0.67106973,0.67442508,0.6777972,0.68118619,0.68459212,0.68801508,0.69145515,0.69491243,0.69838699,0.70187893,0.70538832,0.70891526,0.71245984,0.71602214,0.71960225,0.72320026,0.72681626,0.73045034,0.7341026,0.73777311,0.74146197,0.74516928,0.74889513,0.75263961,0.7564028,0.75759354,0.76000652,0.76018482,0.76398574,0.76780567,0.7716447,0.77550292,0.77938044,0.78327734,0.78719373,0.79112969,0.79508534,0.79906077,0.80305607,0.80707135,0.81110671,0.81516224,0.81923806,0.82333425,0.82745092,0.83158817,0.83574611,0.83992484,0.84412447,0.84834509,0.85258682,0.85684975,0.861134,0.86543967,0.86976687,0.8741157,0.87848628,0.88287871,0.8872931,0.89172957,0.89618822,0.90066916,0.9051725,0.90969837,0.91424686,0.91881809,0.92341218,0.92802924,0.93266939,0.93733274,0.9420194,0.9467295,0.95146315,0.95622046,0.96100156,0.96580657,0.9706356,0.97548878,0.98036623,0.98526806,0.9901944,0.99514537,1.0001211,1.0051217,1.0101473,1.015198,1.020274,1.0253754,1.0305023,1.0356548,1.0408331,1.0460372,1.0512674,1.0565238,1.0618064,1.0671154,1.072451,1.0778132,1.0832023,1.0886183,1.0940614,1.0995317,1.1050294,1.1105545,1.1161073,1.1216878,1.1272963,1.1329328,1.1385974,1.1442904,1.1500119,1.1557619,1.1615407,1.1673484,1.1731852,1.1790511,1.1849464,1.1908711,1.1968254,1.2028096,1.2088236,1.2148677,1.2209421,1.2270468,1.233182,1.2393479,1.2455447,1.2517724,1.2580312,1.2643214,1.270643,1.2769962,1.2833812,1.2897981,1.2962471,1.3027283,1.309242,1.3157882,1.3223671,1.328979,1.3356239,1.342302,1.3490135,1.3557586,1.3625374,1.36935,1.3761968,1.3830778,1.3899932,1.3969431,1.4039278,1.4109475,1.4180022,1.4250922,1.4322177,1.4393788,1.4465757,1.4538086,1.4610776,1.468383,1.4757249,1.4831035,1.490519,1.4979716,1.5054615,1.5129888,1.5205537,1.5281565,1.5357973,1.5434763,1.5511937,1.5589496,1.5667444,1.5745781,1.582451,1.5903633,1.5983151,1.6063066,1.6143382,1.6224099,1.6305219,1.6386745,1.6468679,1.6551022,1.6633777,1.6716946,1.6800531,1.6884534,1.6968956,1.7053801,1.713907,1.7224766,1.7310889,1.7397444,1.7484431,1.7571853,1.7659712,1.7748011,1.7836751,1.7925935,1.8015565,1.8105642,1.8196171,1.8287151,1.8378587,1.847048,1.8562833,1.8655647,1.8748925,1.884267,1.8936883,1.9031567,1.9126725,1.9222359,1.9318471,1.9415063,1.9512138,1.9609699,1.9707747,1.9806286,1.9905318,2.0004844,2.0104868,2.0205393,2.030642,2.0407952,2.0509992,2.0612542,2.0715604,2.0819182,2.0923278,2.1027895,2.1133034,2.1238699,2.1344893,2.1451617,2.1558875,2.166667,2.1775003,2.1883878,2.1993297,2.2051132,2.2062866,2.2103264,2.221378,2.2324849,2.2436473,2.2548656,2.2661399,2.2774706,2.2888579,2.2904585,2.2917415,2.3003022,2.3118037,2.3233628,2.3349796,2.3466545,2.3583878,2.3701797,2.3820306,2.3939407,2.4059104,2.41794,2.4300297,2.4421798,2.4543907,2.4666627,2.478996,2.491391,2.5038479,2.5163672,2.528949,2.5415938,2.5543017,2.5670732,2.5799086,2.5928082,2.6057722,2.6188011,2.6318951,2.6450545,2.6582798,2.6715712,2.6849291,2.6983537,2.7118455,2.7254047,2.7390317,2.7404353,2.7455647,2.7527269,2.7664905,2.780323,2.7942246,2.8081957,2.8222367,2.8363479,2.8505296,2.8647823,2.8791062,2.8935017,2.9079692,2.9225091,2.9371216,2.9518072,2.9665662,2.9813991,2.9963061,3.0112876,3.026344,3.0414758,3.0566831,3.0719666,3.0873264,3.102763,3.1182768,3.1338682,3.1431727,3.1495376,3.1524272,3.1652853,3.1811117,3.1970172,3.2130023,3.2290673,3.2452127,3.2614387,3.2777459,3.2941347,3.3106053,3.3271584,3.3437941,3.3605131,3.3773157,3.3942023,3.4111733,3.417742,3.4282291,3.4320581,3.4453703,3.4625971,3.4799101,3.4973097,3.5147962,3.5323702,3.5500321,3.5677822,3.5856211,3.6035492,3.621567,3.6396748,3.6578732,3.6761626,3.6945434,3.7130161,3.7315812,3.7502391,3.7689903,3.7878352,3.8067744,3.8258083,3.8449373,3.864162,3.8834828,3.9029002,3.9224147,3.9420268,3.9617369,3.9815456,4.0014533,4.0214606,4.0415679,4.0617757,4.0820846,4.102495,4.1230075,4.1436226,4.1643407,4.1851624,4.2060882,4.2271186,4.2482542,4.2694955,4.290843,4.3122972,4.3338587,4.355528,4.3773056,4.3991921,4.4211881,4.443294,4.4655105,4.4878381,4.5102772,4.5328286,4.5554928,4.5782702,4.6011616,4.6241674,4.6472882,4.6705247,4.6938773,4.7173467,4.7409334,4.7646381,4.7884613,4.8124036,4.8364656,4.8606479,4.8849512,4.9093759,4.9339228,4.9585924,4.9833854,5.0083023,5.0333438,5.0585105,5.0838031,5.1092221,5.1347682,5.1604421,5.1862443,5.2121755,5.2382364,5.2644276,5.2907497,5.3172034,5.3437895,5.3705084,5.3973609,5.4243477,5.4514695,5.4787268,5.5061205,5.5336511,5.5613193,5.5891259,5.6170716,5.6451569,5.6733827,5.7017496,5.7302584,5.7589096,5.7877042,5.8166427,5.8457259,5.8749546,5.9043293,5.933851,5.9635202,5.9933378,6.0233045,6.053421,6.0836882,6.1141066,6.1446771,6.1754005,6.2062775,6.2373089,6.2684954,6.2998379,6.3313371,6.3629938,6.3948088,6.4267828,6.4589167,6.4912113,6.5236674,6.5562857,6.5890671,6.6220125,6.6551225,6.6883981,6.7218401,6.7554493,6.7892266,6.8231727,6.8572886,6.891575,6.9260329,6.9606631,6.9954664,7.0304437,7.0655959,7.1009239,7.1364285,7.1721107,7.2079712,7.2440111,7.2802311,7.3166323,7.3532155,7.3899815,7.4269314,7.4640661,7.5013864,7.5388934,7.5765878,7.6144708,7.6525431,7.6908058,7.7292599,7.7679062,7.8067457,7.8457794,7.8850083,7.9244334,7.9640555,8.0038758,8.0438952,8.0841147,8.1245352,8.1651579,8.2059837,8.2470136,8.2882487,8.3296899,8.3713384,8.4131951,8.455261,8.4975373,8.540025,9.030794,9.653919,10.32004,11.03212,11.68033,11.79334,11.85911,11.90678,11.97829,12.15707,12.60708,13.45893,13.47697,13.66493,13.71987,13.80227,14.00827,14.06574,14.28104,14.33845,14.40688,14.42456,14.63986,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,71.45536,76.38578,79.1104,80.32127,80.64417,81.12852,81.6564,82.3394,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Au_attenuation = np.array([42705,41481,32469,32352,32341,57837,56718,59861,67392,77054,88504,101290,114780,128180,140530,150790,158010,161410,160580,155490,146550,134520,120340,105070,89725,75109,61791,50114,40190,31965,25284,20477,19938,19396,19120,34412,32653,28930,22850,18036,14260,13546,12851,12673,17817,17073,16498,13697,12769,12228,12088,13428,12854,12797,12757,12191,12044,12810,12253,11335,9286.9,7774.4,7376.1,7130,7068.6,7167.5,6960.4,6900.6,6215.8,5862.7,5835.1,6094.1,6625.1,7424.6,8457.9,9680.7,11031,12427,13777,14982,15957,16634,16976,16968,16906,16837,16816,17141,17065,17015,16987,16880,16849,17056,16939,16739,15988,15057,14012,12911,12209,12126,12043,11960,11878,11795,11713,11630,11548,11466,11385,11303,11222,11141,11060,10979,10899,10818,10805,11434,11408,11326,11244,11163,11082,11001,10920,10840,10760,10681,10601,10523,10444,10366,10288,10211,10134,10057,9980.6,9904.7,9829.1,9753.9,9679.1,9604.7,9530.6,9457,9383.7,9310.8,9238.3,9166.2,9094.5,9023.2,8952.3,8926.5,9013.1,9008.6,8938.4,8868.7,8799.4,8730.5,8662,8593.9,8526.2,8459,8392.1,8325.7,8259.6,8194,8128.8,8064,7999.6,7935.6,7872,7808.8,7746.1,7683.7,7621.8,7560.3,7499.1,7438.4,7378.1,7318.2,7258.7,7199.5,7140.8,7082.5,7024.6,6967.1,6949,7063.3,7060.6,7003.4,6946.6,6890.1,6834.1,6778.4,6723.2,6668.3,6613.8,6559.7,6506,6452.7,6399.7,6347.1,6294.9,6243.1,6191.7,6140.6,6089.9,6039.6,5989.6,5940,5890.8,5841.9,5793.3,5745.2,5697.3,5649.8,5602.7,5555.9,5509.5,5463.4,5417.6,5372.2,5327.2,5282.4,5238,5194,5150.3,5106.8,5063.6,5020.6,4978,4935.8,4893.8,4852.2,4810.9,4769.9,4729.2,4688.9,4648.8,4609.1,4569.7,4530.6,4491.8,4453.2,4409.1,4365.4,4322.2,4279.3,4236.9,4194.9,4153.3,4112.1,4071.4,4031,3991,3951.5,3912.3,3873.5,3835.1,3797.1,3759.5,3722.1,3685,3648.3,3612,3576.1,3540.5,3505.2,3470.4,3435.8,3401.7,3367.8,3334.3,3301.2,3268.4,3235.9,3203.8,3172,3140.5,3109.3,3078.4,3047.9,3017.7,2987.8,2958.2,2928.9,2899.9,2871.2,2842.8,2814.7,2786.9,2759.4,2732.1,2705.2,2678.4,2651.9,2625.7,2599.8,2574.1,2548.7,2523.6,2498.7,2472.7,2446.9,2421.5,2396.3,2371.4,2346.7,2322.4,2298.3,2274.5,2250.9,2227.3,2203.4,2179.8,2156.5,2133.4,2110.6,2088.1,2065.8,2043.8,2022,2000.5,1979.2,1958.2,1937.4,1916.8,1896.5,1876.4,1856.6,1836.9,1817.5,1797.9,1778.2,1758.7,1739.4,1720.4,1701.6,1683.1,1664.7,1646.6,1628.2,1610,1592,1574.2,1556.7,1539.4,1522.3,1505.4,1488.8,1472.3,1456.1,1440,1424.2,1408.5,1393,1377.8,1362.7,1347.8,1333.1,1318.5,1304.2,1290,1276,1262.2,1248.5,1235,1221.7,1208.5,1195.5,1182.6,1169.9,1157.4,1145,1132.7,1120.6,1108.7,1096.9,1085.2,1073.7,1062.3,1051,1039.9,1028.9,1018.1,1007.3,996.74,986.26,975.91,965.67,955.56,945.57,935.7,925.94,916.29,906.76,897.34,888.04,878.84,869.74,860.64,851.58,846.85,2514.2,2503.1,2472.8,2443,2413.5,2384.3,2355.6,2327.2,2299.2,2295.3,3356.6,3325.9,3285.2,3245,3205.3,3166.1,3127.4,3089.2,3051.5,3014.2,2977.4,2941.1,2905.2,2869.8,2834.8,2800.2,2766.1,2732.4,2699.2,2666.3,2633.9,2601.8,2570.2,2539,2508.1,2477.6,2447.4,2417.6,2388.2,2359.2,2330.5,2302.2,2274.2,2246.6,2219.3,2192.4,2165.8,2163.1,2501.2,2486.1,2457.6,2429.4,2401.6,2374.1,2346.9,2320,2293.5,2267.3,2240.9,2214.7,2188.9,2163.4,2138.1,2113.2,2088.5,2064.1,2039.8,2014.3,1988.8,1963.6,1938.6,1914,1889.7,1865.7,1842,1818.6,1804.9,1913.6,1909.1,1889.4,1865.5,1841.9,1818.6,1795.5,1772.8,1750.4,1728.3,1706.4,1685,1664,1643.4,1623,1602.9,1583,1563.5,1556,1611.8,1607.5,1592.4,1573.2,1554.2,1535.5,1517.1,1498.9,1480.9,1463.2,1445.5,1428,1410.7,1393.7,1376.9,1360.3,1344,1327.8,1311.9,1296.1,1280.6,1265.3,1250.1,1235.2,1220.4,1205.9,1191.5,1177.3,1163.3,1149.4,1135.8,1122.3,1109,1095.8,1082.9,1070,1057.4,1044.8,1032.3,1020,1007.8,995.78,983.9,972.17,960.59,949.15,937.85,926.69,915.67,904.78,894.03,883.41,872.75,862.02,851.4,840.91,830.56,820.34,810.25,800.29,790.27,780.34,770.54,760.87,751.34,741.93,732.64,723.48,714.44,705.52,696.72,688.04,679.47,671.01,662.67,654.44,646.31,638.3,630.39,622.58,614.88,607.28,599.78,592.38,585.08,577.87,570.76,563.74,556.81,549.98,543.24,536.58,530.01,523.53,517.13,510.82,504.48,498.19,491.98,485.85,479.8,473.84,467.95,462.14,456.41,450.75,445.17,439.66,434.23,428.87,423.57,418.35,413.2,408.11,403.1,398.14,393.26,388.43,383.67,378.98,374.34,369.77,365.25,360.77,356.29,351.88,347.53,343.23,338.99,334.8,330.67,326.6,322.57,318.6,314.67,310.78,306.95,303.17,299.44,295.76,292.12,288.52,284.96,281.45,277.99,274.57,271.2,267.87,264.59,261.35,258.15,254.99,251.87,248.8,245.76,242.77,239.81,236.89,234.01,231.17,228.36,225.59,222.86,220.16,217.49,214.86,212.27,209.7,207.18,204.68,202.21,199.78,197.38,195.01,192.67,190.36,188.08,185.83,183.6,181.41,179.24,177.11,174.99,172.91,170.85,149.06,125.39,105.3,88.517,76.448,74.595,73.547,72.792,183.24,175.95,159.3,133.43,132.96,128.07,126.69,173.54,166.76,164.94,158.35,156.65,178.69,178.14,171.62,151,127.32,107.35,90.437,76.16,64.175,54.116,45.639,38.493,32.267,26.952,22.522,18.842,15.783,13.238,11.117,9.3493,7.8732,6.6388,5.5957,4.7234,3.987,3.3597,2.8356,2.3973,2.1966,2.1146,2.0927,8.6614,8.5221,8.3462,7.2069,6.0889,5.1383,4.3367,3.6619,3.0883,2.6066,2.2022,1.863,1.5784,1.3393,1.1375,0.96814,0.82599,0.70659,0.60621,0.52175,0.4506,0.39062,0.33997,0.29715,0.26087,0.2301,0.20391,0.1816])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Au_attenuation_interpolator = interp1d(energy_Au_attenuation,mass_coefficient_Au_attenuation*Audensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV


energy_Ir_attenuation = np.array([0.00688883,0.006991648,0.007901491,0.008022432,0.008054683,0.008103059,0.008224,0.01069,0.01142761,0.01221612,0.01305903,0.0139601,0.01492335,0.01595306,0.01705382,0.01823053,0.01948844,0.02083314,0.02227063,0.0238073,0.02545001,0.02720606,0.02908327,0.03109002,0.03323523,0.03552846,0.03797993,0.04060054,0.04340198,0.04639671,0.04949,0.04959809,0.0502475,0.0504495,0.0507525,0.05151,0.05302035,0.05667876,0.05929,0.0601975,0.0604395,0.06058959,0.0608025,0.06171,0.06174,0.062132,0.062685,0.062937,0.063083,0.0633366,0.063717,0.06426,0.064668,0.06477028,0.06923942,0.07401695,0.07912411,0.08458368,0.09041995,0.093296,0.094724,0.0951048,0.095676,0.09665893,0.097104,0.1033284,0.1104581,0.1180797,0.1262272,0.1349368,0.1442475,0.1542005,0.1648404,0.1762144,0.1883732,0.2013709,0.2152655,0.2301188,0.245997,0.2629708,0.2811158,0.289002,0.2934255,0.2946051,0.2963745,0.3005128,0.300798,0.305172,0.309843,0.3110886,0.312957,0.317628,0.3212482,0.3434143,0.3671099,0.3924405,0.4195189,0.4484657,0.4794098,0.484414,0.4918285,0.4938057,0.4967715,0.5,0.5025,0.5050125,0.50753756,0.51007525,0.51262563,0.51518875,0.5177647,0.52035352,0.52295529,0.52557007,0.52819792,0.53083891,0.5334931,0.53616057,0.53884137,0.54153558,0.54424325,0.54696447,0.54969929,0.55244779,0.55521003,0.55798608,0.56077601,0.56357989,0.56639779,0.56922978,0.57207593,0.5749363,0.57626318,0.57781099,0.57793677,0.58070004,0.58360354,0.58652156,0.58945417,0.59240144,0.59536345,0.59834026,0.60133196,0.60433862,0.60736032,0.61039712,0.6134491,0.61651635,0.61959893,0.62269693,0.62581041,0.62893946,0.63208416,0.63524458,0.6384208,0.64161291,0.64482097,0.64804508,0.6512853,0.65454173,0.65781444,0.66110351,0.66440903,0.66773107,0.67106973,0.67442508,0.6777972,0.68118619,0.68459212,0.68801508,0.68899585,0.69120417,0.69145515,0.69491243,0.69838699,0.70187893,0.70538832,0.70891526,0.71245984,0.71602214,0.71960225,0.72320026,0.72681626,0.73045034,0.7341026,0.73777311,0.74146197,0.74516928,0.74889513,0.75263961,0.7564028,0.76018482,0.76398574,0.76780567,0.7716447,0.77550292,0.77938044,0.78327734,0.78719373,0.79112969,0.79508534,0.79906077,0.80305607,0.80707135,0.81110671,0.81516224,0.81923806,0.82333425,0.82745092,0.83158817,0.83574611,0.83992484,0.84412447,0.84834509,0.85258682,0.85684975,0.861134,0.86543967,0.86976687,0.8741157,0.87848628,0.88287871,0.8872931,0.89172957,0.89618822,0.90066916,0.9051725,0.90969837,0.91424686,0.91881809,0.92341218,0.92802924,0.93266939,0.93733274,0.9420194,0.9467295,0.95146315,0.95622046,0.96100156,0.96580657,0.9706356,0.97548878,0.98036623,0.98526806,0.9901944,0.99514537,1.0001211,1.0051217,1.0101473,1.015198,1.020274,1.0253754,1.0305023,1.0356548,1.0408331,1.0460372,1.0512674,1.0565238,1.0618064,1.0671154,1.072451,1.0778132,1.0832023,1.0886183,1.0940614,1.0995317,1.1050294,1.1105545,1.1161073,1.1216878,1.1272963,1.1329328,1.1385974,1.1442904,1.1500119,1.1557619,1.1615407,1.1673484,1.1731852,1.1790511,1.1849464,1.1908711,1.1968254,1.2028096,1.2088236,1.2148677,1.2209421,1.2270468,1.233182,1.2393479,1.2455447,1.2517724,1.2580312,1.2643214,1.270643,1.2769962,1.2833812,1.2897981,1.2962471,1.3027283,1.309242,1.3157882,1.3223671,1.328979,1.3356239,1.342302,1.3490135,1.3557586,1.3625374,1.36935,1.3761968,1.3830778,1.3899932,1.3969431,1.4039278,1.4109475,1.4180022,1.4250922,1.4322177,1.4393788,1.4465757,1.4538086,1.4610776,1.468383,1.4757249,1.4831035,1.490519,1.4979716,1.5054615,1.5129888,1.5205537,1.5281565,1.5357973,1.5434763,1.5511937,1.5589496,1.5667444,1.5745781,1.582451,1.5903633,1.5983151,1.6063066,1.6143382,1.6224099,1.6305219,1.6386745,1.6468679,1.6551022,1.6633777,1.6716946,1.6800531,1.6884534,1.6968956,1.7053801,1.713907,1.7224766,1.7310889,1.7397444,1.7484431,1.7571853,1.7659712,1.7748011,1.7836751,1.7925935,1.8015565,1.8105642,1.8196171,1.8287151,1.8378587,1.847048,1.8562833,1.8655647,1.8748925,1.884267,1.8936883,1.9031567,1.9126725,1.9222359,1.9318471,1.9415063,1.9512138,1.9609699,1.9707747,1.9806286,1.9905318,2.0004844,2.0104868,2.0205393,2.030642,2.0399144,2.0407952,2.0408856,2.0509992,2.0612542,2.0715604,2.0819182,2.0923278,2.1027895,2.1133034,2.1153256,2.1168746,2.1238699,2.1344893,2.1451617,2.1558875,2.166667,2.1775003,2.1883878,2.1993297,2.2103264,2.221378,2.2324849,2.2436473,2.2548656,2.2661399,2.2774706,2.2888579,2.3003022,2.3118037,2.3233628,2.3349796,2.3466545,2.3583878,2.3701797,2.3820306,2.3939407,2.4059104,2.41794,2.4300297,2.4421798,2.4543907,2.4666627,2.478996,2.491391,2.5038479,2.5163672,2.528949,2.5415938,2.5483176,2.5530823,2.5543017,2.5670732,2.5799086,2.5928082,2.6057722,2.6188011,2.6318951,2.6450545,2.6582798,2.6715712,2.6849291,2.6983537,2.7118455,2.7254047,2.7390317,2.7527269,2.7664905,2.780323,2.7942246,2.8081957,2.8222367,2.8363479,2.8505296,2.8647823,2.8791062,2.8935017,2.9045696,2.9079692,2.9128303,2.9225091,2.9371216,2.9518072,2.9665662,2.9813991,2.9963061,3.0112876,3.026344,3.0414758,3.0566831,3.0719666,3.0873264,3.102763,3.1182768,3.1338682,3.1495376,3.1652853,3.1671623,3.1802379,3.1811117,3.1970172,3.2130023,3.2290673,3.2452127,3.2614387,3.2777459,3.2941347,3.3106053,3.3271584,3.3437941,3.3605131,3.3773157,3.3942023,3.4111733,3.4282291,3.4453703,3.4625971,3.4799101,3.4973097,3.5147962,3.5323702,3.5500321,3.5677822,3.5856211,3.6035492,3.621567,3.6396748,3.6578732,3.6761626,3.6945434,3.7130161,3.7315812,3.7502391,3.7689903,3.7878352,3.8067744,3.8258083,3.8449373,3.864162,3.8834828,3.9029002,3.9224147,3.9420268,3.9617369,3.9815456,4.0014533,4.0214606,4.0415679,4.0617757,4.0820846,4.102495,4.1230075,4.1436226,4.1643407,4.1851624,4.2060882,4.2271186,4.2482542,4.2694955,4.290843,4.3122972,4.3338587,4.355528,4.3773056,4.3991921,4.4211881,4.443294,4.4655105,4.4878381,4.5102772,4.5328286,4.5554928,4.5782702,4.6011616,4.6241674,4.6472882,4.6705247,4.6938773,4.7173467,4.7409334,4.7646381,4.7884613,4.8124036,4.8364656,4.8606479,4.8849512,4.9093759,4.9339228,4.9585924,4.9833854,5.0083023,5.0333438,5.0585105,5.0838031,5.1092221,5.1347682,5.1604421,5.1862443,5.2121755,5.2382364,5.2644276,5.2907497,5.3172034,5.3437895,5.3705084,5.3973609,5.4243477,5.4514695,5.4787268,5.5061205,5.5336511,5.5613193,5.5891259,5.6170716,5.6451569,5.6733827,5.7017496,5.7302584,5.7589096,5.7877042,5.8166427,5.8457259,5.8749546,5.9043293,5.933851,5.9635202,5.9933378,6.0233045,6.053421,6.0836882,6.1141066,6.1446771,6.1754005,6.2062775,6.2373089,6.2684954,6.2998379,6.3313371,6.3629938,6.3948088,6.4267828,6.4589167,6.4912113,6.5236674,6.5562857,6.5890671,6.6220125,6.6551225,6.6883981,6.7218401,6.7554493,6.7892266,6.8231727,6.8572886,6.891575,6.9260329,6.9606631,6.9954664,7.0304437,7.0655959,7.1009239,7.1364285,7.1721107,7.2079712,7.2440111,7.2802311,7.3166323,7.3532155,7.3899815,7.4269314,7.4640661,7.5013864,7.5388934,7.5765878,7.6144708,7.6525431,7.6908058,7.7292599,7.7679062,7.8067457,7.8457794,7.8850083,7.9244334,7.9640555,8.0038758,8.0438952,8.0841147,8.1245352,8.1651579,8.2059837,8.2470136,8.2882487,8.3296899,8.3713384,8.4131951,8.455261,8.4975373,8.540025,9.030794,9.653919,10.32004,10.9909,11.03212,11.15912,11.20399,11.27128,11.4395,11.79334,12.56762,12.60708,12.75998,12.81128,12.88822,13.08058,13.15013,13.35141,13.40508,13.47697,13.48559,13.68687,14.40688,15.40095,16.46362,17.59961,18.81398,20.11215,21.49988,22.98338,24.56923,26.2645,28.07676,30.01405,32.08502,34.29889,36.66551,39.19543,41.89992,44.79101,47.88159,51.18542,54.71721,58.4927,62.5287,66.84318,71.45536,74.58878,75.73045,76.03489,76.38578,76.49156,77.63322,81.6564,87.29069,93.31374,99.75239,106.6353,113.9931,121.8587,130.2669,139.2553,148.864,159.1356,170.1159,181.8539,194.4018,207.8156,222.1548,237.4835,253.8699,271.3869,290.1126,310.1304,331.5294,354.4049,378.8588,405.0001,432.9451])*1e3	# eV
mass_coefficient_Ir_attenuation = np.array([25921,25050,19380,19072,19006,48342,46942,46273,52931,61738,72476,84772,98015,111330,123640,133750,140580,143310,141570,135500,125720,113210,99084,84454,70284,57254,45806,36121,28168,21794,16925,16780,15941,15690,31770,30081,27041,21149,17894,16910,16660,18390,18155,17195,17164,16771,16235,15999,22217,21946,22760,22184,21764,21661,17803,14744,12272,10350,8948.8,8458.7,8257.8,8208.6,8330.3,8218.6,8171.8,7734.9,7647.4,7920,8517.7,9409.6,10542,11855,13281,14728,16105,17327,18319,19022,19400,19412,19053,18804,18646,18602,18965,18813,18802,18632,18441,18389,18621,18426,18271,17262,16140,14959,13765,12595,11469,11299,11053,10989,11601,11495,11414,11334,11254,11174,11095,11016,10937,10859,10781,10704,10627,10550,10474,10398,10323,10248,10173,10099,10025,9951.9,9879,9806.4,9734.2,9662.5,9591.1,9520.1,9449.5,9379.3,9347,9450.7,9447.7,9381.3,9312.3,9243.7,9175.4,9107.6,9040.1,8973.1,8906.4,8840.2,8774.3,8708.8,8643.7,8579,8514.6,8450.7,8387.1,8323.9,8261.1,8198.7,8136.7,8075,8013.7,7952.8,7892.3,7832.1,7772.3,7712.9,7653.8,7595.1,7536.7,7478.8,7421.1,7363.9,7307,7250.5,7234.4,7369.2,7365.1,7308.7,7252.6,7196.9,7141.5,7086.5,7031.8,6977.5,6923.5,6869.9,6816.6,6763.7,6711.1,6658.8,6606.9,6555.3,6504.1,6453.2,6402.6,6352.4,6302.4,6252.9,6203.6,6154.6,6106,6057.7,6009.7,5962,5914.7,5867.6,5820.9,5774.5,5728.4,5682.6,5637.1,5591.9,5547,5502.4,5458.2,5413.9,5370,5326.4,5283,5240,5197.2,5154.8,5112.6,5070.7,5029.1,4987.9,4946.8,4906.1,4865.7,4825.5,4785.7,4746.1,4706.8,4667.7,4629,4590.5,4552.3,4514.4,4476.7,4439.4,4402.2,4365.4,4328.8,4292.5,4256.4,4220.6,4185,4149.7,4114.6,4079.8,4045.1,4006.3,3967.8,3929.6,3891.8,3854.3,3817.2,3780.4,3744,3707.9,3672.2,3636.7,3601.7,3566.9,3532.5,3498.4,3464.6,3431.2,3398,3365.2,3332.7,3300.6,3268.7,3237.2,3205.9,3175,3143.6,3111.6,3080,3048.7,3017.8,2987.2,2956.8,2926.8,2895.5,2864.2,2832.3,2800.7,2769.6,2738.8,2708.4,2678.4,2648.8,2619.5,2590.6,2560.9,2531.7,2502.8,2474.2,2446.1,2418.3,2390.8,2363.7,2336.9,2310.5,2284.4,2258.6,2233.1,2208,2183.2,2158.7,2134.5,2110.6,2087,2063.7,2040.7,2017.9,1995.5,1973.3,1951.4,1929.8,1908.4,1887.3,1866.5,1845.9,1825.6,1805.5,1785.7,1766.1,1746.8,1727.7,1708.8,1690.1,1671.7,1653.5,1635.6,1617.8,1600.3,1582.9,1565.8,1548.9,1532.2,1515.7,1499.4,1483.3,1467.4,1451.6,1436.1,1420.8,1405.6,1390.6,1375.8,1361.2,1346.7,1332.4,1318.3,1304.4,1290.6,1276.9,1263.5,1250.2,1237,1224.1,1211.2,1198.5,1186,1173.6,1161.4,1149.2,1137.3,1125.4,1113.8,1102.2,1090.8,1079.5,1068.3,1057.3,1046.4,1035.6,1025,1014.5,1004.1,993.77,983.6,973.55,963.61,953.79,944,934.19,924.49,914.91,905.43,896.07,887.6,2760.2,2759.9,2727.2,2694.6,2662.4,2630.6,2599.1,2568.1,2537.5,2531.6,3721.1,3691.3,3646.6,3602.4,3558.8,3515.7,3473.2,3431.2,3389.7,3348.8,3308.3,3268.3,3228.9,3189.9,3151.4,3113.4,3075.8,3038.7,3002.1,2965.8,2930,2894.6,2859.7,2825.2,2791.1,2757.4,2724.2,2691.3,2658.9,2626.8,2595.2,2564,2533.1,2502.6,2472.5,2442.8,2413.4,2384.4,2369.2,2766.5,2763.2,2729,2695.3,2661.9,2629,2596.5,2564.5,2532.8,2501.2,2469.7,2439.6,2410.5,2381.9,2353.7,2326,2298.6,2271.3,2244.4,2217.8,2191.6,2165.7,2140.2,2115.1,2090.2,2065.7,2041.5,2023.1,2017.5,2148,2130.3,2104,2078,2052.4,2027.1,2002.1,1977.2,1952.6,1928.5,1904.8,1881.6,1858.6,1836.1,1813.8,1791.8,1770.1,1748.6,1746.1,1805.1,1803.9,1782.4,1761.2,1740.3,1719.6,1699.1,1678.9,1658.9,1639.1,1619.5,1600,1580.7,1561.7,1542.9,1524.3,1505.9,1487.7,1469.8,1452,1434.4,1417.1,1399.9,1382.9,1366.2,1349.6,1333.2,1317,1301,1285.2,1269.6,1254.2,1238.9,1223.9,1209,1194.3,1179.9,1165.7,1151.6,1137.8,1124.1,1110.6,1097.3,1084.1,1071.1,1058.3,1045.6,1033.1,1020.7,1008.6,996.51,984.59,972.4,960.38,948.48,936.73,925.14,913.7,902.41,891.06,879.8,868.69,857.73,846.93,836.26,825.75,815.37,805.14,795.04,785.08,775.25,765.56,756,746.56,737.25,728.07,719.01,710.07,701.25,692.55,683.97,675.5,667.14,658.89,650.76,642.73,634.81,627,619.29,611.68,604.17,596.76,589.45,582.24,575.12,568.1,561.11,554.1,547.18,540.35,533.61,526.96,520.4,513.93,507.54,501.24,495.02,488.88,482.83,476.85,470.96,465.14,459.4,453.71,448.08,442.53,437.06,431.65,426.32,421.06,415.87,410.75,405.69,400.63,395.61,390.66,385.77,380.95,376.2,371.5,366.87,362.3,357.79,353.35,348.96,344.62,340.35,336.13,331.97,327.86,323.8,319.78,315.82,311.9,308.04,304.23,300.47,296.76,293.09,289.48,285.91,282.39,278.92,275.49,272.1,268.76,265.47,262.21,259,255.83,252.71,249.62,246.57,243.57,240.6,237.67,234.78,231.93,229.11,226.33,223.58,220.88,218.2,215.56,212.96,210.39,207.85,205.34,202.87,200.43,198.01,195.64,193.29,190.97,188.68,186.42,184.19,181.98,179.81,177.66,175.54,173.45,171.39,169.35,167.33,165.34,163.38,161.44,159.53,157.64,137.51,115.28,96.803,82.271,81.482,79.074,78.245,199.2,191.4,176.34,148.49,147.23,142.49,140.94,192.58,185.09,182.48,175.2,173.33,197.44,197.13,189.9,166.82,140.6,118.47,99.741,83.941,70.69,59.574,50.212,42.31,35.675,29.838,24.91,20.821,17.427,14.604,12.254,10.297,8.6643,7.2999,6.1489,5.1856,4.3731,3.6804,3.1025,2.6191,2.3512,2.2636,2.2411,9.5213,9.4892,9.1524,8.0709,6.7992,5.7312,4.835,4.0699,3.429,2.892,2.442,2.0646,1.7481,1.4801,1.2544,1.0655,0.90702,0.77411,0.66255,0.56882,0.49002,0.42367,0.36775,0.32056,0.28078,0.24828,0.22047,0.19664,0.17615])/10	# [µ/￼density] Total [m2 kg-1]
linear_coefficient_Ir_attenuation_interpolator = interp1d(energy_Ir_attenuation,mass_coefficient_Ir_attenuation*Irdensity,bounds_error=False,fill_value='extrapolate')	# [µ] Total [m-1] vs eV









#
