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
files8=['/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000027','/home/ffederic/work/irvb/flatfield/Mar08_2018/ff_full-000028','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000029','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000030','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000031','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000032','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000033','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000034','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000035','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000036','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000037','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000038','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000039','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000040','/home/ffederic/work/irvb/flatfield/Mar09_2018/ff_full-000041']
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
power_interpolator1 = interp1d([0,0.0079,0.5,10],[0,0,4.16*1e-3,4.16*1e-3])

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
