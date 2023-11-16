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
# more detailed explanation (iltimately unnecessary) at
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











#
