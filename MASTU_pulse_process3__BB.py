# Here I only rotate and crop

from PIL import Image
MASTU_wireframe = Image.open("/home/ffederic/work/irvb/MAST-U/Calcam_theoretical_view.png")
MASTU_wireframe_resize = np.array(np.asarray(MASTU_wireframe)[:,:,3].tolist()).astype(np.float64)
MASTU_wireframe_resize[MASTU_wireframe_resize>0] = 1
MASTU_wireframe_resize = np.flip(MASTU_wireframe_resize,axis=0)
masked = np.ma.masked_where(MASTU_wireframe_resize == 0, MASTU_wireframe_resize)

# foil_position_dict = dict([('angle',0.9),('foilcenter',[158,136]),('foilhorizwpixel',240)])	# fixed orientation, for now, this is from 2021-06-04/44168
foil_position_dict = dict([('angle',0.7),('foilcenter',[157,136]),('foilhorizwpixel',240)])	# modified 2021/09/21 to match sensitivity matrix


print(laser_to_analyse[:-4]+' rotation missing, rorating')
# full_saved_file_dict.allow_pickle=True
# full_saved_file_dict = dict(laser_dict)

# time_partial = []
# laser_temperature_no_dead_pixels = []
# laser_temperature_std_no_dead_pixels = []
# for i in range(len(laser_digitizer_ID)):
# 	laser_temperature_no_dead_pixels.append(( full_saved_file_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_median'] + full_saved_file_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_minus_median'].astype(np.float32).T ).T)
# 	laser_temperature_std_no_dead_pixels.append(( full_saved_file_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_median'] + full_saved_file_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_minus_median'].astype(np.float32).T ).T)
# 	time_partial.append(full_saved_file_dict['full_frame'].all()[str(laser_digitizer_ID[i])]['time'])
if False:	# done in MASTU_pulse_process2.py now
	# I'm going to use the reference frames for foil position
	testrot=reference_background_temperature_no_dead_pixels[0]

	rotangle = foil_position_dict['angle'] #in degrees
	foilrot=rotangle*2*np.pi/360
	foilrotdeg=rotangle
	foilcenter = foil_position_dict['foilcenter']
	foilhorizw=0.09	# m
	foilvertw=0.07	# m
	foilhorizwpixel = foil_position_dict['foilhorizwpixel']
	foilvertwpixel=int(round((foilhorizwpixel*foilvertw)/foilhorizw))
	r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	a=foilvertwpixel/np.cos(foilrot)
	tgalpha=np.tan(foilrot)
	delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	foilxint=(np.rint(foilx)).astype('int')
	foilyint=(np.rint(foily)).astype('int')
	# rotangle = foil_position_dict['angle'] #in degrees
	# foilrot=rotangle*2*np.pi/360
	# foilrotdeg=rotangle
	# foilcenter = foil_position_dict['foilcenter']
	# foilhorizw=0.09	# m
	# foilvertw=0.07	# m
	# foilhorizwpixel = foil_position_dict['foilhorizwpixel']
	# foilvertwpixel=int((foilhorizwpixel*foilvertw)//foilhorizw)
	# r=((foilhorizwpixel**2+foilvertwpixel**2)**0.5)/2  # HALF DIAGONAL
	# a=foilvertwpixel/np.cos(foilrot)
	# tgalpha=np.tan(foilrot)
	# delta=-(a**2)/4+(1+tgalpha**2)*(r**2)
	# foilx=np.add(foilcenter[0],[(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha-delta**0.5)/(1+tgalpha**2),(0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2),(-0.5*a*tgalpha+delta**0.5)/(1+tgalpha**2)])
	# foily=np.add(foilcenter[1]-tgalpha*foilcenter[0],[tgalpha*foilx[0]+a/2,tgalpha*foilx[1]+a/2,tgalpha*foilx[2]-a/2,tgalpha*foilx[3]-a/2,tgalpha*foilx[0]+a/2])
	# foilxint=(np.rint(foilx)).astype('int')
	# foilyint=(np.rint(foily)).astype('int')

	plt.figure(figsize=(20, 10))
	plt.title('Foil search in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel')
	plt.imshow(testrot,'rainbow',origin='lower')
	plt.ylabel('Horizontal axis [pixles]')
	plt.xlabel('Vertical axis [pixles]')
	temp = np.sort(testrot[testrot>0])
	plt.clim(vmin=np.nanmean(temp[:max(20,int(len(temp)/20))]), vmax=np.nanmean(temp[-max(20,int(len(temp)/20)):]))
	# plt.clim(vmax=27.1,vmin=26.8)
	# plt.clim(vmin=np.nanmin(testrot[testrot>0]), vmax=np.nanmax(testrot))
	plt.colorbar().set_label('counts [au]')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.savefig(laser_to_analyse[:-4]+'_foil_fit.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(20, 10))
	temp = np.max(laser_temperature_no_dead_pixels[0][:,:,:200],axis=(-1,-2)).argmax()
	plt.title('Foil search in '+laser_to_analyse+'\nFoil center '+str(foilcenter)+', foil rot '+str(foilrotdeg)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel\nhottest frame at %.4gsec' %(time_partial[0][temp]))
	plt.imshow(laser_temperature_no_dead_pixels[0][temp],'rainbow',origin='lower',vmax=np.max(laser_temperature_no_dead_pixels[0][:,:,:200],axis=(-1,-2))[temp])
	plt.ylabel('Horizontal axis [pixles]')
	plt.xlabel('Vertical axis [pixles]')
	plt.clim(vmax=np.max(laser_temperature_no_dead_pixels[0][:,:,:200],axis=(-1,-2))[temp])
	plt.colorbar().set_label('temp increase [K]')
	plt.plot(foilx,foily,'r')
	plt.plot(foilcenter[0],foilcenter[1],'k+',markersize=30)
	plt.savefig(laser_to_analyse[:-4]+'_foil_fit_plasma.eps', bbox_inches='tight')
	plt.close('all')
	testrotback=rotate(testrot,foilrotdeg,axes=(-1,-2))
	precisionincrease=10
	dummy=np.ones(np.multiply(np.shape(testrot),precisionincrease))
	dummy[foilcenter[1]*precisionincrease,foilcenter[0]*precisionincrease]=2
	dummy[int(foily[0]*precisionincrease),int(foilx[0]*precisionincrease)]=3
	dummy[int(foily[1]*precisionincrease),int(foilx[1]*precisionincrease)]=4
	dummy[int(foily[2]*precisionincrease),int(foilx[2]*precisionincrease)]=5
	dummy[int(foily[3]*precisionincrease),int(foilx[3]*precisionincrease)]=6
	dummy2=rotate(dummy,foilrotdeg,axes=(-1,-2),order=0)
	foilcenterrot=(np.rint([np.where(dummy2==2)[1][0]/precisionincrease,np.where(dummy2==2)[0][0]/precisionincrease])).astype('int')
	foilxrot=(np.rint([np.where(dummy2==3)[1][0]/precisionincrease,np.where(dummy2==4)[1][0]/precisionincrease,np.where(dummy2==5)[1][0]/precisionincrease,np.where(dummy2==6)[1][0]/precisionincrease,np.where(dummy2==3)[1][0]/precisionincrease])).astype('int')
	foilyrot=(np.rint([np.where(dummy2==3)[0][0]/precisionincrease,np.where(dummy2==4)[0][0]/precisionincrease,np.where(dummy2==5)[0][0]/precisionincrease,np.where(dummy2==6)[0][0]/precisionincrease,np.where(dummy2==3)[0][0]/precisionincrease])).astype('int')
	# plt.plot(foilcenterrot[0],foilcenterrot[1],'k+',markersize=30)
	# plt.plot(foilxrot,foilyrot,'r')
	# plt.title('Foil center '+str(foilcenterrot)+', foil rot '+str(0)+'deg, foil size '+str([foilhorizwpixel,foilvertwpixel])+'pixel',size=9)
	# plt.colorbar().set_label('counts [au]')
	# plt.pause(0.01)

	foillx=min(foilxrot)
	foilrx=max(foilxrot)
	foilhorizwpixel=foilrx-foillx
	foildw=min(foilyrot)
	foilup=max(foilyrot)
	foilvertwpixel=foilup-foildw

	out_of_ROI_mask = np.ones_like(testrotback)
	out_of_ROI_mask[testrotback<np.nanmin(testrot[testrot>0])]=np.nan
	out_of_ROI_mask[testrotback>np.nanmax(testrot[testrot>0])]=np.nan
	a = generic_filter((testrotback),np.std,size=[19,19])
	out_of_ROI_mask[a>np.mean(a)]=np.nan
else:
	pass


# rotation and crop
# laser_temperature_no_dead_pixels_crop = []
# laser_temperature_std_no_dead_pixels_crop = []
# for i in range(len(laser_digitizer_ID)):
# 	laser_temperature_no_dead_pixels_rot=rotate(laser_temperature_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	laser_temperature_std_no_dead_pixels_rot=rotate(laser_temperature_std_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
# 		laser_temperature_std_no_dead_pixels_rot*=out_of_ROI_mask
# 		laser_temperature_std_no_dead_pixels_rot[np.logical_and(laser_temperature_no_dead_pixels_rot<np.nanmin(laser_temperature_no_dead_pixels[i]),laser_temperature_no_dead_pixels_rot>np.nanmax(laser_temperature_no_dead_pixels[i]))]=0
# 		laser_temperature_no_dead_pixels_rot*=out_of_ROI_mask
# 		laser_temperature_no_dead_pixels_rot[np.logical_and(laser_temperature_no_dead_pixels_rot<np.nanmin(laser_temperature_no_dead_pixels[i]),laser_temperature_no_dead_pixels_rot>np.nanmax(laser_temperature_no_dead_pixels[i]))]=0
# 	laser_temperature_no_dead_pixels_crop.append(laser_temperature_no_dead_pixels_rot[:,foildw:foilup,foillx:foilrx])
# 	laser_temperature_std_no_dead_pixels_crop.append(laser_temperature_std_no_dead_pixels_rot[:,foildw:foilup,foillx:foilrx])
laser_temperature_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(laser_temperature_no_dead_pixels,3,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
laser_temperature_std_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(laser_temperature_std_no_dead_pixels,3,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
nan_ROI_mask = np.isfinite(np.nanmedian(laser_temperature_no_dead_pixels_crop[0][:10],axis=0))


# temp_counts_no_dead_pixels_crop = []
# temp_counts_std_no_dead_pixels_crop = []
# temp_ref_counts_no_dead_pixels_crop = []
# temp_ref_counts_std_no_dead_pixels_crop = []
# BB_proportional_no_dead_pixels_crop = []
# BB_proportional_std_no_dead_pixels_crop = []
# for i in range(len(laser_digitizer_ID)):
# 	temp_counts_no_dead_pixels_rot=rotate(temp_counts_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	temp_counts_std_no_dead_pixels_rot=rotate(temp_counts_std_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	temp_ref_counts_no_dead_pixels_rot=rotate(temp_ref_counts_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	temp_ref_counts_std_no_dead_pixels_rot=rotate(temp_ref_counts_std_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	BB_proportional_no_dead_pixels_rot=rotate(BB_proportional_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	BB_proportional_std_no_dead_pixels_rot=rotate(BB_proportional_std_no_dead_pixels[i],foilrotdeg,axes=(-1,-2))
# 	if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
# 		temp_ref_counts_no_dead_pixels_rot*=out_of_ROI_mask
# 		temp_ref_counts_no_dead_pixels_rot[np.logical_and(temp_counts_no_dead_pixels_rot<np.nanmin(temp_counts_no_dead_pixels[i]),temp_counts_no_dead_pixels_rot>np.nanmax(temp_counts_no_dead_pixels[i]))]=0
# 		temp_ref_counts_std_no_dead_pixels_rot*=out_of_ROI_mask
# 		temp_ref_counts_std_no_dead_pixels_rot[np.logical_and(temp_counts_no_dead_pixels_rot<np.nanmin(temp_counts_no_dead_pixels[i]),temp_counts_no_dead_pixels_rot>np.nanmax(temp_counts_no_dead_pixels[i]))]=0
# 		BB_proportional_no_dead_pixels_rot*=out_of_ROI_mask
# 		BB_proportional_no_dead_pixels_rot[np.logical_and(temp_counts_no_dead_pixels_rot<np.nanmin(temp_counts_no_dead_pixels[i]),temp_counts_no_dead_pixels_rot>np.nanmax(temp_counts_no_dead_pixels[i]))]=0
# 		BB_proportional_std_no_dead_pixels_rot*=out_of_ROI_mask
# 		BB_proportional_std_no_dead_pixels_rot[np.logical_and(temp_counts_no_dead_pixels_rot<np.nanmin(temp_counts_no_dead_pixels[i]),temp_counts_no_dead_pixels_rot>np.nanmax(temp_counts_no_dead_pixels[i]))]=0
# 		temp_counts_std_no_dead_pixels_rot*=out_of_ROI_mask
# 		temp_counts_std_no_dead_pixels_rot[np.logical_and(temp_counts_no_dead_pixels_rot<np.nanmin(temp_counts_no_dead_pixels[i]),temp_counts_no_dead_pixels_rot>np.nanmax(temp_counts_no_dead_pixels[i]))]=0
# 		temp_counts_no_dead_pixels_rot*=out_of_ROI_mask
# 		temp_counts_no_dead_pixels_rot[np.logical_and(temp_counts_no_dead_pixels_rot<np.nanmin(temp_counts_no_dead_pixels[i]),temp_counts_no_dead_pixels_rot>np.nanmax(temp_counts_no_dead_pixels[i]))]=0
# 	temp_counts_no_dead_pixels_crop.append(temp_counts_no_dead_pixels_rot[:,foildw:foilup,foillx:foilrx])
# 	temp_counts_std_no_dead_pixels_crop.append(temp_counts_std_no_dead_pixels_rot[:,foildw:foilup,foillx:foilrx])
# 	temp_ref_counts_no_dead_pixels_crop.append(temp_ref_counts_no_dead_pixels_rot[foildw:foilup,foillx:foilrx])
# 	temp_ref_counts_std_no_dead_pixels_crop.append(temp_ref_counts_std_no_dead_pixels_rot[foildw:foilup,foillx:foilrx])
# 	BB_proportional_no_dead_pixels_crop.append(BB_proportional_no_dead_pixels_rot[foildw:foilup,foillx:foilrx])
# 	BB_proportional_std_no_dead_pixels_crop.append(BB_proportional_std_no_dead_pixels_rot[foildw:foilup,foillx:foilrx])
temp_counts_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(temp_counts_no_dead_pixels,3,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
temp_counts_std_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(temp_counts_std_no_dead_pixels,3,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
temp_ref_counts_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(temp_ref_counts_no_dead_pixels,2,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
temp_ref_counts_std_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(temp_ref_counts_std_no_dead_pixels,2,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
BB_proportional_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(BB_proportional_no_dead_pixels,2,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
BB_proportional_std_no_dead_pixels_crop = coleval.rotate_and_crop_multi_digitizer(BB_proportional_std_no_dead_pixels,2,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)


# counts_full_rot=rotate(counts_full,foilrotdeg,axes=(-1,-2))
# if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
# 	counts_full_rot*=out_of_ROI_mask
# 	counts_full_rot[np.logical_and(counts_full_rot<np.nanmin(counts_full),counts_full_rot>np.nanmax(counts_full))]=0
# counts_full_extended_crop = counts_full_rot[:,foildw-10:foilup+10,foillx-10:foilrx+10]
counts_full_extended_crop = coleval.rotate_and_crop_3D(counts_full,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw-10,foilup+10,foillx-10,foilrx+10)

# counts_full_crop = counts_full_rot[:,foildw:foilup,foillx:foilrx]
plt.figure(figsize=(10, 5))
# horizontal_displacement = np.diff((np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2).T/(np.sum(laser_counts[i][:,ROI_horizontal[0]:ROI_horizontal[1],ROI_horizontal[2]:ROI_horizontal[3]],axis=2)[:,0])).T,axis=1).argmin(axis=1)
horizontal_displacement = np.diff((np.sum(counts_full_extended_crop[:,-15:-5,60:110],axis=2)),axis=1).argmin(axis=1)
horizontal_displacement = horizontal_displacement-horizontal_displacement[10]
# vertical_displacement = np.diff((np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1).T/(np.sum(laser_counts[i][:,ROI_vertical[0]:ROI_vertical[1],ROI_vertical[2]:ROI_vertical[3]],axis=1)[:,0])).T,axis=1).argmax(axis=1)
vertical_displacement = np.diff((np.sum(counts_full_extended_crop[:,110:180,5:15],axis=1)),axis=1).argmax(axis=1)
vertical_displacement = vertical_displacement-vertical_displacement[10]
plt.plot(time_full_int,horizontal_displacement,label='horizontal',color=color[0])
plt.plot(time_full_int,vertical_displacement,label='vertical',color=color[1])
plt.xlabel('time [s]')
plt.ylabel('pixels of displacement [au]')
plt.title('Displacement of the image\nlooking for the effect of disruptions')
plt.legend(loc='best', fontsize='x-small')
plt.savefig(laser_to_analyse[:-4]+'_2.eps', bbox_inches='tight')
plt.close('all')

# reference_background_temperature_rot=rotate(reference_background_temperature_no_dead_pixels,foilrotdeg,axes=(-1,-2))
# reference_background_temperature_std_rot=rotate(reference_background_temperature_std_no_dead_pixels,foilrotdeg,axes=(-1,-2))
# if not (laser_dict['height']==max_ROI[0][1]+1 and laser_dict['width']==max_ROI[1][1]+1):
# 	reference_background_temperature_std_rot*=out_of_ROI_mask
# 	reference_background_temperature_std_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=0
# 	reference_background_temperature_rot*=out_of_ROI_mask
# 	reference_background_temperature_rot[np.logical_and(reference_background_temperature_rot<np.nanmin(reference_background_temperature_no_dead_pixels),reference_background_temperature_rot>np.nanmax(reference_background_temperature_no_dead_pixels))]=np.nanmean(reference_background_temperature_no_dead_pixels)
# reference_background_temperature_crop=reference_background_temperature_rot[:,foildw:foilup,foillx:foilrx]
# reference_background_temperature_std_crop=reference_background_temperature_std_rot[:,foildw:foilup,foillx:foilrx]
reference_background_temperature_crop = coleval.rotate_and_crop_3D(reference_background_temperature_no_dead_pixels,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)
reference_background_temperature_std_crop = coleval.rotate_and_crop_3D(reference_background_temperature_std_no_dead_pixels,foilrotdeg,max_ROI,laser_dict['height'],laser_dict['width'],out_of_ROI_mask,foildw,foilup,foillx,foilrx)


for i in range(len(laser_digitizer_ID)):
	ani = coleval.movie_from_data(np.array([np.flip(np.transpose(laser_temperature_no_dead_pixels_crop[i],(0,2,1)),axis=2)]), laser_framerate/len(laser_digitizer_ID),integration=laser_int_time/1000,time_offset=time_partial[i][0],xlabel='horizontal coord [pixels]', ylabel='vertical coord [pixels]',barlabel='foil temp [C]', prelude='shot ' + laser_to_analyse[-9:-4] + '\n')
	ani.save(laser_to_analyse[:-4] + '_temp_dig'+str(laser_digitizer_ID[i])+'.mp4', fps=5*laser_framerate/len(laser_digitizer_ID)/383, writer='ffmpeg',codec='mpeg4')
	plt.close('all')
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])] = dict([])
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_crop_minus_median'] = np.float16(laser_temperature_no_dead_pixels_crop[i].T-np.median(laser_temperature_no_dead_pixels_crop[i],axis=(-1,-2))).T
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['laser_temperature_no_dead_pixels_crop_median'] = np.median(laser_temperature_no_dead_pixels_crop[i],axis=(-1,-2))
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_crop_minus_median'] = np.float16(laser_temperature_std_no_dead_pixels_crop[i].T-np.median(laser_temperature_std_no_dead_pixels_crop[i],axis=(-1,-2))).T
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['laser_temperature_std_no_dead_pixels_crop_median'] = np.median(laser_temperature_std_no_dead_pixels_crop[i],axis=(-1,-2))
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['reference_background_temperature_crop'] = reference_background_temperature_crop[i]
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['reference_background_temperature_std_crop'] = reference_background_temperature_std_crop[i]
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['temp_counts_no_dead_pixels_crop_minus_median'] = np.float16(temp_counts_no_dead_pixels_crop[i].T-np.median(temp_counts_no_dead_pixels_crop[i],axis=(-1,-2))).T
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['temp_counts_no_dead_pixels_crop_median'] = np.median(temp_counts_no_dead_pixels_crop[i],axis=(-1,-2))
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['temp_counts_std_no_dead_pixels_crop_minus_median'] = np.float16(temp_counts_std_no_dead_pixels_crop[i].T-np.median(temp_counts_std_no_dead_pixels_crop[i],axis=(-1,-2))).T
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['temp_counts_std_no_dead_pixels_crop_median'] = np.median(temp_counts_std_no_dead_pixels_crop[i],axis=(-1,-2))
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['temp_ref_counts_no_dead_pixels_crop'] = temp_ref_counts_no_dead_pixels_crop[i]
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['temp_ref_counts_std_no_dead_pixels_crop'] = temp_ref_counts_std_no_dead_pixels_crop[i]
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['BB_proportional_no_dead_pixels_crop'] = BB_proportional_no_dead_pixels_crop[i]
	full_saved_file_dict['only_foil'][str(laser_digitizer_ID[i])]['BB_proportional_std_no_dead_pixels_crop'] = BB_proportional_std_no_dead_pixels_crop[i]
full_saved_file_dict['only_foil']['nan_ROI_mask'] = nan_ROI_mask

np.savez_compressed(laser_to_analyse[:-4],**full_saved_file_dict)

try:
	del laser_temperature_no_dead_pixels_crop,laser_temperature_std_no_dead_pixels_crop,laser_temperature_no_dead_pixels,laser_temperature_std_no_dead_pixels
	del laser_temperature,laser_temperature_std,laser_temperature_minus_background,laser_temperature_std_minus_background,laser_temperature_full
	del laser_temperature_std_full,laser_temperature_minus_background_full,laser_temperature_std_minus_background_full,laser_counts_filtered
except:
	print('wrong memory cleaning')

print('completed rotating/cropping ' + laser_to_analyse)
