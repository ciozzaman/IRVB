
def efit_reconstruction_to_separatrix_on_foil(efit_reconstruction,refinement=1000):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	from scipy.interpolate import interp2d
	all_time_sep_r = []
	all_time_sep_z = []
	r_fine = np.unique(np.linspace(efit_reconstruction.R.min(),efit_reconstruction.R.max(),refinement).tolist() + np.linspace(R_centre_column-0.01,R_centre_column+0.08,refinement).tolist())
	z_fine = np.linspace(efit_reconstruction.Z.min(),efit_reconstruction.Z.max(),refinement)
	for time in range(len(efit_reconstruction.time)):
		gna = efit_reconstruction.psidat[time]
		sep_up = efit_reconstruction.upper_xpoint_p[time]
		sep_low = efit_reconstruction.lower_xpoint_p[time]
		x_point_z_proximity = np.abs(np.nanmin([efit_reconstruction.upper_xpoint_z[time],efit_reconstruction.lower_xpoint_z[time],-0.573-0.2]))-0.2	# -0.573 is an arbitrary treshold in case both are nan

		psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,gna)
		psi = psi_interpolator(r_fine,z_fine)
		psi_up = -np.abs(psi-sep_up)
		psi_low = -np.abs(psi-sep_low)
		all_peaks_up = []
		all_z_up = []
		all_peaks_low = []
		all_z_low = []
		for i_z,z in enumerate(z_fine):
			# psi_z = psi[i_z]
			peaks = find_peaks(psi_up[i_z])[0]
			all_peaks_up.append(peaks)
			all_z_up.append([i_z]*len(peaks))
			peaks = find_peaks(psi_low[i_z])[0]
			all_peaks_low.append(peaks)
			all_z_low.append([i_z]*len(peaks))
		all_peaks_up = np.concatenate(all_peaks_up).astype(int)
		all_z_up = np.concatenate(all_z_up).astype(int)
		found_psi_up = np.abs(psi_up[all_z_up,all_peaks_up])
		all_peaks_low = np.concatenate(all_peaks_low).astype(int)
		all_z_low = np.concatenate(all_z_low).astype(int)
		found_psi_low = np.abs(psi_low[all_z_low,all_peaks_low])

		# plt.figure()
		# plt.plot(all_z[found_points<(gna.max()-gna.min())/500],all_peaks[found_points<(gna.max()-gna.min())/500],'+b')
		all_peaks_up = all_peaks_up[found_psi_up<(gna.max()-gna.min())/500]
		all_z_up = all_z_up[found_psi_up<(gna.max()-gna.min())/500]
		all_peaks_low = all_peaks_low[found_psi_low<(gna.max()-gna.min())/500]
		all_z_low = all_z_low[found_psi_low<(gna.max()-gna.min())/500]

		left_up = []
		right_up = []
		left_up_z = []
		right_up_z = []
		left_low = []
		right_low = []
		left_low_z = []
		right_low_z = []
		for i_z,z in enumerate(z_fine):
			if i_z in all_z_up:
				temp = all_peaks_up[all_z_up==i_z]
				if len(temp) == 1:
					right_up.append(temp[0])
					right_up_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					left_up.append(temp.min())
					left_up_z.append(i_z)
					right_up.append(temp.max())
					right_up_z.append(i_z)
				elif len(temp) == 3:
					left_up.append(np.sort(temp)[1])
					left_up_z.append(i_z)
					right_up.append(temp.max())
					right_up_z.append(i_z)
				elif len(temp) == 4:
					left_up.append(np.sort(temp)[1])
					left_up_z.append(i_z)
					right_up.append(np.sort(temp)[2])
					right_up_z.append(i_z)
			if i_z in all_z_low:
				temp = all_peaks_low[all_z_low==i_z]
				if len(temp) == 1:
					right_low.append(temp[0])
					right_low_z.append(i_z)
				elif len(temp) == 2:
					# # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
					# if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
					left_low.append(temp.min())
					left_low_z.append(i_z)
					right_low.append(temp.max())
					right_low_z.append(i_z)
				elif len(temp) == 3:
					left_low.append(np.sort(temp)[1])
					left_low_z.append(i_z)
					right_low.append(temp.max())
					right_low_z.append(i_z)
				elif len(temp) == 4:
					left_low.append(np.sort(temp)[1])
					left_low_z.append(i_z)
					right_low.append(np.sort(temp)[2])
					right_low_z.append(i_z)
		# sep_r = [left_up,right_up,left_low,right_low]
		# sep_z = [left_up_z,right_up_z,left_low_z,right_low_z]
		all_time_sep_r.append([left_up,right_up,left_low,right_low])
		all_time_sep_z.append([left_up_z,right_up_z,left_low_z,right_low_z])
	return all_time_sep_r,all_time_sep_z,r_fine,z_fine


# silouette of the centre column
MASTU_silouette_z = [-1.881,-1.505,-1.304,-1.103,-0.853,-0.573,-0.505,-0.271,-0.147]
MASTU_silouette_r = [0.906,0.539,0.333,0.333,0.305,0.270,0.261,0.261,0.261]
from scipy.interpolate.interpolate import interp1d
R_centre_column_interpolator = interp1d(MASTU_silouette_z+(-np.flip(MASTU_silouette_z,axis=0)).tolist(),MASTU_silouette_r+np.flip(MASTU_silouette_r,axis=0).tolist(),fill_value=np.nan,bounds_error=False)

def return_all_time_strike_points_location(efit_reconstruction,all_time_sep_r,all_time_sep_z,r_fine,z_fine,resolution = 1000):
	from scipy.signal import find_peaks, peak_prominences as get_proms
	temp_R = np.ones((len(efit_reconstruction.time),20))*np.nan
	temp_Z = np.ones((len(efit_reconstruction.time),20))*np.nan
	for time in range(len(efit_reconstruction.time)):
		temp = np.array([efit_reconstruction.strikepointR[time],efit_reconstruction.strikepointZ[time]]).T
		if temp.max()<1e-1:
			a=np.concatenate([r_fine[all_time_sep_r[time][0]],r_fine[all_time_sep_r[time][2]]])
			b=np.concatenate([z_fine[all_time_sep_z[time][0]],z_fine[all_time_sep_z[time][2]]])
			# c=np.abs(a-R_centre_column)
			c=np.abs(a-R_centre_column_interpolator(b))
			peaks = find_peaks(-c)[0]
			peaks = peaks[c[peaks]<4e-3]
			# efit_reconstruction.strikepointR[time][:min(len(efit_reconstruction.strikepointR[time]),len(peaks))] = a[peaks][:min(len(efit_reconstruction.strikepointR[time]),len(peaks))]
			# efit_reconstruction.strikepointZ[time][:min(len(efit_reconstruction.strikepointZ[time]),len(peaks))] = b[peaks][:min(len(efit_reconstruction.strikepointZ[time]),len(peaks))]
			temp_R[time][:min(20,len(peaks))] = a[peaks][:min(20,len(peaks))]
			temp_Z[time][:min(20,len(peaks))] = b[peaks][:min(20,len(peaks))]
		else:
			temp_R[time][:min(20,len(efit_reconstruction.strikepointR[time]))] = efit_reconstruction.strikepointR[time][:min(20,len(efit_reconstruction.strikepointR[time]))]
			temp_Z[time][:min(20,len(efit_reconstruction.strikepointZ[time]))] = -np.abs(efit_reconstruction.strikepointZ[time][:min(20,len(efit_reconstruction.strikepointZ[time]))])
	return temp_R,temp_Z

