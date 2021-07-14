
# position of the visible structure and the fueling location for the overlays
# data for the fueling location
fueling_r = [[0.260]]
fueling_z = [[-0.264]]
fueling_t = [[105]]
fueling_r.append([0.260])
fueling_z.append([-0.264])
fueling_t.append([195])
# tile directly below the the string of bolts on the centre column
# neighbouring points
stucture_r=[[0.333]*11]
stucture_z=[[-1.304]*11]
stucture_t=[np.linspace(60,195+15,11)]
stucture_r.append([0.539]*11)
stucture_z.append([-1.505]*11)
stucture_t.append(np.linspace(60,195+15,11))
for value in np.linspace(60,195+15,11):
	stucture_r.append([0.333,0.539])
	stucture_z.append([-1.304,-1.505])
	stucture_t.append([value]*2)
# neighbouring points
stucture_r.append([0.305]*4)
stucture_z.append([-0.853]*4)
stucture_t.append([33,93,153,213])
# neighbouring points
stucture_r.append([0.270]*4)
stucture_z.append([-0.573]*4)
stucture_t.append([33,93,153,213])
for value in [33,93,153,213]:
	stucture_r.append([0.305,0.270])
	stucture_z.append([-0.853,-0.573])
	stucture_t.append([value]*2)
# tiles around the nose
# neighbouring points
stucture_r.append([0.898]*8)
stucture_z.append([-1.300]*8)
stucture_t.append(np.linspace(0,90+15,8))
# stucture_r.append([0.895]*7)
# stucture_z.append([-1.302]*7)
# stucture_t.append(np.linspace(0,90,7))
# stucture_r.append([0.840]*7)
# stucture_z.append([-1.383]*7)
# stucture_t.append(np.linspace(0,90,7))
# stucture_r.append([0.822]*7)
# stucture_z.append([-1.514]*7)
# stucture_t.append(np.linspace(0,90,7))
stucture_r.append([0.872]*8)
stucture_z.append([-1.581]*8)
stucture_t.append(np.linspace(0,90+15,8))
for value in np.linspace(0,90+15,8):
	stucture_r.append([0.898,0.895,0.840,0.822,0.872])
	stucture_z.append([-1.3,-1.302,-1.383,-1.514,-1.581])
	stucture_t.append([value]*5)
# neighbouring points
stucture_r.append([0.898]*7)
stucture_z.append([-1.300]*7)
stucture_t.append(np.linspace(0,90,7))
stucture_r.append([1.184]*7)
stucture_z.append([-1.013]*7)
stucture_t.append(np.linspace(0,90,7))
for value in np.linspace(0,90,7):
	stucture_r.append([0.898,1.184])
	stucture_z.append([-1.3,-1.013])
	stucture_t.append([value]*2)
# neighbouring points	# a coil
stucture_r.append([1.373]*100)
stucture_z.append([-0.886]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.361]*100)
stucture_z.append([-0.878]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.312]*100)
stucture_z.append([-0.878]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.305]*100)
stucture_z.append([-0.884]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.270]*100)
stucture_z.append([-0.945]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.268]*100)
stucture_z.append([-0.954]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.269]*100)
stucture_z.append([-1.009]*100)
stucture_t.append(np.linspace(0-15,150,100))
for value in np.linspace(0-15,150,100):
	stucture_r.append([1.373,1.361,1.312,1.305,1.270,1.268,1.269])
	stucture_z.append([-0.886,-0.878,-0.878,-0.884,-0.945,-0.954,-1.009])
	stucture_t.append([value]*7)
# neighbouring points	# a coil
stucture_r.append([1.736]*100)
stucture_z.append([-0.262]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.562]*100)
stucture_z.append([-0.262]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.554]*100)
stucture_z.append([-0.270]*100)
stucture_t.append(np.linspace(0-15,150,100))
stucture_r.append([1.554]*100)
stucture_z.append([-0.444]*100)
stucture_t.append(np.linspace(0-15,150,100))
for value in np.linspace(0-15,150,100):
	stucture_r.append([1.736,1.562,1.554,1.554])
	stucture_z.append([-0.262,-0.262,-0.270,-0.444])
	stucture_t.append([value]*4)
# silouette of the centre column
# neighbouring points
for value in [60,210]:
	stucture_r.append([0.906,0.539,0.333,0.333,0.305,0.270,0.261,0.261,0.261])
	stucture_z.append([-1.881,-1.505,-1.304,-1.103,-0.853,-0.573,-0.505,-0.271,-0.147])
	stucture_t.append([value]*9)
# neighbouring points	# super-x divertor tiles
stucture_r.append([1.391]*5)
stucture_z.append([-2.048]*5)
stucture_t.append(np.linspace(15-30,45-30,5))
# stucture_r.append([1.763]*5)
# stucture_z.append([-1.680]*5)
# stucture_t.append(np.linspace(15-30,45-30,5))
for value in np.linspace(15-30,45-30,5):
	stucture_r.append([1.391,1.549])
	stucture_z.append([-2.048,-1.861])
	stucture_t.append([value]*2)
# neighbouring points	# super-x divertor tiles
stucture_r.append([1.073]*3)
stucture_z.append([-2.060]*3)
stucture_t.append(np.linspace(0,30,3))
stucture_r.append([1.371]*3)
stucture_z.append([-2.060]*3)
stucture_t.append(np.linspace(0,30,3))
for value in np.linspace(0,30,3):
	stucture_r.append([1.073,1.371])
	stucture_z.append([-2.060,-2.060])
	stucture_t.append([value]*2)
