import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist

from copy import deepcopy

from collections import deque
# from params import 
from scipy.interpolate import interp1d

def calculate_velocity(xs, ys, ts, idx, n_f, n_b):
    # xs, ys, ts: traj in x, y axis
    # idx: idx that we want to estimate the velocity
    # n_f, n_b: data points at front, at back (eg. idx -n_f |____|____| idx + n_f)
    # Interpolate the data using a quadratic spline
    tck_x = interp1d(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], 
                     xs[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], kind='cubic', bounds_error=False, fill_value="extrapolate")
    tck_y = interp1d(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], 
                     ys[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Differentiate the splines to obtain the velocity in the x and y directions
    vxs = np.gradient(tck_x(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)]), 1/30)
    vys = np.gradient(tck_y(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)]), 1/30)
    vx, vy= vxs[min(n_f, idx)], vys[min(n_f, idx)]
    return vx,vy

########### HERMITE SPLINE 2D ##############

def get_spline(x0, x1, y0, y1, theta0, theta1, steps=100, t=0):
	if np.size(t)==1:
		t = np.linspace(0, 1, steps) # np.asarray([1, 5, 10]) #

	# SCALING CHANGES THE RESULT
	######################### scale for better curves ############################
	dx = x1-x0
	dy = y1-y0

	X0 = x0+0
	Y0 = y0+0

	x0 = 0
	y0 = 0

	if np.abs(dx)<np.abs(dy): # y changes more than x, so y should be 1
		# remove dy
		s = np.abs(dy)*1.5
		x1 = (dx+0.0)/s
		y1 = (dy+0.0)/s
		
	else: 
		# remove dx
		s = np.abs(dx)*1.5
		x1 = (dx+0.0)/s
		y1 = (dy+0.0)/s
	################################################################################

	# X = T*b
	# x  =   a*t**3 + b*t**2 + c*t + d
	# x' = 3*a*t**2 + 2*b*t  + c
	# x0 = 0  
	# y0 = 0
	# theta0 = 0
	dx0 = np.cos(theta0) # = c  # all change is at along x at start
	dy0 = np.sin(theta0)
	# x1 = 2  
	# y1 = 1
	# theta1 = 0+np.random.uniform(0.001, 0.01)
	dx1 = np.cos(theta1) # = c  # all change is at along x at start
	dy1 = np.sin(theta1)

	t0 = 0
	t1 = 1

	# TREAT X AND Y SEPARATE, SO THERE'S NOT A SINGULARITY

	Ax = np.asarray([[1, t0,   t0**2,   t0**3],  # x  @ 0
					[0, 1,  2*t0,    3*t0**2],  # x' @ 0
					[1, t1,   t1**2,   t1**3],  # x  @ 1
					[0, 1,  2*t1,    3*t1**2]]) # x' @ 1

	X = np.asarray([x0, dx0, x1, dx1]).transpose()
	bx = np.linalg.solve(Ax, X)

	Ay = np.asarray([[1, t0,   t0**2,   t0**3],  # x  @ 0
					[0, 1,  2*t0,    3*t0**2],  # x' @ 0
					[1, t1,   t1**2,   t1**3],  # x  @ 1
					[0, 1,  2*t1,    3*t1**2]]) # x' @ 1
	Y = np.asarray([y0, dy0, y1, dy1]).transpose()
	by = np.linalg.solve(Ay, Y)


	x = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(),bx)
	y = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(),by)

	x = X0+x*s
	y = Y0+y*s

	return x, y

def get_line(x0, x1, y0, y1, steps=100, t=0):
	if np.size(t)==1:
		t = np.linspace(0, 1, steps) # np.asarray([1, 5, 10]) #

	dx = x1-x0
	dy = y1-y0

	x = x0 + dx*t
	y = y0 + dy*t

	return x, y

def get_spline_timing(x, y, v0, timesteps, dt):
	xf = spline_length(x,y)
	# xf = x0 + v0*t + 0.5a*t^2
	T = timesteps*dt
	a = 2.0*(xf - v0*T)/(T**2)
	t = np.linspace(0, T, timesteps)
	# distance at each time step
	x = v0*t + 0.5*a*(t**2)
	v = v0 + a*t  
	normalized = x/x[-1]
	return normalized, v, a


def spline_length(x,y):
	xdiff = x[1:]-x[0:-1]
	ydiff = y[1:]-y[0:-1]
	length = np.sum(np.sqrt(xdiff**2 + ydiff**2))
	return length


#################################### CONVERTING ######################################################################

def convert_to_XYVxVy(traffic_pred):
	# converts XYthetaVbeta to XYVxVy
	num_cars, time_steps, five = np.shape(traffic_pred)
	XYVxVy = np.zeros((num_cars, time_steps, 4))
	# for c in range(num_cars):
	XYVxVy[:,:,0:2] = traffic_pred[:,:,0:2]
	XYVxVy[:,:,2] = traffic_pred[:,:,3]*np.cos(traffic_pred[:,:,2])
	XYVxVy[:,:,3] = traffic_pred[:,:,3]*np.sin(traffic_pred[:,:,2])
	return XYVxVy

def convert_to_5d(trajectory):
	# converts XYVxVy to XYthetaVbeta, steer is 0
	num_cars, time_steps, four = np.shape(trajectory)
	fiveD = np.zeros((num_cars, time_steps, 5))
	# for c in range(num_cars):
	fiveD[:,:,0:2] = trajectory[:,:,0:2]
	fiveD[:,:,2] = np.arctan2(trajectory[:,:,3], trajectory[:,:,2])
	# fiveD[:,:,3] = np.sqrt(np.sum((trajectory[:,:,2]**2+trajectory[:,:,3]**2)))
	fiveD[:,:,3] = np.sqrt((trajectory[:,:,2]**2+trajectory[:,:,3]**2))
	return fiveD

def convert_2d_to_5d(x, y, v): ##########################################################################################
	time_steps = np.size(x)
	fiveD = np.zeros((1, time_steps-1, 5))
	# for c in range(num_cars):
	fiveD[:,:,0] = x[:-1]
	fiveD[:,:,1] = y[:-1]
	xdiff = x[1:]-x[0:-1]
	ydiff = y[1:]-y[0:-1]

	fiveD[:,:,2] = np.arctan2(ydiff, xdiff)
	# fiveD[:,-1,2] = np.arctan2(ydiff[-1], xdiff[-1])
	fiveD[:,:,3] = np.sqrt((xdiff**2+ydiff**2)) # TODO: i think this needs to be scaled relative to dt 
	# fiveD[:,-1,3] = np.sqrt(np.sum(xdiff[-1]**2+ydiff[-1]**2))
	return fiveD

def subgoal_to_traj(ego, subgoal, time_steps, dt):

	shape = np.shape(subgoal)
	# if # print("x dist", ego[0], subgoal[-1,0,0]) going backward make alt traj!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if len(shape)==3:
		# print("x dist", ego[0], subgoal[-1,0,0])
		x, y = get_spline(ego[0], subgoal[-1,0,0], ego[1], subgoal[-1,0,1], ego[2], subgoal[-1,0,2], steps=100, t=0)
	else:
		# print("x dista", ego[0], subgoal[0,0])
		x, y = get_spline(ego[0], subgoal[0,0], ego[1], subgoal[0,1], ego[2], subgoal[0,2], steps=100, t=0)
	

	t, v, a = get_spline_timing(x, y, ego[3], time_steps+1, dt)
	

	if len(shape)==3:
		xs, ys = get_spline(ego[0], subgoal[-1,0,0], ego[1], subgoal[-1,0,1], ego[2], subgoal[-1,0,2], steps=time_steps+1, t=t)
	else:
		xs, ys = get_spline(ego[0], subgoal[0,0], ego[1], subgoal[0,1], ego[2], subgoal[0,2], steps=time_steps+1, t=t)
	

	trajectory = convert_to_XYVxVy(convert_2d_to_5d(xs, ys, v))

	return trajectory, a 


############################################# COLLISION CHECKING ######################################################


def boundary_collision(traj, world):
	corners = points_to_corners(traj, 0.7, 1.4) # match 366
	collision = 0
	risk = -10000 # 0 at collision
	for pts in corners:
		for p in pts:  
			if p[0]<world[0] or p[0]>world[1] or p[1]<world[2] or p[1]>world[3]:
				print("Wall!", corners, world)
				collision = 1
				risk = 0
				break
			else:
				dists = [np.abs(p[0]-world[0]), np.abs(p[0]-world[1]), np.abs(p[1]-world[2]), np.abs(p[1]-world[3])] 
				if -np.min(dists)>risk:
					risk = -np.min(dists)
		if collision == 1:
			print("boundary", np.shape(p), p[0], p[1])
			break
	return collision, risk



def points_to_corners(points, car_width = 1.89, car_length = 4.98, is_ego=0):
	# print("::::::::", points)
	# points is [traj_length] x [x, y, vx, vy]
	# return 4, traj_length, 2
	# The cars position is given as a point. Cars aren't actually point masses. car is 2x5 meters
	# car_width = 1.89 # common_modules/models/launch/accord_2x
	# car_length = 4.98
	dim, four = np.shape(points[:,:4])

	if is_ego:
		afront_pad =0.0
		aback_pad = 0.0
		bfront_pad = .0
		bback_pad = .0
	else:
		afront_pad =0.0
		aback_pad = 0.0
		bfront_pad = .0
		bback_pad = .0 # 6
	#
	#  c1     c2
	#
	#      p
	#
	#
	#  c3     c4
	#
	# points[:,0:2]+
	z = np.zeros((dim,2))
	c1 = z+[[-car_width/2-bback_pad, car_length/2+afront_pad]]
	C1 = np.hstack((c1,np.ones((dim,1))))
	c2 = z+[[car_width/2+bfront_pad, car_length/2+afront_pad]]
	C2 = np.hstack((c2,np.ones((dim,1))))
	c3 = z+[[-car_width/2-bback_pad, -car_length/2-aback_pad]]
	C3 = np.hstack((c3,np.ones((dim,1))))
	c4 = z+[[car_width/2+bfront_pad, -car_length/2-aback_pad]]
	C4 = np.hstack((c4,np.ones((dim,1))))
	angles = np.arctan2(points[:,2],points[:,3])
	# print angles
	for d in range(dim):
		R = np.asarray([[np.cos(angles[d]), -np.sin(angles[d]), 0],
						[np.sin(angles[d]), np.cos(angles[d]),  0],
						[0,                 0,                  1]])
		c1[d,:] = np.dot(C1[d,:],R)[0:2]
		c2[d,:] = np.dot(C2[d,:],R)[0:2]
		c3[d,:] = np.dot(C3[d,:],R)[0:2]
		c4[d,:] = np.dot(C4[d,:],R)[0:2]

	c1 += points[:,0:2]
	c2 += points[:,0:2]
	c3 += points[:,0:2]
	c4 += points[:,0:2]

	corners = []
	corners.append(c1)
	corners.append(c2)
	corners.append(c4)
	corners.append(c3) # order for not crossed polygon
	corners = np.asarray(corners)
	# print np.shape(corners)
	return corners


def points_to_corners_padded(points, car_width = 1.89, car_length = 4.98, front_pad=[0.0, 0.0, 0.0], rear_pad=[0.0, 0.0, 0.0], is_ego=0):
	# print("::::::::", np.shape(points))
	# points is [traj_length] x [x, y, vx, vy]

	# return 4, traj_length, 2
	# The cars position is given as a point. Cars aren't actually point masses. car is 2x5 meters
	# car_width = 1.89 # common_modules/models/launch/accord_2x
	# car_length = 4.98
	# print("front", front_pad)
	# print("rear", rear_pad)

	dim, four = np.shape(points) # dim = traj_length

	f_pad = front_pad[1]
	r_pad = front_pad[0]
	l_pad = rear_pad[0]
	b_pad = rear_pad[1] # 6
	# print(f_pad, r_pad, b_pad, l_pad)
	#  y
	#  c1     c2
	#
	#      p
	#	   
	#
	#  c3     c4 x 
	#
	# points[:,0:2]+
	z = np.zeros((dim,2))
	c1 = z+[[-car_width/2-l_pad, car_length/2+f_pad]]
	C1 = np.hstack((c1,np.ones((dim,1))))
	c2 = z+[[car_width/2+r_pad, car_length/2+f_pad]]
	C2 = np.hstack((c2,np.ones((dim,1))))
	c3 = z+[[-car_width/2-l_pad, -car_length/2-b_pad]]
	C3 = np.hstack((c3,np.ones((dim,1))))
	c4 = z+[[car_width/2+r_pad, -car_length/2-b_pad]]
	C4 = np.hstack((c4,np.ones((dim,1))))
	angles = np.arctan2(-points[:,3],points[:,2])
	# print("traffic angles,", angles, points)
	# print("::",points[:,2],points[:,3])

	for d in range(dim):
		R = np.asarray([[np.cos(angles[d]), -np.sin(angles[d]), 0],
						[np.sin(angles[d]), np.cos(angles[d]),  0],
						[0,                 0,                  1]])
		c1[d,:] = np.dot(C1[d,:],R)[0:2]
		c2[d,:] = np.dot(C2[d,:],R)[0:2]
		c3[d,:] = np.dot(C3[d,:],R)[0:2]
		c4[d,:] = np.dot(C4[d,:],R)[0:2]

	c1 += points[:,0:2]
	c2 += points[:,0:2]
	c3 += points[:,0:2]
	c4 += points[:,0:2]

	corners = []
	corners.append(c1)
	corners.append(c2)
	corners.append(c4)
	corners.append(c3) # order for not crossed polygon
	corners = np.asarray(corners)
	# print np.shape(corners)
	return corners




def combine_gaussians(mu1, mu2, sig1, sig2):
	# COMBINE GAUSSIANS
	eps = 0.001
	# print("combined gaussian",sig1,sig2)
	k = sig1**2/(sig1**2 + sig2**2 + eps)

	# MU
	mu_combined = mu1 + k*(mu2 - mu1)

	# SIG
	# print("sig",sig1,sig2)
	sig_combined = sig1**2 - k*(sig1**2)

	return mu_combined, sig_combined


def close_poly(corners):
	dims = np.shape(corners)
	if len(dims)==2:
		return np.vstack((corners,corners[0,:]))
	if len(dims)==3: # 4 x num_cars x features
		return np.vstack((corners,corners[0:1,:,:]))
	else:
		print("you need to fix close poly. len(dims)=",len(dims) )

########################################## OPTIMIZATION #####################################################


def project_vector(heading_vec, angle):

	# R = np.asarray([[np.cos(angle), -np.sin(angle), 0],
	# 				[np.sin(angle), np.cos(angle),  0],
	# 				[0,                 0,          1]])
	angle_vec = np.asarray([np.cos(angle), np.sin(angle), 0])
	proj = np.dot(heading_vec, angle_vec)

	return proj


def get_transform_matrix(location, rotation):
	"""
	Creates matrix from carla transform.
	"""

	c_y = np.cos(np.radians(rotation[2]))
	s_y = np.sin(np.radians(rotation[2]))
	c_r = np.cos(np.radians(rotation[1]))
	s_r = np.sin(np.radians(rotation[1]))
	c_p = np.cos(np.radians(rotation[0]))
	s_p = np.sin(np.radians(rotation[0]))
	matrix = np.matrix(np.identity(4))
	matrix[0, 3] = location[0]
	matrix[1, 3] = location[1]
	matrix[2, 3] = location[2]
	matrix[0, 0] = c_p * c_y
	matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
	matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
	matrix[1, 0] = s_y * c_p
	matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
	matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
	matrix[2, 0] = s_p
	matrix[2, 1] = -c_p * s_r
	matrix[2, 2] = c_p * c_r
	return matrix


def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [yaw, pitch, roll]

def project_point(pt, line_pts, with_dists=False):
	# pt IS THE POINT TO BE PROJECTED
	# line_pts IS THE TWO POINTS DEFINING A LINE, [0] -> [1]
	if len(line_pts)>2:
		# entire path was passed in, find the closest points
		dists = np.sum((pt - line_pts)**2,1)
		ind = np.argmin(dists)
		if ind==len(line_pts)-1:
			ind -= 1
		line_pts = line_pts[[ind,ind+1],:]

	# print(np.shape(line_pts)) # [1]-line_pts[0])
	v1 = np.atleast_2d(line_pts[1]-line_pts[0]) # x,y
	v2 = np.atleast_2d(pt - line_pts[0])
	val = np.dot(v2, v1.transpose())
	# get squared length of v1
	# lensq = v1.x * v1.x + v1.y * v1.y;
	lensq = np.dot(v1, v1.transpose())
	eps = 0.00001
	projected = line_pts[0] + (val * v1) / (lensq + eps)
	projected = projected[0]
	value = val / np.sqrt(lensq + eps)
	# print("lensq", lensq, value)
	# plt.figure(2)
	# plt.clf()
	# plt.plot([line_pts[0][0],line_pts[1][0]], [line_pts[0][1],line_pts[1][1]], '-ob')
	# plt.plot(pt[0], pt[1], 'ok')
	# plt.plot(projected[0], projected[1], 'or')
	# plt.show(block=False)
	# plt.pause(0.0000000001)
	if with_dists: 
		orth_dists = np.sqrt(np.sum((pt - projected)**2))
		return projected, val, line_pts, orth_dists
	else:
		return projected, val, line_pts

def convert_points(ego_pos, angle, pts):
	center = [50, 25]
	pts = pts - ego_pos 
	# print(np.shape(pts), len(pts), np.ones((len(pts),1)) )
	pts = np.hstack((pts,np.ones((len(pts),1)) )) 
	
	rot = np.asarray([[np.cos(-angle), -np.sin(-angle), 0],
			[np.sin(-angle), np.cos(-angle),0],
			[0			, 	0			,1]])
	rot_pts = np.dot(rot, pts.transpose()).transpose()	
	return center + rot_pts[:,:2]

def convert_to_circles(xyvTh, L, W, circle_offset, padding, ego=False, center_only=False):
	# can be a single car, a time seq, or many cars
	# L, W can be scalar or dimension of xyTh
	xyvTh = np.atleast_2d(xyvTh)
	num_pts, four = np.shape(xyvTh)
	R = W/2.0 * padding #1.25
	circle_offset = L * circle_offset
	if np.size(R)==1:
		R = R*np.ones((num_pts,1))
		# circle_offset = circle_offset*np.ones((num_pts,1))
	# TODO: accomodate trucks 
	# add = 
	rear =   xyvTh[:,0:2] - circle_offset*np.asarray([np.cos(xyvTh[:,3]), np.sin(xyvTh[:,3])]).transpose()
	center = xyvTh[:,0:2] 
	front =  xyvTh[:,0:2] + circle_offset*np.asarray([np.cos(xyvTh[:,3]), np.sin(xyvTh[:,3])]).transpose()

	inds = np.asarray([range(num_pts)]).transpose()
	# print("shapes", np.shape(rear), np.shape(R), np.shape(inds))
	rear =   np.hstack((rear,   xyvTh[:,2:], R, inds))
	center = np.hstack((center, xyvTh[:,2:], R, inds))
	front =  np.hstack((front,  xyvTh[:,2:], R, inds))

	if ego==True:

		front_pad2 = front[:,0:2] + (FRONT_PADDING + xyvTh[0,2]*VEL_PADDING)*np.asarray([np.cos(xyvTh[:,3]), np.sin(xyvTh[:,3])]).transpose()
		front_pad = front[:,0:2] + (FRONT_PADDING/2.0)*np.asarray([np.cos(xyvTh[:,3]), np.sin(xyvTh[:,3])]).transpose()
		back_pad =  rear[:,0:2] - (REAR_PADDING)*np.asarray([np.cos(xyvTh[:,3]), np.sin(xyvTh[:,3])]).transpose()
		front_pad = np.hstack((front_pad, xyvTh[:,2:], R, inds))
		back_pad =  np.hstack((back_pad, xyvTh[:,2:], R, inds))
		front_pad2 = np.hstack((front_pad2, xyvTh[:,2:], R, inds))

	if center_only==True:
		centers = center
	elif ego==True:
		centers = np.vstack((rear, center, front, front_pad2, front_pad, back_pad))
	else:

		front_pad2 = front[:,0:2] + (xyvTh[0,2]*TRAFFIC_VEL_PADDING)*np.asarray([np.cos(xyvTh[:,3]), np.sin(xyvTh[:,3])]).transpose()
		front_pad2 = np.hstack((front_pad2, xyvTh[:,2:], R, inds))
		centers = np.vstack((rear, center, front, front_pad2))


	return centers # [x,y,v,Th, R, ind(time)]

def car_corners(state,angle,length=5.0,width=2.0):

	pts = np.asarray([[+length/2, +width/2],
					[-length/2, +width/2],
					[-length/2, -width/2],
					[+length/2, -width/2],
					[+length/2, +width/2]])
	rot = np.asarray([[np.cos(angle), np.sin(angle)], # TODO: I'm A BIT CONCERNED ABOUT NEEDING THE SIGN FLIP ON SIN
					[-np.sin(angle), np.cos(angle)]])
	corners = state[0:2] + np.dot(pts, rot)
	return corners

def check_circle_sample_collision(ego_circles, traffic_circles):
	# # [x,y,v,Th,R,ind(time)] Note: ego[:,5] is a sample index, traffic[:,3] is a time step
	# # [x,y,v,Th,R,ind(time), prob]
	# print("collision", np.shape(ego_circles), np.shape(traffic_circles))
	# TODO, USE PROB=traffic_circles[future_ids,4]
	likely = traffic_circles[:,6]>0.05
	traffic_circles = traffic_circles[likely,:]

	times = np.unique(ego_circles[:,5]) # 
	# print("times:",times)
	num_samples = len(times)
	num_cars, six = np.shape(traffic_circles)
	comfy_dist = 10

	if num_cars==0:
		print("no cars")
		return np.zeros(num_samples), comfy_dist+np.ones(num_samples), np.zeros(num_samples)

	dists = cdist(ego_circles[:,:2], traffic_circles[:,:2], 'euclidean')
	dist = np.min(dists,1)
	ids = np.argmin(dists,1)

	# print("circles",ego_circles[:,4], traffic_circles[ids,4])
	closest = dist - ego_circles[:,4] - traffic_circles[ids,4]
	# print("dists",np.min(closest), np.shape(ego_circles))
	

	# num_samples = np.max(ego_circles[:,3])+1  
	# print("Num samples", num_samples)
	# CHECK SAFETY AT EACH TIME STEP
	safety_rating = np.zeros((num_samples))
	for s in range(num_samples): # samples indicator # TODO check this!!!!!!!!
		safety_rating[s] = np.min(closest[ego_circles[:,5]==times[s]])

	# print("safety",safety_rating)
	collisions = safety_rating<0
	# print("collision",np.sum(collisions))

	# comfy_dist = 10
	risk = comfy_dist - safety_rating  
	risk = np.maximum(risk, 0) # dist=0 -> risk=comfy_dist, dist>=comfy -> risk=0, dist<0 (collision) -> risk>comfy_dist
	# print("risk", risk, collision)
	return np.sum(collisions), np.min(safety_rating), np.max(risk)

def check_circle_collision(ego_circles, traffic_circles):
	# [x,y,R,time] Note: ego[:,3] is a time, traffic[:,3] is a time step
	# [x,y,R,time, prob]
	# print("collision", np.shape(ego_circles), np.shape(traffic_circles))
	# TODO, USE PROB=traffic_circles[future_ids,4]
	comfy_dist = 10
	if np.size(traffic_circles)==0 or np.size(ego_circles)==0:
		print("no cars")
		return np.asarray([0]), np.asarray([comfy_dist]), np.asarray([0])


	likely = traffic_circles[:,4]>0.05
	traffic_circles = traffic_circles[likely,:]

	# num_samples = len(np.unique(ego_circles[:,3]))
	num_cars, five = np.shape(traffic_circles)

	# if num_cars==0 or np.size(ego_circles)==0:
	# 	print("no cars")
	# 	return np.asarray([0]), np.asarray([comfy_dist]), np.asarray([0])

	dists = cdist(ego_circles[:,:2], traffic_circles[:,:2], 'euclidean')
	dist = np.min(dists,1)
	ids = np.argmin(dists,1)

	# print("derp", np.shape(ego_circles[:,2]), np.shape(traffic_circles[ids,2]))
	closest = dist - ego_circles[:,2] - traffic_circles[ids,2]
	print("closest:", closest)
	# num_samples = np.max(ego_circles[:,3])+1  
	
	safety_rating = np.min(closest)

	collisions = safety_rating<0

	comfy_dist = 10
	risk = comfy_dist - safety_rating  
	risk = np.maximum(risk, 0) # dist=0 -> risk=comfy_dist, dist>=comfy -> risk=0, dist<0 (collision) -> risk>comfy_dist
	# print("risk", risk, collisions)
	return collisions, safety_rating, risk