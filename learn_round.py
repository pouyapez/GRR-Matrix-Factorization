import numpy as np
import random
import sys
from numpy.linalg import norm
import logging
import math


def logistic(x):
	if x>=0:
		return 1
	else:
		return 0


def g_logistic(x): return 50.0/((np.exp(50*x/2)+np.exp(-50*x/2))*(np.exp(50*x/2)+np.exp(-50*x/2)))

def hinge_inverse(x):
	if x<0:
		return 0
	elif x>=0.01:
		return x-0.005
	else:
		return 50*(x)**2



def ghinge_inverse(x):
	if x<0:
		return 0
	elif x>=0.01:
		return 1
	else:
		return 100*x

def pred(Y,U,V,B,link):
	import math
	D=len(B)+1
	Yh = []
	err = 0.0
	n=0
	p=0
	print B
	for (i,j,v) in Y:
		n=n+1
		ui = U[i]
		vj = V[j]
		uv = np.dot(ui, vj)
		vhat = 0.0 
		for d in xrange(D-1):
			vhat += link(uv - B[d])
		Yh.append((i,j,vhat))
		err += pow(v-vhat,2) #*(v-vhat)
		if abs(v-vhat)<=0.5:
			p=p+1
	p=p+0.0
	print p,n
	return Yh,math.sqrt(err/len(Y)),p/n


def get_link(link_func):
	if link_func == "Linear":
		return (identity, g_identity)
	if link_func == "Multi-sigmoid":
		return (logistic, g_logistic)
	return None

def chunks(l, n):
	""" Yield successive n-sized chunks from l.
	"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]



def objective(Y,U,V,B,link,beta):
	D=len(B)+1
	C = 0.0
	for (i,j,v) in Y:
		v=int(v)
		if v==D-1:
			vnew_up=B[D-2]+1## we are tring to find the upper bound of interval base on v / posetive infinity
		else:
			vnew_up=B[v]
		if v==0:
			vnew_down=B[0]-1## Lower bounds/ negetive infinity
		else:
			vnew_down=B[v-1]
			ui = U[i]
		vj = V[j]
		uv = np.dot(ui, vj)
		vhat = 0.0
		C += hinge_inverse(vnew_down-uv)+hinge_inverse(uv-vnew_up)
	C=C+beta*np.linalg.norm(U, ord=None)+beta*np.linalg.norm(V, ord=None)	
	return C


def read_params(dir,itr):
	import gzip
	import cPickle as pickle
	Uf=gzip.open(dir + "U." + str(itr) + ".pkl")
	U=pickle.load(Uf)
	Uf.close()
	Vf=gzip.open(dir + "V." + str(itr) + ".pkl")
	V=pickle.load(Vf)
	Vf.close()
	Bf=gzip.open(dir + "B." + str(itr) + ".pkl")
	B=pickle.load(Bf)
	Bf.close()
	return U,V,B



def learn(Y, n, m, k, link, g_link, iters=5000, D=2, init_step=0.01,update_B = True,tau=True,
	beta=0.0000001, mini_batch_size=-1, iter_f=None, debug_every=1):
	"""Returns learnt parameters of matrix factorization. (A=U.V^T , returns U and V and a vector B as the thresholds)
	Y - array of matrix cells: (i,j,v)
	n - number of rows of Y
	m - number of columns of Y
	k - rank of the factorizationv
	link - link function to use
	g_link - gradient of the link function
	iters - number of iterations
	"""
	if tau:
		U,V,B=rnd_init(n,m,k,D,Y,link,g_link,init_step,beta,mini_batch_size,iters/1000)
	else:
		U,V,B=rnd_init(n,m,k,D,Y,link,g_link,init_step,beta,mini_batch_size,iters/1000,tau=False)
	itr=19
	errs = []
	Cs=[]
	logging.info("Starting training")
	p=0
	e=0
	for itr in xrange(iters):
		p=p+1
		if (itr)%debug_every == 0: logging.info("Starting iteration %d of %d", (itr+1), iters)
		if update_B:
			U,V,B,err,C= learn_iter(Y, n, m, k, link, g_link, U, V, B, D, init_step, beta, mini_batch_size,iteration=p)
		else:
			U,V,B,err,C= learn_iter(Y, n, m, k, link, g_link, U, V, B, D, init_step, beta, mini_batch_size,iteration=p,update_B=False)
		errs.append(err)
		Cs.append(C)
		if p<0:
			logging.info("%f %f %f %s", err, norm(U), norm(V), str(B))
			if p < 100:
				Yh = []
				Yp=[]
				erp = 0.0
				for (i,j,v) in Y:
					ui = U[i]
					vj = V[j]
					uv = np.dot(ui, vj)
					vhat = 0.0
					for d in xrange(D-1):
						vhat += link(uv - B[d])
					erp += abs(v-vhat)
					Yh.append((i,j,vhat))
					Yp.append((i,j,uv))
            
		if (itr)%debug_every == 0:
			logging.info("%f %f %f %s", err, norm(U), norm(V), str(B))
			if iter_f is not None:
				iter_f(itr,U,V,B,errs)
	return U,V,B,errs,Cs

def learn_iter(Y, n, m, k, link, g_link, U, V, B, D, init_step=0.01, beta=0.0000001, 
	mini_batch_size=-1,update_U = True, update_V = True, update_B = True,iteration=0 ):

	T=1
	import math
	if type(update_B) is bool:
		up_B = [update_B for d in xrange(D-1)]
		update_B = up_B
	step=init_step+0.0
	err = 0.0
	if mini_batch_size > 0: random.shuffle(Y)
	batches = []
	if(mini_batch_size <= 0):
		batches = [Y]
	else:
		batches = chunks(Y, mini_batch_size)
	a0=0
	a1=0
	a2=0
	a3=0
	a4=0
	a5=0
	for batch in batches:
		gU = np.zeros((n,k))
		gV = np.zeros((m,k))
		gB = np.zeros(D-1)
		for c in batch:
			(i,j,v) = c 
			v=int(v)
			if v==D-1:
				u=0    
				vnew_up=B[D-2]+T## we are tring to find the upper bound of interval base on v / posetive infinity
			else:
				u=1
				vnew_up=B[v]
			if v==0:
				d=0
				vnew_down=B[0]-T## Lower bounds/ negetive infinity
			else:
				d=1
				vnew_down=B[v-1]
			ui = U[i]
			vj = V[j]
			uv = np.dot(ui, vj)
			vhat=0
			for d in xrange(D-1):
				vhat += link(uv - B[d])
			if vhat==0:
				a0=a0+1
			elif vhat==1:
				a1=a1+1
			elif vhat==2:
				a2=a2+1
			elif vhat==3:
				a3=a3+1
			elif vhat==4:
				a4=a4+1
			else:
				a5=a5+1 

			err += pow(v-vhat,2)
			# gradient wrt u New version
			if update_U:
			    gU[i] += -ghinge_inverse(vnew_down-uv)*vj+ghinge_inverse(uv-vnew_up)*vj+ 2.0*beta*ui

			# gradient wrt V  New version
			if update_V:
				gV[j] += -ghinge_inverse(vnew_down-uv)*ui+ghinge_inverse(uv-vnew_up)*ui+ 2.0*beta*vj

#			# gradient wrt B[d]
			if iteration%1==0:
				if update_B:
					if v==D-1:
						gB[D-2]+=ghinge_inverse(vnew_down-uv)
					elif v==0:
						gB[0]+=-ghinge_inverse(uv-vnew_up)
					else:
						gB[v]+=-ghinge_inverse(uv-vnew_up)
						gB[v-1]+=ghinge_inverse(vnew_down-uv)

		# update
		batch_size = float(len(batch))
		if update_U: U = U - (step*gU)/100#/(batch_size)
		if update_V: V = V - step*gV/100#/(batch_size)
		for d in xrange(D-1):
			if update_B[d]:
				B[d] = B[d] - tep*gB[d]/1000#/(batch_size)
		B=np.sort(B)
	print a0,a1,a2,a3,a4,a5
	C=objective(Y,U,V,B,link,beta)
	return U,V,B,math.sqrt(err/len(Y)),C



def rnd_init(n,m,k,D,Y,link,g_link,init_step, beta,mini_batch_size,iters,tau= True):
	U = (np.random.rand(n,k)-0.5)/100.0
	V = (np.random.rand(m,k)-0.5)/100.0
	if tau:
		B = np.array([float(d) for d in xrange(D-1)])
	else:
		B = np.array([0,1,2,3,4])
	return U,V,B

def incr_init(n,m,k,D,Y,link,g_link,init_step,beta,mini_batch_size,iters):
	U,V,B=rnd_init(n,m,k,D,Y,link,g_link,init_step, beta,mini_batch_size,iters)
	for d in xrange(1,D):
		update_B = [False for x in xrange(D-1)]
		update_B[d-1] = True
		logging.info(update_B)
		Yd = map(lambda (i,j,v): (i,j,min(v,float(d))), Y)
		logging.info("Starting(%d): %d", d, len(Yd))
		for itr in xrange(iters):
			U,V,B,err=learn_iter(Yd, n, m, k, link, g_link, U, V, B, d+1, init_step, beta, mini_batch_size,
				True, True, update_B)
			if itr%(1)==0: logging.info("    %d: %f %f %f %s", itr, err, norm(U), norm(V), str(B))
		logging.info("Done(%d): %f %f %f %s", d, err, norm(U), norm(V), str(B))
	return U,V,B

