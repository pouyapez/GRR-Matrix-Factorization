import numpy as np
import random
import sys
from numpy.linalg import norm
import logging
import math

def identity(x): return x

def g_identity(x): return 1.0

def logistic(x):
	from scipy.stats import logistic
	return 1.0 / (1.0 + math.exp(-x)) #logistic.cdf(x) 

def g_logistic(x): return 1.0/((np.exp(x/2)+np.exp(-x/2))*(np.exp(x/2)+np.exp(-x/2)))

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

def pred(Y,U,V,B,link):
	D=len(B)+1
	Yh = []
	err = 0.0
	for (i,j,v) in Y:
		ui = U[i]
		vj = V[j]
		uv = np.dot(ui, vj)
		vhat = 0.0
		for d in xrange(D-1):
			vhat += link(uv - B[d])
		Yh.append((i,j,vhat))
		err += abs(v-vhat) 
	return Yh,err/len(Y)

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



def learn(Y, n, m, k, link, g_link, iters=5000, D=2, init_step=0.1,update_B = True,tau=True,
	beta=0.002, mini_batch_size=-1, iter_f=None, debug_every=1):
	"""Returns learnt parameters of matrix factorization.

	Y - array of matrix cells: (i,j,v)
	n - number of rows of Y
	m - number of columns of Y
	k - rank of the factorization
	link - link function to use
	g_link - gradient of the link function
	iters - number of iterations
	"""
	if tau:
    	U,V,B=rnd_init(n,m,k,D,Y,link,g_link,init_step,beta,mini_batch_size,iters/1000)
	else:
        U,V,B=rnd_init(n,m,k,D,Y,link,g_link,init_step,beta,mini_batch_size,iters/1000,tau=False)

	errs = []
	itr=10
	logging.info("Starting training")
	p=0
	for itr in xrange(iters):
		p=p+1
		if (itr)%debug_every == 0: logging.info("Starting iteration %d of %d", (itr+1), iters)
            if update_B :
                U,V,B,err = learn_iter(Y, n, m, k, link, g_link, U, V, B, D, init_step, beta, mini_batch_size,iteration=p)
            else:
                U,V,B,err = learn_iter(Y, n, m, k, link, g_link, U, V, B, D, init_step, beta, mini_batch_size,update_B = False,iteration=p)
		errs.append(err)
		if (itr)%debug_every == 0:
			logging.info("%f %f %f %s", err, norm(U), norm(V), str(B))
			if iter_f is not None:
				iter_f(itr,U,V,B,errs)
	return U,V,B,errs

def learn_iter(Y, n, m, k, link, g_link, U, V, B, D, init_step=0.1, beta=0.002, 
	mini_batch_size=-1,update_U = True, update_V = True, update_B = True,iteration=0):
        import math
	if type(update_B) is bool:
		up_B = [update_B for d in xrange(D-1)]
		update_B = up_B
	step=init_step #/(itr+1)
	step=init_step+0.0
	err = 0.0
	# shuffle instances, only if doing mini-batches
	if mini_batch_size > 0: random.shuffle(Y)
	# figure out the batches
	batches = []
	if(mini_batch_size <= 0):
		batches = [Y]
	else:
		batches = chunks(Y, mini_batch_size)
	for batch in batches:
		gU = np.zeros((n,k))
		gV = np.zeros((m,k))
		gB = np.zeros(D-1)
	 	for c in batch:
			(i,j,v) = c
			ui = U[i]
			vj = V[j]
			uv = np.dot(ui, vj)
			vhat = 0.0
			for d in xrange(D-1):
				vhat += link(uv - B[d])
			#err += abs(v-vhat)
			err += pow(v-vhat,2)

			glink_sum = 0.0
			for d in xrange(D-1):
				glink_sum += g_link(uv - B[d])

			# gradient wrt u
			if update_U:
				g_err_u = glink_sum*vj
				gU[i] += -2.0*(v-vhat)*g_err_u + 2.0*beta*ui

			# gradient wrt V
			if update_V:
				g_err_v = glink_sum*ui
				gV[j] += -2.0*(v-vhat)*g_err_v + 2.0*beta*vj

			# gradient wrt B[d]
			for d in xrange(D-1):
				if update_B[d]:
					g_err_b = -g_link(uv - B[d])
					gB[d] += -2.0*(v-vhat)*g_err_b

		# update
		batch_size = float(len(batch))
		if update_U: U = U - step*gU/100#batch_size
		if update_V: V = V - step*gV/100#batch_size
		for d in xrange(D-1):
			if update_B[d]:
				B[d] = B[d] - step*gB[d]/1000#batch_size
	return U,V,B,math.sqrt(err/len(Y))

def rnd_init(n,m,k,D,Y,link,g_link,init_step, beta,mini_batch_size,iters,tau= True):
	U = (np.random.rand(n,k)-0.5)/100.0
	V = (np.random.rand(m,k)-0.5)/100.0
	if tau:
        B = np.array([float(d) for d in xrange(D-1)])
    else:
        B = np.array([0])
	return U,V,B

def incr_init(n,m,k,D,Y,link,g_link,init_step,beta,mini_batch_size,iters):
	U,V,B=rnd_init(n,m,k,D,Y,link,g_link,init_step, beta,mini_batch_size,iters)
	for d in xrange(1,D):
		update_B = [False for x in xrange(D-1)]
		update_B[d-1] = True
		#for x in xrange(d): update_B[x] = True
		logging.info(update_B)
		Yd = map(lambda (i,j,v): (i,j,min(v,float(d))), Y)
		logging.info("Starting(%d): %d", d, len(Yd))
		for itr in xrange(iters):
			U,V,B,err=learn_iter(Yd, n, m, k, link, g_link, U, V, B, d+1, init_step, beta, mini_batch_size,
				True, True, update_B)
			if itr%(1)==0: logging.info("    %d: %f %f %f %s", itr, err, norm(U), norm(V), str(B))
		logging.info("Done(%d): %f %f %f %s", d, err, norm(U), norm(V), str(B))
	return U,V,B
