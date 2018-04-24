from data import ensure_dir

def read_file(fname):
        import string
        import gzip
        res=[]
        f=open(fname,'rb')
        n=0
        m=0
        D=0
        for l in f:
                cols=string.split(l)
                i=int(cols[0])-1
                j=int(cols[1])-1
                v=float(cols[2])
                res.append((i,j,v))
                if i>n: n=i
                if j>m: m=j
                if v>D: D=v
        f.close()
        return res,n+1,m+1,int(D+1)

def train(dataset, k=20, link_func = "Round"):
	import learn
	import learn_round
	import logging
	from time import localtime, strftime
	from os.path import expanduser

	fold = "0"
	mini_batch_size=1000
	init_step=1
	beta=0.01
	data_path = "data/"+ dataset + "/"
	debug_parent =  "data/" + dataset + "/" + fold + "/" + link_func + "/" 
	print debug_parent

	timef=strftime("%Y-%m-%d-%H-%M-%S", localtime())
	debug_dir=debug_parent + timef + "/" + str(k) + "/"
	ensure_dir(debug_dir)
	logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging.INFO) 
		
	import json
	param_f=open(debug_dir + "params.json", 'w')
	params = { "fold": fold, 'k': k, 'name': timef,
		'mini_batch_size': mini_batch_size, 'beta': beta,
		'init_step': init_step, 'link_func': link_func,
		'dataset': dataset }
	json.dump(params, param_f, indent=2)
	print json.dumps(params, param_f, indent=2)
	logging.info(json.dumps(params, param_f, indent=2))
	param_f.close()

	import cPickle as pickle
	import gzip
	U_f=gzip.open(debug_dir + "U." + str(iter) + ".pkl", 'wb')
	pickle.dump(beta, U_f)
	U_f.close()
	
	if link_func == "Round":
		link=learn_round.logistic
		g_link=learn.g_logistic
	else:
		(link, g_link) = learn.get_link(link_func)

	train_file = "data/"+  'u1.base' 
	test_file = "data/"+  'u1.test' 

	train,n1,m1,D1 = read_file(train_file)
	test,n2,m2,D2 = read_file(test_file)
	n = max(n1,n2)
	m = max(m1,m2)
	D = max(D1,D2)
	if link_func=="Linear": D = 2

	print n, m, D
	print "Train:", len(train)
	print "Test :", len(test)

	def iter_f(iter,U,V,B,errs):
		import cPickle as pickle
		import gzip
		U_f=gzip.open(debug_dir + "U." + str(iter) + ".pkl", 'wb')
		pickle.dump(U, U_f)
		U_f.close()
		V_f=gzip.open(debug_dir + "V." + str(iter) + ".pkl", 'wb')
		pickle.dump(V, V_f)
		V_f.close()
		B_f=gzip.open(debug_dir + "B." + str(iter) + ".pkl", 'wb')
		pickle.dump(B, B_f)
		B_f.close()
		errs_f=open(debug_dir + "errs.pkl", 'w')
		pickle.dump(errs, errs_f)
		errs_f.close()

	if link_func == "Round":
		U,V,B,errs,_ = learn_round.learn(train, n, m, k, link, g_link, D=D, beta=beta,
			init_step=init_step,mini_batch_size=mini_batch_size, iter_f=iter_f,iters=40)
	elif link_func == "Multi-sigmoid":
		U,V,B,errs = learn.learn(train, n, m, k, link, g_link, D=D, beta=beta,
			init_step=init_step,mini_batch_size=mini_batch_size, iter_f=iter_f,iters=40)
	else:
		U,V,B,errs = learn.learn(train, n, m, k, link, g_link, D=D, beta=beta,
			init_step=init_step,mini_batch_size=mini_batch_size, iter_f=iter_f, update_B = False,iters=40)
	return

if __name__ == "__main__":
	train("movielens", k = 20, link_func = "Linear")# "Linear" or "Multi-sigmoid" or "Round"

