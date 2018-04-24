import movie_lens
import learn, learn_round
import matplotlib.pyplot as plt

def read_params(dir,iters):
        import gzip
        import cPickle as pickle
        Us = []
        Vs = []
        Bs = []
        for itr in xrange(iters):
                Uf=gzip.open(dir + "U." + str(itr) + ".pkl")
                U=pickle.load(Uf)
                Us.append(U)
                Uf.close()
                Vf=gzip.open(dir + "V." + str(itr) + ".pkl")
                V=pickle.load(Vf)
                Vs.append(V)
                Vf.close()
                Bf=gzip.open(dir + "B." + str(itr) + ".pkl")
                B=pickle.load(Bf)
                Bs.append(B)
                Bf.close()
        return Us,Vs,Bs

def plot_curve(dir,iters,Y,link_func):
        Us,Vs,Bs = read_params(dir,iters+1)
        if link_func == "Round":
                link=learn_round.logistic
        else:
                (link, _) = learn.get_link(link_func)
        errs=[]
        cells=float(len(Y))
        for itr in xrange(iters+1):
                Yh,err,percent = learn_round.pred(Y, Us[itr], Vs[itr], Bs[itr], link)
                errs.append(err)
                print "error:",err,"percent:",percent
        if link_func == "Multi-sigmoid":
                plt.plot(errs,label="Multi Sigmoid")
        elif link_func == "Linear":
                plt.plot(errs,label="Linear")
        else:
                plt.plot(errs,label="Round")

if __name__ == "__main__":
	link_func = "Linear"#"Linear" or "Multi-sigmoid" or "Round"
	fold = "0"
	dataset="movielens"
	data_path = "data/"+ dataset + "/"
	debug_path =  "data/"+ dataset + "/"+ fold + "/" 
	
	lin_dir = debug_path + "Linear" + "/2018-04-23-18-07-47/20/"
	Mul_dir = debug_path + "Multi-sigmoid" + "/2018-04-23-17-46-52/20/"
	Rou_dir = debug_path + "Round" + "/2018-04-23-17-34-55/20/"
	test_file = "data/"+  'u1.test' 
	test,_,_,_ = movie_lens.read_file(test_file)

	if link_func == "Linear":
		print "Linear"
		plot_curve(lin_dir, 39, test, "Linear")
	if link_func == "Multi-sigmoid":
		print "Multi-sigmoid"
		plot_curve(Mul_dir, 39, test, "Multi-sigmoid")
	if link_func == "Round":
		print "Round"
		plot_curve(Rou_dir, 39, test, "Round")
	plt.ylim([0,3])
	plt.legend()
	plt.savefig("movielens_test.pdf")
	plt.show()
