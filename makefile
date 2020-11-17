cheby:
	python3 gibbs_cheby.py > output.txt
	more output.txt
cg:
	rm profile.txt
	kernprof -l conj_grad.py	
	python3 -m line_profiler *.lprof >> profile.txt
	more profile.txt
eigs:
	touch dim.txt
	octave makeA.m
	python3 -i test_eigenvalues.py
	rm dim.txt
cov:
	touch dim.txt
	octave makeA.m
	python3 -i test_covariance.py
	rm dim.txt
