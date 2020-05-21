cheby:
	python3 gibbs_cheby.py > output.txt
	more output.txt
cg:
	rm profile.txt
	kernprof -l conj_grad.py	
	python3 -m line_profiler *.lprof >> profile.txt
	more profile.txt
