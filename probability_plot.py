def probabiluty(model,X,y):

	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt
	import numpy as np
	import scipy as sp

	# графік не повинен мати структури
	predictions = model.predict(X)
	test_res = y - predictions
	sns.scatterplot(x=y,y=test_res)
	plt.axhline(y=0, color='r', linestyle='--')

	# графік вірогідності
	fig, ax = plt.subplots(figsize=(6,8),dpi=100)
	_ = sp.stats.probplot(test_res,plot=ax)

	plt.show()