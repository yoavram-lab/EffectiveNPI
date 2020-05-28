for country in Austria Belgium Denmark France Germany Italy Norway Spain Sweden Switzerland United_Kingdom
do
	python Fig_tau_posterior.py 2020-05-27-more1week $country -q 
	python Fig_tau_posterior.py 2020-05-26-more2weeks $country -q
	python Fig_tau_posterior.py 2020-05-25-normal-endapril-1M $country -q
done



