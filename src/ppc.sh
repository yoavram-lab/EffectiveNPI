for country in Austria Belgium Denmark France Germany Italy Norway Spain Sweden Switzerland United_Kingdom
# for country in Spain
do
	python Fig_ppc.py 2020-05-25-normal-endapril-1M $country red
	# python Fig_tau_posterior.py 2020-05-26-more2weeks $country 
	# python Fig_ppc.py 2020-05-14-n1-normal-1M $country red >> free_tau_loss.txt
	# python ppc.py 2020-05-14-n1-notau-1M $country green >> no_tau_loss.txt
	# python ppc.py 2020-05-15-n1-fixed-tau-1M $country red >> fixed_tau_loss.txt
done



