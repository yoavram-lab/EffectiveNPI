for country in Austria Belgium Denmark France Germany Italy Norway Spain Sweden Switzerland United_Kingdom
# for country in Spain
do
	# python Fig_ppc.py 2020-06-23-28Mar $country red
	# python Fig_tau_posterior.py 2020-06-23-28Mar $country  -q

	# python Fig_ppc.py 2020-06-25-11Mar-fixedtau $country blue
	python Fig_ppc.py 2020-06-27-11Mar-notau $country green

	# python Fig_tau_posterior.py 2020-06-25-11Mar-fixedtau $country  -q
	python Fig_tau_posterior.py 2020-06-27-11Mar-notau $country  -q

	# python ppc.py 2020-05-14-n1-notau-1M $country green >> no_tau_loss.txt
	# python ppc.py 2020-05-15-n1-fixed-tau-1M $country red >> fixed_tau_loss.txt
done



