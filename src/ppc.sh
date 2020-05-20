for country in Austria Belgium Denmark France Germany Italy Norway Spain Sweden Switzerland United_Kingdom
do
	python ppc.py 2020-05-14-n1-normal-1M $country blue >> free_tau_loss.txt
	python ppc.py 2020-05-14-n1-notau-1M $country green >> no_tau_loss.txt
	python ppc.py 2020-05-15-n1-fixed-tau-1M $country red >> fixed_tau_loss.txt
done



