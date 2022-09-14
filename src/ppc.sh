for country in Austria Belgium Denmark France Germany Italy Norway Spain Sweden Switzerland United_Kingdom
do
	python Fig_ppc.py 2020-06-25-11Mar-fixedtau $country blue
	python Fig_ppc.py 2020-06-27-Apr11-notau $country red
done



