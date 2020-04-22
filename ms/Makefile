name=ms
bib_library=~/Documents/library.bib

$(name).bib: $(name).aux
	python get_citations.py $< citation_keys
	python get_bibtex.py citation_keys $(bib_library) $@

$(name).pdf: $(name).tex
	pdflatex $<
	bibtex $(name)
	pdflatex $<
	pdflatex $<
	@open $@

diff: 
	latexdiff $(name).tex $(name)2.tex > diff.tex
	@pdflatex diff.tex
	@bibtex $(name)
	@pdflatex diff.tex
	pdflatex diff.tex
	@open diff.pdf

list:
	@sh -c "$(MAKE) -p no_targets__ | awk -F':' '/^[a-zA-Z0-9][^\$$#\/\\t=]*:([^=]|$$)/ {split(\$$1,A,/ /);for(i in A)print A[i]}' | grep -v '__\$$' | grep -v 'make\[1\]' | grep -v 'Makefile' | sort"
no_targets__:

clean:
	@rm $(name).{aux,blg,log,out,pdf,bbl}