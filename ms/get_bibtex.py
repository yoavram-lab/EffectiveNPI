import re

from bibtexparser import load as load_bib
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import click

species_name_pattern = re.compile(
	r"^(.*?)\{\\textless\}i\{\\textgreater\}(.*?)\{\\textless\}\/i\{\\textgreater\}(.*?)$")

@click.command()
@click.argument('keys_filename', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('bibtex_filename', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('output_filename', type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.option('-v/-V', '--verbose/--no-verbose', default=False)
def main(keys_filename, bibtex_filename, output_filename, verbose):
	with open(keys_filename) as f:
		citation_keys = [line.strip() for line in f.readlines()]
	if verbose:
		print("Read {} keys from {}".format(len(citation_keys), keys_filename))

	with open(bibtex_filename) as f:
		main_bib = load_bib(f)
	if verbose:
		print("Read {} entries from {}".format(len(main_bib.entries), bibtex_filename))

	out_bib = BibDatabase()
	for key in citation_keys:
		try:
			e = main_bib.entries_dict[key]
		except KeyError as err:
			print("** Failed to find '{}', check for typos **".format(key))
			raise err
		out_bib.entries.append(e)
	
	if verbose:
		print("Writing {} entries to {}".format(len(out_bib.entries), output_filename))

	for ent in out_bib.entries:
		m = species_name_pattern.match(ent['title'])
		if m:
			prefix, species, postfix = m.groups()
			ent['title'] = prefix + r"\emph{" + species + r"}" + postfix
		for key in ['url', 'file', 'abstract', 'mendeley-tags', 'keyword']:
			if key in ent:
				ent.pop(key)		

	writer = BibTexWriter()
	with open(output_filename, 'w') as f:
	    f.write(writer.write(out_bib))


if __name__ == '__main__':
	main()