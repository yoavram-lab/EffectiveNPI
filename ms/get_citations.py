import click

@click.command()
@click.argument('aux_filename', default='ms.aux', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('output_filename', default='citation_keys', type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.option('-v/-V', '--verbose/--no-verbose', default=False)
def main(aux_filename, output_filename, verbose):
	with open(aux_filename) as f:
		lines = f.readlines()	

	lines = (line.strip() for line in lines if line.startswith(r'\citation'))
	lines = (line[len('\citation{'):-1] for line in lines)
	citations = set()
	for line in lines:
		for c in line.split(','):
			citations.add(c)	
	if verbose: 
		print("Found {} citations in {}".format(len(citations), aux_filename))
	with open(output_filename, 'wt') as f:
		f.write('\n'.join(sorted(citations)))


if __name__ == '__main__':
	main()