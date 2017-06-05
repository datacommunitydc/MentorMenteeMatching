import click
from matching import *

@click.command()
# @click.option('--count', default=1, help='Number of files.')
@click.option('--name', prompt='Enter Filename',
              help='The file of mentor and mentee responses.')
@click.option('--mentor', default=False, help='Shall we match Mentors and Mentees?')
def main(name,mentor):
    """Simple program that greets NAME for a total of COUNT times."""
    if mentor=="False":
      mentor = False
    else:
      mentor = True
    match(name,mentor)
    click.echo('Loading file: %s.' % name)
    click.echo('Matching with mentors: %s.' % mentor)

if __name__ == '__main__':
    main()