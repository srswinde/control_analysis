import click
from tornado import template
from pathlib import Path
@click.command()
@click.option('-p', '--path', default='../notebooks', type=str, help="Path to notebook files")
@click.option('-a', '--address', type=str, default=None, help="IP Address of oscilloscope")
@click.option('-n', '--how-many', type=int, default=10, help="Number of frequency response data points")
@click.argument("name", nargs=1, required=True)
def main(path, address, how_many, name):
    """
    Create a bode analysis Jupyter Notebook from
    the bode_template notebook.

    Usage: do_bode_nb.py [OPTIONS] NAME

    Create a bode analysis Jupyter Notebook from the bode_template notebook.

    Options:
        -p, --path TEXT         Path to notebook files
        -a, --address TEXT      IP Address of oscilloscope
        -n, --how-many INTEGER  Number of frequency response data points
        --help                  Show this message and exit.


    """

    loader = template.Loader(Path(path))
    tmp = loader.load('bode_template.ipynb')
    click.echo(tmp.generate(how_many=how_many, comm=address))
    with open(Path(path)/f'{name}.ipynb', 'w') as nb:
        nb.write(tmp.generate(how_many=how_many, comm=address).decode())

    return


main()

