import click

@click.group()
def cli():
    pass

@cli.command()
def hello():
    print("hello eemeter world")