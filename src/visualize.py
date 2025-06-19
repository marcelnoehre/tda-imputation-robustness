from tabulate import tabulate
from IPython.display import display, Markdown

def table(data, headers):
    display(Markdown(tabulate(data, headers=headers, tablefmt="github")))