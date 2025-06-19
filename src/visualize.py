from tabulate import tabulate
from IPython.display import display, Markdown

def table(data, headers):
    """
    Display a table using Markdown.

    :param list[list] data: The data to be displayed in the table
    :param list[str] headers: The headers for the table
    """
    display(Markdown(tabulate(data, headers=headers, tablefmt="github")))