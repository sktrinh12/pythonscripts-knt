import webbrowser
import argparse
import re

description = """
chemscript script to invoke chrome web-browser to open a reactjs app that
renders chemical properties based off of PLP webservice
"""
available_endpoints = ["STANDALONE_CALC_PROPERTIES_TABLE", "STANDALONE_QSAR"]


def check_carbons(arg_string):
    if not re.search("[C|c]", arg_string):
        raise argparse.ArgumentTypeError(
            f"""The smiles string does not contain
                                         carbon, {arg_string}"""
        )
    elif arg_string.isdigit():
        raise argparse.ArgumentTypeError(
            f"""The smiles string is invalid, {arg_string}"""
        )
    else:
        return arg_string


parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-e",
    "--endpoint",
    type=str,
    default=available_endpoints[0],
    choices=available_endpoints,
    metavar="PLP webservice",
    required=True,
    help="Enter an endpoint for the api",
)
parser.add_argument(
    "-s",
    "--smiles",
    type=check_carbons,
    required=True,
    metavar="SMILES",
    help="Enter SMILES string",
)


def open_browser_link(smiles, endpoint):
    url = f"http://plp-calc-props.kinnate/?smiles={smiles}&endpoint={endpoint}"
    print(url)
    # chrome.open_new_tab(url)
    webbrowser.open_new_tab("https://jsonplaceholder.typicode.com/posts/1/comments")


args = parser.parse_args()

if __name__ == "__main__":
    open_browser_link(args.smiles, args.endpoint)
