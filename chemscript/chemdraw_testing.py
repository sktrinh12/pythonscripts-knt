# -------------------------------------------------------------------------
# This script takes the input structure opens reactJS app that calls all
# Pipeline Pilot webservice API to render a table of chemical properties
# -------------------------------------------------------------------------

# ChemDraw/ChemScript interface Header
import sys
from ChemScript20 import *
import webbrowser
from os.path import expanduser, join, dirname
import logging

# home = expanduser("~")
input_filepath = sys.argv[1]
output_filepath = sys.argv[2]
logging.basicConfig(
    filename=join(dirname(output_filepath), "chemdraw_plp_ws_log.txt"),
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.debug("input: {}".format(input_filepath))
logging.debug("output: {}".format(output_filepath))
# available_endpoints = ["STANDALONE_CALC_PROPERTIES_TABLE", "STANDALONE_QSAR", "BOTH"]


def open_browser_link(smiles):
    url = "http://plp-calc-props.kinnate/?smiles={}".format(smiles)
    logging.debug("url: {}".format(url))
    # chrome.open_new_tab(url)
    webbrowser.open_new_tab("https://jsonplaceholder.typicode.com/posts/1/comments")


m1 = StructureData.LoadFile(input_filepath, "cdxml")
smiles = m1.WriteData("smiles").decode("utf-8")
logging.info("smiles: {}".format(smiles))
open_browser_link(smiles)
# with open(join(home, "output"), "w") as f:
#     f.write("{}|{}|{}\n\r".format(input_filepath, output_filepath, smiles))
#     for a in m1.Atoms:
#         f.write(
#             "{}|{}|{}|{}".format(
#                 a.Name, a.GetCartesian().x, a.GetCartesian().y, a.GetCartesian().z
#             )
#         )
#     open_browser_link(smiles, available_endpoints[0], f)
