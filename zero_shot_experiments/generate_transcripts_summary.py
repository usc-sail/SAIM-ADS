import os 
import pandas as pd 
import numpy as np 
import openai 
import argparse 
import json
import time
import sys
from tqdm import tqdm

key_file=""
with open(key_file,"r") as f:
    key=f.read().strip().split("\n")

openai.api_key = key

