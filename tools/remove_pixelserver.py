#!/usr/bin/python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import sys
import glob


SIGNATURE = "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions"


def remove_pixelserver_from_notebook(file_path):
    """
    Remove pixelserver tracking from a notebook. If the pixcelserver signature found in
    the notebook, the pixelserver cell will be removed from the notebook file. File will
    be modified only when the pixelserver signature is found in it.

    Args:
        file_path (str): The notebook file path.
    """

    with open(file_path, encoding='utf-8') as fd:
        raw_json = json.load(fd)

        if 'cells' not in raw_json:
            return
        
        cells = raw_json['cells']
        pixel_cells = []

        for idx, cell in enumerate(cells):
            if cell['cell_type'] != 'markdown':
                continue
            
            source = cell['source']
            for row in source:
                if row.startswith(SIGNATURE):
                    pixel_cells.append(idx)
                    print("Found pixelserver in file: \"{}\", cell {}".format(file_path, idx))
        
        for cell_id in pixel_cells[::-1]:
            cells.pop(cell_id)

    if pixel_cells:
        with open(file_path, 'w', encoding='utf-8') as fd:
            json.dump(raw_json, fd, indent=1)


def get_all_notebook_files():
    """
    Get all example notebook files' path and return them as a list.

    Returns:
        list of str. A list of notebook file paths. 
    """

    root_path = os.path.dirname(sys.path[0])
    examples_path = os.path.join(root_path, "examples")
    if not os.path.exists(examples_path):
        raise ValueError("Cannot find examples file path: {}".format(examples_path))

    files = [f for f in glob.glob(os.path.join(examples_path, "*/*.ipynb"), recursive=True)]
    return files


def main():
    """
    Remove pixelserver from all example notebooks.
    """
    
    notebooks = get_all_notebook_files()
    for notebook in notebooks:
        remove_pixelserver_from_notebook(notebook)


if __name__ == "__main__":
    main()
