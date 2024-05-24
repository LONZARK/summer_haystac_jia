import os 
import sys 
import glob 

import lxml 

def create_sumocfg(output_dir : str, 
                   default_cfg : str,
                   net_path : str, 
                   seed : int):
    """Create a general sumo config given the path to the net file and a seed.

    Parameters
    ----------
    output_dir : str
        directory location to save the sumocfg
    default_cfg : str
        location of the default sumocfg to copy other settings from; this file is only read and not modified
    net_path : str
        path of the net file to be referenced in the sumocfg
    seed : int
        randomization seed for reproducibility

    Returns
    -------
    path-like string
        full path of the resulting sumocfg
    """

    tree = lxml.etree.parse(default_cfg)

    for elem in tree.iter():
        
        if elem.tag == "net-file":
            elem.attrib['value'] = net_path
        
        if elem.tag == "seed":
            elem.attrib['value'] = str(seed)
    
    output_path = os.path.join(output_dir, "simulation.sumocfg")
    tree.write(output_path, pretty_print=True)

    return output_path