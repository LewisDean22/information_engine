# -*- coding: utf-8 -*-
"""

@author: Lewis Dean
"""
import os
  
DIRECTORIES = ("data_files", "plots")

def main(directories=DIRECTORIES):
  
    parent_directory = os.getcwd()
      
    for directory in directories:
        path = os.path.join(parent_directory, directory)
        os.mkdir(path)
        
    return None


if __name__ == '__main__':
    main()
