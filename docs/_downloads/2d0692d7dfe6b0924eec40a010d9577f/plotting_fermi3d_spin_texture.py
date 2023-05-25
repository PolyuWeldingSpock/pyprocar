"""

.. _ref_plotting_fermi3d_spin_texture:

Plotting fermi3d spin_texture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting fermi3d spin_texture example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-colinear',
                                calc_type='fermi')
"""
# sphinx_gallery_thumbnail_number = 1

###############################################################################

import pyvista
# You do not need this. This is to ensure an image is rendered off screen when generating exmaple gallery.
pyvista.OFF_SCREEN = True

###############################################################################
# importing pyprocar and specifying local data_dir

import os
import pyprocar

project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = f"{project_dir}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-colinear{os.sep}fermi"


# First create the FermiHandler object, this loads the data into memory. Then you can call class methods to plot
fermiHandler = pyprocar.FermiHandler(
                                    code="vasp",
                                    dirname=data_dir,
                                    apply_symmetry=True)




###############################################################################
# Spin Texture mode
# +++++++++++++++++++++++++++++++++++++++
# 
#
fermiHandler.plot_fermi_surface(mode="spin_texture",
                              spin_texture=True,
                              arrow_size=0.1,
                              show=True,)


