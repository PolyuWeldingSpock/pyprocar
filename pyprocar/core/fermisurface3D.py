__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import random
import math
import sys
import copy
import itertools
from typing import List, Tuple

import numpy as np
import scipy.interpolate as interpolate
from matplotlib import colors as mpcolors
from matplotlib import cm

from . import Isosurface, Surface, BrillouinZone


np.set_printoptions(threshold=sys.maxsize)


HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg

class FermiSurface3D(Surface):
    """
    The object is used to store and manapulate a 3d fermi surface.

    Parameters
    ----------
    kpoints : (n,3) float
        A numpy array of kpoints used in the DFT calculation, this list
        has to be (n,3), n being number of kpoints and 3 being the
        3 different cartesian coordinates.
    band : (n,) float
        A numpy array of energies of ith band cooresponding to the
        kpoints.
    fermi : float
        Value of the fermi energy or any energy that one wants to
        find the isosurface with.
    reciprocal_lattice : (3,3) float
        Reciprocal lattice of the structure.
    spd :
        numpy array containing the information about projection of atoms,
        orbitals and spin on each band 
    spd_spin :
        numpy array containing the information about spin projection of atoms
    fermi_shift : float
        Value to shift fermi energy.
    fermi_tolerance : float = 0.1
        This is used to improve search effiency by doing a prior search selecting band within a tolerance of the fermi energy
    interpolation_factor : int
        The default is 1. number of kpoints in every direction
        will increase by this factor.
    colors : list str or list tuples of size 4, optional
        List of colors for each band. If you use tuple, it represents rgba values
        This argument does not work whena 3d file is saved. 
        The colors for when ``save3d`` is used, we
        recomend using qualitative colormaps, as this function will
        automatically choose colors from the colormaps. e.g. 
        
        .. code-block::
            :linenos: 

            colors=['red', 'blue', 'green']
            colors=[(1,0,0,1), (0,1,0,1), (0,0,1,1)]
            
    projection_accuracy : str, optional
        Controls the accuracy of the projects. 2 types ('high', normal) 
        The default is ``projection_accuracy=normal``.
    cmap : str
        The default is 'viridis'. Color map used in projecting the
        colors on the surface
    vmin :  float
        Value to normalize the minimum projection value. The default is 0.
    vmax :  float
        Value to normalize the maximum projection value.. The default is 1.
    supercell : list int
        This is used to add padding to the array 
        to assist in the calculation of the isosurface.
    """
    def __init__(self, ebs: np.ndarray, 
                 **kwargs):
        self.ebs = copy.deepcopy(ebs)
        self.kwargs = kwargs

        self._initialize_data()
        self._preprocess_data()
        full_isosurface, self.isosurfaces = self._generate_isosurfaces()

        super().__init__(verts=full_isosurface.points, faces=full_isosurface.faces)


    
    def _initialize_data(self):
        """
        Initialize the data for the Fermi surface calculation.
        """
        # Process input arguments and set default values
        self.bands_to_keep = self.kwargs.get('bands_to_keep', None)
        self.spin = self.kwargs.get('spin', 0)
        self.spd = self.kwargs.get('spd', None)
        self.spd_spin = self.kwargs.get('spd_spin', None)
        self.fermi_shift = self.kwargs.get('fermi_shift', 0.0)
        self.fermi_tolerance = self.kwargs.get('fermi_tolerance', 0.1)
        self.interpolation_factor = self.kwargs.get('interpolation_factor', 1)
        self.colors = self.kwargs.get('colors',None)
        self.surface_color = self.kwargs.get('surface_color', None)
        self.projection_accuracy = self.kwargs.get('projection_accuracy', "high")
        self.cmap = self.kwargs.get('cmap', "viridis")
        self.vmin = self.kwargs.get('vmin', 0)
        self.vmax = self.kwargs.get('vmax', 1)
        self.supercell = np.array(self.kwargs.get('supercell', [1, 1, 1]))

        # Initialize other class attributes
        self.full_isosurface = None
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)
        self.color_band_dict = {} 
        return None

    def _preprocess_data(self):
        """Preprocessing the ElectronicBandStructure. 
        """

        self._move_kpoints()
        self._keep_user_defined_bands()
        self._find_bands_near_fermi_level()
        
        return None

    def _move_kpoints(self):
        """Moves kpoints in between -0.5,0.5
        """
        bound_ops = -1.0*(self.ebs.kpoints > 0.5) + 1.0*(self.ebs.kpoints <= -0.5)
        self.ebs.kpoints = self.ebs.kpoints + bound_ops
        return None
    
    def _keep_user_defined_bands(self):
        """Reduce the number of bands as defined by the user with band_to_keep
        """
        self.ebs.bands = self.ebs.bands[:,:,self.spin]
        if self.bands_to_keep is None:
            self.bands_to_keep = len(self.ebs.bands[0,:])
        elif len(self.bands_to_keep) < len(self.ebs.bands[0,:]) :
            self.ebs.bands = self.ebs.bands[:,self.bands_to_keep]
        return None
    
    def _find_bands_near_fermi_level(self):
        """In the case of a fermi surfaces not all bands are important, 
        only using isosurface near the fermi level will speed the calculation up

        Raises
        ------
        Exception
            No bands within tolerance. Increase tolerance to increase search space.
        """
        # Find the bands within the Fermi level tolerance
        bands_within_tolerance = np.logical_and(
            self.ebs.bands >= self.ebs.efermi - self.fermi_tolerance,
            self.ebs.bands <= self.ebs.efermi + self.fermi_tolerance
        )
        full_band_indices = np.unique(np.where(bands_within_tolerance)[1])

        if not full_band_indices.size:
            raise Exception("No bands within tolerance. Increase tolerance to increase search space.")
        
        # Update bands with only those within the Fermi level tolerance
        self.ebs.bands = self.ebs.bands[:, full_band_indices]

        # Create mapping between full and reduced band indices
        reduced_band_indices = np.arange(full_band_indices.size)

        self._reduced_to_full_index_map = dict(zip(reduced_band_indices, full_band_indices))
        self._full_to_reduced_index_map = dict((reversed(item) for item in self._reduced_to_full_index_map .items()))
        # self.fullBandIndex_to_reducedBandIndex = dict(zip(full_band_indices, reduced_band_indices))
        # reduced_bands_to_keep_index = [iband for iband in range(len(self.bands_to_keep))]
        return None
    
    def _generate_isosurfaces(self):
        """
        Generate isosurfaces for each band and stores them in a list
        """
        isosurfaces = []

        # Loop through each band and generate isosurface
        for iband in range(self.ebs.bands.shape[1]):
            isosurface_band = Isosurface(
                XYZ=self.ebs.kpoints,
                V=self.ebs.bands[:, iband],
                isovalue=self.ebs.efermi,
                algorithm="lewiner",
                interpolation_factor=self.interpolation_factor,
                padding=self.supercell,
                transform_matrix=self.ebs.reciprocal_lattice,
                boundaries=self.brillouin_zone,
            )

            # Add to list of isosurfaces
            isosurfaces.append(isosurface_band)

        full_isosurface = self._combine_isosurfaces(isosurfaces)

        return full_isosurface,isosurfaces
    
    def _combine_isosurfaces(self,isosurfaces):
        "Combines the isosurface into a single object"
        # Initializing the full surface with first isosurface_band
        full_isosurface = copy.deepcopy(isosurfaces[0])
        for isosurface in isosurfaces[1:]:
            full_isosurface += copy.deepcopy(isosurface)
        return full_isosurface

    @property
    def n_bands(self):
        """
        The number of bands that compose the fermi surface
        """
        return len(self.isosurfaces)
    
    @property
    def reduced_to_full_index_map(self):
        """
        This will map the index of the isosurface, which represent bands in the fermi surface, 
        to the index in the ebs.bands
        """
        return self._reduced_to_full_index_map
    
    @property
    def full_to_reduced_index_map(self):
        """
        This will map  the index in the ebs.bands to the index of the isosurface, 
        which represent bands in the fermi surface, 
        """
        return self._full_to_reduced_index_map
    
    @property
    def n_points(self):
        """
        This is the total number of points that compose the fermi surface
        """
        return  len(self.points)

    def add_vector_texture(self,
                            vectors_array: np.ndarray, 
                            vectors_name: str="vector" ):
        """
        This method will map a list of vector to the 3d fermi surface mesh

        Parameters
        ----------
        vectors_array : np.ndarray
            The vector array corresponding to the kpoints
        vectors_name : str, optional
            The name of the vectors, by default "vector"
        """
        
        final_vectors_X = []
        final_vectors_Y = []
        final_vectors_Z = []
        for iband, isosurface in enumerate(self.isosurfaces):
            XYZ_extended = self.ebs.kpoints.copy()
            vectors_extended_X = vectors_array[:,iband,0].copy()
            vectors_extended_Y = vectors_array[:,iband,1].copy()
            vectors_extended_Z = vectors_array[:,iband,2].copy()

            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, vectors_array[:,iband,0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, vectors_array[:,iband,1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, vectors_array[:,iband,2], axis=0
                    )
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, vectors_array[:,iband,0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, vectors_array[:,iband,1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, vectors_array[:,iband,2], axis=0
                    )
    
            XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
    
            if self.projection_accuracy.lower()[0] == "n":
               
                vectors_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, isosurface.points, method="nearest"
                )
                vectors_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, isosurface.points, method="nearest"
                )
                vectors_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, isosurface.points, method="nearest"
                )
    
            elif self.projection_accuracy.lower()[0] == "h":
    
                vectors_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, isosurface.points, method="linear"
                )
                vectors_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, isosurface.points, method="linear"
                )
                vectors_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, isosurface.points, method="linear"
                )

            final_vectors_X.extend( vectors_X)
            final_vectors_Y.extend( vectors_Y)
            final_vectors_Z.extend( vectors_Z)
                
        self.set_vectors(final_vectors_X, final_vectors_Y, final_vectors_Z,vectors_name = vectors_name)
        return None

    def project_color(self, 
                    scalars_array:np.ndarray,
                    scalar_name:str="scalars"):
        """
        Projects the scalars to the 3d fermi surface.

        Parameters
        ----------
        scalars_array : np.array size[len(kpoints),len(self.bands)]   
            the length of the self.bands is the number of bands with a fermi iso surface
        scalar_name :str, optional
            The name of the scalars, by default "scalars"
        
        Returns
        -------
        None.
        """
        final_scalars = []
        for iband, isosurface in enumerate(self.isosurfaces):
            XYZ_extended = self.ebs.kpoints.copy()
            scalars_extended =  scalars_array[:,iband].copy()
    
    
            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
  
            XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
    
            if self.projection_accuracy.lower()[0] == "n":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, isosurface.centers, method="nearest"
                )
            elif self.projection_accuracy.lower()[0] == "h":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, isosurface.centers, method="linear"
                )
                
            final_scalars.extend(colors)

        self.set_scalars(final_scalars, scalar_name = scalar_name)
        return None
    
    
    
    
    # def _set_band_point_data(self):
    #     """
    #     Set the point data for the Fermi surface.
    #     """

    #     self.fermi_surface_area = self.area
    #     combined_band_color_array = []
    #     for iband, isosurface in enumerate(self.isosurfaces):

    #         # creat list to store colors
    #         new_color_array = []
    #         color_array_name = list(isosurface.point_data.keys())[0]
    #         band_color = isosurface.point_data[color_array_name]

    #         # Iterate over full surface points to apply band color in the correct order
    #         for point in self.points:
    #             # Check if point on full surface is on band isosurface
    #             if point in isosurface.points:
    #                 new_color_array.append(band_color)
    #             else:
    #                 new_color_array.append(np.array([0, 0, 0, 1]))
            
    #         # Iterate over band isosurface 
    #         for _ in range(len(isosurface.points)):
    #             combined_band_color_array.append(band_color)

    #         # i_reduced_band = int(color_array_name.split('_')[1])
    #         full_band_name = self.reduced_to_full_index_map[iband]
    #         print(new_color_array)
    #         self.point_data[f"band_{full_band_name}"] = np.array(new_color_array)
    #     self.point_data["bands"] = np.array(combined_band_color_array)
    #     return None

    # def _apply_colors_to_bands(self):

    #     nsurface = len(self.isosurfaces)
    #     if self.surface_color:
    #         solid_color_surface = np.arange(nsurface ) / nsurface

    #         if isinstance(self.surface_color,str):
    #             surface_color = mpcolors.to_rgba_array(self.surface_color, alpha =1 )[0,:]
    #         self.bands_color = np.array([surface_color for x in solid_color_surface[:]]).reshape(-1, 4)

    #     elif self.colors:
    #         self.bands_color =[]
    #         for color in self.colors:
    #             if isinstance(color,str):
    #                 color = mpcolors.to_rgba_array(color, alpha =1 )[0,:]
    #                 self.bands_color.append(color)
    #     else:
    #         norm = mpcolors.Normalize(vmin=self.vmin, vmax=self.vmax)
    #         cmap = cm.get_cmap(self.cmap)
    #         solid_color_surface = np.arange(nsurface ) / nsurface
    #         self.bands_color = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)



    #     for i_band,isosurface in enumerate(self.isosurfaces):
    #         n_points = len(isosurface.points[:,0])
    #         band_color = np.array([self.bands_color[i_band,:]]*n_points)
    #         # isosurface.point_data[f"band"] = band_color

    #         full_band_name = self.reduced_to_full_index_map[i_band]
    #         isosurface.point_data[f"band_{full_band_name}"] = band_color

    #     return self.bands_color
    
    # def create_vector_texture(self,
    #                         vectors_array: np.ndarray, 
    #                         vectors_name: str="vector" ):
    #     """
    #     This method will map a list of vector to the 3d fermi surface mesh

    #     Parameters
    #     ----------
    #     vectors_array : np.ndarray
    #         The vector array corresponding to the kpoints
    #     vectors_name : str, optional
    #         The name of the vectors, by default "vector"
    #     """
        
    #     final_vectors_X = []
    #     final_vectors_Y = []
    #     final_vectors_Z = []
    #     for iband, isosurface in enumerate(self.isosurfaces):
    #         XYZ_extended = self.XYZ.copy()
    #         vectors_extended_X = vectors_array[:,iband,0].copy()
    #         vectors_extended_Y = vectors_array[:,iband,1].copy()
    #         vectors_extended_Z = vectors_array[:,iband,2].copy()
    
    #         for ix in range(3):
    #             for iy in range(self.supercell[ix]):
    #                 temp = self.XYZ.copy()
    #                 temp[:, ix] += 1 * (iy + 1)
    #                 XYZ_extended = np.append(XYZ_extended, temp, axis=0)
    #                 vectors_extended_X = np.append(
    #                     vectors_extended_X, vectors_array[:,iband,0], axis=0
    #                 )
    #                 vectors_extended_Y = np.append(
    #                     vectors_extended_Y, vectors_array[:,iband,1], axis=0
    #                 )
    #                 vectors_extended_Z = np.append(
    #                     vectors_extended_Z, vectors_array[:,iband,2], axis=0
    #                 )
    #                 temp = self.XYZ.copy()
    #                 temp[:, ix] -= 1 * (iy + 1)
    #                 XYZ_extended = np.append(XYZ_extended, temp, axis=0)
    #                 vectors_extended_X = np.append(
    #                     vectors_extended_X, vectors_array[:,iband,0], axis=0
    #                 )
    #                 vectors_extended_Y = np.append(
    #                     vectors_extended_Y, vectors_array[:,iband,1], axis=0
    #                 )
    #                 vectors_extended_Z = np.append(
    #                     vectors_extended_Z, vectors_array[:,iband,2], axis=0
    #                 )
    
    #         XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
    
    #         if self.projection_accuracy.lower()[0] == "n":
               
    #             vectors_X = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_X, isosurface.points, method="nearest"
    #             )
    #             vectors_Y = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Y, isosurface.points, method="nearest"
    #             )
    #             vectors_Z = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Z, isosurface.points, method="nearest"
    #             )
    
    #         elif self.projection_accuracy.lower()[0] == "h":
    
    #             vectors_X = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_X, isosurface.points, method="linear"
    #             )
    #             vectors_Y = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Y, isosurface.points, method="linear"
    #             )
    #             vectors_Z = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Z, isosurface.points, method="linear"
    #             )

    #         final_vectors_X.extend( vectors_X)
    #         final_vectors_Y.extend( vectors_Y)
    #         final_vectors_Z.extend( vectors_Z)
                
    #     self.set_vectors(final_vectors_X, final_vectors_Y, final_vectors_Z,vectors_name = vectors_name)
    #     return None
            
    # def create_spin_texture(self):
    #     """
    #     This method will create the spin textures for the 3d fermi surface in the case of a non-colinear calculation
    #     """
    #     if self.spd_spin is not None:
    #         XYZ_extended = self.XYZ.copy()
    #         vectors_extended_X = self.spd_spin[0].copy()
    #         vectors_extended_Y = self.spd_spin[1].copy()
    #         vectors_extended_Z = self.spd_spin[2].copy()

    #         for ix in range(3):
    #             for iy in range(self.supercell[ix]):
    #                 temp = self.XYZ.copy()
    #                 temp[:, ix] += 1 * (iy + 1)
    #                 XYZ_extended = np.append(XYZ_extended, temp, axis=0)
    #                 vectors_extended_X = np.append(
    #                     vectors_extended_X, self.spd_spin[0], axis=0
    #                 )
    #                 vectors_extended_Y = np.append(
    #                     vectors_extended_Y, self.spd_spin[1], axis=0
    #                 )
    #                 vectors_extended_Z = np.append(
    #                     vectors_extended_Z, self.spd_spin[2], axis=0
    #                 )
    #                 temp = self.XYZ.copy()
    #                 temp[:, ix] -= 1 * (iy + 1)
    #                 XYZ_extended = np.append(XYZ_extended, temp, axis=0)
    #                 vectors_extended_X = np.append(
    #                     vectors_extended_X, self.spd_spin[0], axis=0
    #                 )
    #                 vectors_extended_Y = np.append(
    #                     vectors_extended_Y, self.spd_spin[1], axis=0
    #                 )
    #                 vectors_extended_Z = np.append(
    #                     vectors_extended_Z, self.spd_spin[2], axis=0
    #                 )



    #         XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
    #         if self.projection_accuracy.lower()[0] == "n":

    #             spin_X = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_X, self.points, method="nearest"
    #             )
    #             spin_Y = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Y, self.points, method="nearest"
    #             )
    #             spin_Z = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Z, self.points, method="nearest"
    #             )

    #         elif self.projection_accuracy.lower()[0] == "h":

    #             spin_X = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_X, self.points, method="linear"
    #             )
    #             spin_Y = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Y, self.points, method="linear"
    #             )
    #             spin_Z = interpolate.griddata(
    #                 XYZ_transformed, vectors_extended_Z, self.points, method="linear"
    #             )

    #         self.set_vectors(spin_X, spin_Y, spin_Z)
    #         return None
   
    # def project_color(self, 
    #                 scalars_array:np.ndarray,
    #                 scalar_name:str="scalars"):
    #     """
    #     Projects the scalars to the 3d fermi surface.

    #     Parameters
    #     ----------
    #     scalars_array : np.array size[len(kpoints),len(self.bands)]   
    #         the length of the self.bands is the number of bands with a fermi iso surface
    #     scalar_name :str, optional
    #         The name of the scalars, by default "scalars"
        
    #     Returns
    #     -------
    #     None.
    #     """
    #     final_scalars = []
    #     for iband, isosurface in enumerate(self.isosurfaces):
    #         XYZ_extended = self.XYZ.copy()
    #         scalars_extended =  scalars_array[:,iband].copy()
    
    
    #         for ix in range(3):
    #             for iy in range(self.supercell[ix]):
    #                 temp = self.XYZ.copy()
    #                 temp[:, ix] += 1 * (iy + 1)
    #                 XYZ_extended = np.append(XYZ_extended, temp, axis=0)
    #                 scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
    #                 temp = self.XYZ.copy()
    #                 temp[:, ix] -= 1 * (iy + 1)
    #                 XYZ_extended = np.append(XYZ_extended, temp, axis=0)
    #                 scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
  
    #         XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
    
    #         if self.projection_accuracy.lower()[0] == "n":
    #             colors = interpolate.griddata(
    #                 XYZ_transformed, scalars_extended, isosurface.centers, method="nearest"
    #             )
    #         elif self.projection_accuracy.lower()[0] == "h":
    #             colors = interpolate.griddata(
    #                 XYZ_transformed, scalars_extended, isosurface.centers, method="linear"
    #             )
                
    #         final_scalars.extend(colors)

    #     self.set_scalars(final_scalars, scalar_name = scalar_name)
    #     return None


    # def calculate_fermi_velocity(self):
    #     """
    #     Method to calculate the fermi velocity of the surface.
    #     """
    #     vectors_array=vectors_array
    #     self.create_vector_texture( vectors_array = vectors_array, vectors_name = "Fermi Velocity Vector"  )  

    # def calculate_fermi_speed(self):
    #     """
    #     Method to calculate the fermi speed of the surface.
    #     """
    #     scalars_array=None
    #     self.project_color(scalars_array = scalars_array, scalar_name = "Fermi Speed")

    # def calculate_effective_mass(self):
    #     """
    #     Method to calculate the effective mass of the surface.
    #     """
    #     scalars_array=None
    #     self.project_color(scalars_array = scalars_array, scalar_name="Geometric Average Effective Mass")

    # def project_atomic_projections(self):
    #     """
    #     Method to calculate the atomic projections of the surface.
    #     """

        
    #     self.spd = self.spd[:,self.fullBandIndex]

    #     scalars_array = []
    #     count = 0
    #     for iband in range(len(self.isosurfaces)):
    #         count+=1
    #         scalars_array.append(self.spd[:,iband])
    #     scalars_array = np.vstack(scalars_array).T
    #     self.project_color(scalars_array = scalars_array, scalar_name = "scalars")

    # def project_spin_texture_atomic_projections(self):
    #     """
    #     Method to calculate atomic spin texture projections of the surface.
    #     """
    #     if self.spd_spin[0] is not None:
    #         self.spd_spin = self.spd_spin[:,self.fullBandIndex,:]
    #     vectors_array = self.spd_spin

    #     self.create_vector_texture(vectors_array = vectors_array, vectors_name = "spin" )
   
    # def extend_surface(self,  extended_zone_directions: List[List[int] or Tuple[int,int,int]]=None,):
    #     """
    #     Method to extend the surface in the direction of a reciprocal lattice vecctor

    #     Parameters
    #     ----------
    #     extended_zone_directions : List[List[int] or Tuple[int,int,int]], optional
    #         List of directions to expand to, by default None
    #     """
        
    #     # The following code  creates exteneded surfaces in a given direction
    #     extended_surfaces = []
    #     if extended_zone_directions is not None:
    #         original_surface = copy.deepcopy(self) 
    #         for direction in extended_zone_directions:
    #             surface = copy.deepcopy(original_surface)
    #             self += surface.translate(np.dot(direction, self.ebs.reciprocal_lattice))
    #         #Clearing unneeded surface from memory
    #         del original_surface
    #         del surface

    def _get_brilloin_zone(self, 
                        supercell: List[int]):

        """Returns the BrillouinZone of the material

        Returns
        -------
        pyprocar.core.BrillouinZone
            The BrillouinZone of the material
        """

        return BrillouinZone(self.ebs.reciprocal_lattice, supercell)




# def __init__(
#         self,
#         kpoints: np.ndarray,
#         bands: np.ndarray,
#         fermi: float,
#         reciprocal_lattice: np.ndarray,

#         ebs=None,
#         bands_to_keep: List[int]=None,
#         spin: int= 0,
#         spd: np.ndarray=None,
#         spd_spin:np.ndarray=None,

#         fermi_shift: float=0.0,
#         fermi_tolerance:float=0.1,
#         interpolation_factor: int=1,
#         colors: List[str] or List[Tuple[float,float,float]]=None,
#         surface_color: str or Tuple[float,float,float, float]=None,
#         projection_accuracy: str="Normal",
#         cmap: str="viridis",
#         vmin: float=0,
#         vmax: float=1,
#         supercell: List[int]=[1, 1, 1],
#         # sym: bool=False
#         ):

#         self.ebs = ebs.deepcopy()


#         self.kpoints = kpoints
        
#         # Shifts kpoints between [0.5,0.5)
#         bound_ops = -1.0*(self.ebs.kpoints > 0.5) + 1.0*(self.ebs.kpoints <= -0.5)
#         self.XYZ = self.ebs.kpoints + bound_ops

#         self.reciprocal_lattice = reciprocal_lattice

#         self.supercell = np.array(supercell)
#         self.fermi = fermi + fermi_shift
#         self.interpolation_factor = interpolation_factor
#         self.projection_accuracy = projection_accuracy
#         self.spd = spd
#         self.spd_spin = spd_spin

#         self.brillouin_zone = self._get_brilloin_zone(self.supercell)

#         self.cmap = cmap
#         self.vmin = vmin
#         self.vmax = vmax
        
#         # Finding bands with a fermi iso-surface. This reduces searching
#         self.ebs.bands = self.ebs.bands[:,:,spin]
#         if bands_to_keep is None:
#             bands_to_keep = len(self.ebs.bands[0,:])
#         elif len(bands_to_keep) < len(self.bands[0,:]) :
#             self.ebs.bands = self.ebs.bands[:,bands_to_keep]

#         fullBandIndex = []
#         reducedBandIndex = []
#         for iband in range(len(self.ebs.bands[0,:])):
#             fermi_surface_test = len(np.where(np.logical_and(self.ebs.bands[:,iband]>=self.fermi-fermi_tolerance, self.ebs.bands[:,iband]<=self.fermi+fermi_tolerance))[0])
#             if fermi_surface_test != 0:
#                 fullBandIndex.append(iband)
#         if len(fullBandIndex)==0:
#             raise Exception("No bands within tolerance. Increase tolerance to increase search space.")
#         self.ebs.bands = self.ebs.bands[:,fullBandIndex]

#         # re-index and creates a mapping to the original bandindex
#         reducedBandIndex = np.arange(len(self.ebs.bands[0,:]))
#         self.fullBandIndex = fullBandIndex
#         self.reducedBandIndex = reducedBandIndex
#         self.reducedBandIndex_to_fullBandIndex = {f"{key}":value for key,value in zip(reducedBandIndex,fullBandIndex)}
#         self.fullBandIndex_to_reducedBandIndex = {f"{key}":value for key,value in zip(fullBandIndex,reducedBandIndex)}
#         reduced_bands_to_keep_index = [iband for iband in range(len(bands_to_keep))]

#         # Generate unique rgba values for the bands
#         nsurface = len(self.ebs.bands[0,:])
#         if surface_color:
#             solid_color_surface = np.arange(nsurface ) / nsurface

#             if isinstance(surface_color,str):
#                 surface_color = mpcolors.to_rgba_array(surface_color, alpha =1 )[0,:]
#             band_colors = np.array([surface_color for x in solid_color_surface[:]]).reshape(-1, 4)
#         elif colors:
#             band_colors =[]
#             for color in colors:
#                 if isinstance(color,str):
#                     color = mpcolors.to_rgba_array(color, alpha =1 )[0,:]
#                     band_colors.append(color)
#         else:
#             norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
#             cmap = cm.get_cmap(cmap)
#             solid_color_surface = np.arange(nsurface ) / nsurface
#             band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)

#         # The following loop generates iso surfaces for each band and then stores them in a list
#         color_band_dict = {}

#         self.isosurfaces = []
#         full_isosurface = None
#         iband_with_surface=0
#         for iband in  range(self.ebs.bands.shape[1]):
#             isosurface_band = Isosurface(
#                                 XYZ=self.XYZ,
#                                 V=self.ebs.bands[:,iband],
#                                 isovalue=self.fermi,
#                                 algorithm="lewiner",
#                                 interpolation_factor=interpolation_factor,
#                                 padding=self.supercell,
#                                 transform_matrix=self.reciprocal_lattice,
#                                 boundaries=self.brillouin_zone,
#                             )

#             # Following condition will handle initializing of fermi isosurface
#             isosurface_band_copy = copy.deepcopy(isosurface_band)
#             if full_isosurface is None and len(isosurface_band_copy.points)!=0:
#                 full_isosurface = isosurface_band_copy    
#             elif len(isosurface_band_copy.points)==0:
#                 if full_isosurface is None and iband == len(self.ebs.bands[0,:])-1:
#                     # print(full_isosurface)
#                     raise Exception("Could not find any fermi surfaces")
#                 continue
#             else:    
#                 full_isosurface += isosurface_band_copy

#             color_band_dict.update({f"band_{iband_with_surface}": {"color" : band_colors[iband_with_surface,:]} })
#             band_color = np.array([band_colors[iband_with_surface,:]]*len(isosurface_band.points[:,0]))
#             isosurface_band.point_data[f"band_{iband_with_surface}"] = band_color
#             self.isosurfaces.append(isosurface_band)
#             iband_with_surface +=1

#         # Initialize the Fermi Surface which is the combination of all the 
#         # isosurface for each band
#         super().__init__(verts=full_isosurface.points, faces=full_isosurface.faces)
#         self.fermi_surface_area = self.area
        
#         # Remapping of the scalar arrays into the combind mesh  
#         count = 0
#         combined_band_color_array = []
#         for iband,isosurface_band in enumerate(self.isosurfaces):
#             new_color_array = []
#             color_array_name = isosurface_band.point_data.keys()[0]
#             for points in self.points:
#                 if points in isosurface_band.points:
#                     new_color_array.append(color_band_dict[color_array_name]['color'])    
#                 else:
#                     new_color_array.append(np.array([0,0,0,1]))

#             for ipoints in range(len(isosurface_band.points)):
#                 combined_band_color_array.append(color_band_dict[color_array_name]['color'])
#             i_reduced_band = color_array_name.split('_')[1]

#             self.point_data[ "band_"+ str(self.reducedBandIndex_to_fullBandIndex[str(i_reduced_band)])] = np.array(new_color_array)
#         self.point_data["bands"] = np.array(combined_band_color_array ) 
#         return None