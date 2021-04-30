# -*- coding: utf-8 -*-

import numpy as np
import pyvista

from matplotlib import colors as mpcolors
from matplotlib import cm
from .core.surface import boolean_add
from .fermisurface3d import FermiSurface3D
from .splash import welcome
from .utilsprocar import UtilsProcar
from .procarparser import ProcarParser
from .procarselect import ProcarSelect
from .bxsfparser import BxsfParser
from .frmsfparser import FrmsfParser
from .qeparser import QEFermiParser
from .lobsterparser import LobsterFermiParser
from .abinitparser import AbinitParser

def fermi3D(
    procar="PROCAR",
    outcar="OUTCAR",
    infile="in.bxsf",
    abinit_output=None,
    fermi=None,
    bands=None,
    interpolation_factor=1,
    mode="plain",
    supercell=[1, 1, 1],
    extended_zone_directions = None,
    colors=None,
    background_color="white",
    save_colors=False,
    cmap="jet",
    atoms=None,
    orbitals=None,
    spin=None,
    spin_texture=False,
    arrow_color=None,
    arrow_size=0.015,
    only_spin=False,
    fermi_shift=0,
    projection_accuracy="normal",
    code="vasp",
    vmin=0,
    vmax=1,
    savegif=None,
    savemp4=None,
    save3d=None,
    save_meshio= False,
    perspective=True,
    save2d=False,
    show_curvature = False,
    show_slice = False,
    camera_pos=[1, 1, 1],
    widget=False,
    show=True,
    repair=True,
):
    """
    Parameters
    ----------
    procar : str, optional (default ``'PROCAR'``)
        Path to the PROCAR file of the simulation
        e.g. ``procar='~/MgB2/fermi/PROCAR'``
    outcar : str, optional (default ``'OUTCAR'``)
        Path to the OUTCAR file of the simulation
        e.g. ``outcar='~/MgB2/fermi/OUTCAR'``
    abinit_output : str, optional (default ``None``)
        Path to the Abinit output file
        e.g. ``outcar='~/MgB2/abinit.out'``
    infile : str, optional (default ``infile = in.bxsf'``)
        This is the path in the input bxsf file
        e.g. ``infile = ni_fs.bxsf'``
    fermi : float, optional (default ``None``)
        Fermi energy at which the fermi surface is created. In other
        words fermi is the isovalue at which the fermi surface is
        created. If not defined it is read from the OUTCAR file.
        e.g. ``fermi=-5.49``
    bands : list int, optional
        Which bands are going to be plotted in the fermi surface. The
        numbering is based on vasp outputs. If nothing is selected,
        this function will iterate over all the bands and plots the
        ones that cross fermi.
        e.g. ``bands=[14, 15, 16, 17]``
    interpolation_factor : int, optional
        The kpoints grid will increase by this factor and interpolated
        at the new points using Fourier interpolation.
        e.g. If the kgrid is 5x5x5, ``interpolation_factor=4`` will
        lead to a kgrid of 20x20x20
    mode : str, optional (default ``mode='plain'``)
        Defines If the fermi surface will have any projection using
        colormaps or is a plotted with a uniform plain color.
        e.g. ``mode='plain'``, ``mode='parametric'``
    supercell : list int, optional (default ``[1, 1, 1]``)
        If one wants plot more than the 1st brillouin zone, this
        parameter can be used.
        e.g. ``supercell=[2, 2, 2]``

    extended_zone_directions : list of list of size 3, optional (default ``None``)
        If one wants plot more than  brillouin zones in a particular direection, this
        parameter can be used.
        e.g. ``extended_zone_directions=[[1,0,0],[0,1,0],[0,0,1]]``
    colors : list str, optional
        List of colors for each band. This argument does not work when
        a 3d file is saved. The colors for when ``save3d`` is used, we
        recomend using qualitative colormaps, as this function will
        automatically choose colors from the colormaps.
        e.g. ``colors=['red', 'blue', 'green']``
    background_color : str, optional (default ``white``)
        Defines the background color.
        e.g. ``background_color='gray'``
    save_colors : bool, optional (default ``False``)
        In case the plot is saved in 3D and some property of the
        material is projected on the fermi surface, this argument
        allows the projection to be stored in the 3D file.
        e.g. ``save_colors=True``
    cmap : str, optional (default ``jet``)
        The color map used for color coding the projections. ``cmap``
        is only relevant in ``mode='parametric'``. A full list of
        color maps in matplotlib are provided in this web
        page. `https://matplotlib.org/2.0.1/users/colormaps.html
        <https://matplotlib.org/2.0.1/users/colormaps.html>`_
    atoms : list int, optional
        ``atoms`` define the projection of the atoms on the fermi
        surfcae . In other words it selects only the contribution of the
        atoms provided. Atoms has to be a python list(or numpy array)
        containing the atom indices. Atom indices has to be order of
        the input files of DFT package. ``atoms`` is only relevant in
        ``mode='parametric'``. keep in mind that python counting
        starts from zero.
        e.g. for SrVO\ :sub:`3`\  we are choosing only the oxygen
        atoms. ``atoms=[2, 3, 4]``, keep in mind that python counting
        starts from zero, for a **POSCAR** similar to following::
            Sr1 V1 O3
            1.0
            3.900891 0.000000 0.000000
            0.000000 3.900891 0.000000
            0.000000 0.000000 3.900891
            Sr V O
            1 1 3
            direct
            0.500000 0.500000 0.500000 Sr atom 0
            0.000000 0.000000 0.000000 V  atom 1
            0.000000 0.500000 0.000000 O  atom 2
            0.000000 0.000000 0.500000 O  atom 3
            0.500000 0.000000 0.000000 O  atom 4
        if nothing is specified this parameter will consider all the
        atoms present.
    orbitals : list int, optional
        ``orbitals`` define the projection of orbitals on the fermi
        surface. In other words it selects only the contribution of
        the orbitals provided. Orbitals has to be a python list(or
        numpy array) containing the Orbital indices. Orbitals indices
        has to be order of the input files of DFT package. The
        following table represents the indecies for different orbitals
        in **VASP**.
            +-----+-----+----+----+-----+-----+-----+-----+-------+
            |  s  | py  | pz | px | dxy | dyz | dz2 | dxz | x2-y2 |
            +-----+-----+----+----+-----+-----+-----+-----+-------+
            |  0  |  1  |  2 |  3 |  4  |  5  |  6  |  7  |   8   |
            +-----+-----+----+----+-----+-----+-----+-----+-------+
        ``orbitals`` is only relavent in ``mode='parametric'``
        e.g. ``orbitals=[1,2,3]`` will only select the p orbitals
        while ``orbitals=[4,5,6,7,8]`` will select the d orbitals.
        If nothing is specified pyprocar will select all the present
        orbitals.
    spin : list int, optional
        e.g. ``spin=[0]``
    spin_texture : bool, optional (default False)
        In non collinear calculation one can choose to plot the spin
        texture on the fermi surface.
        e.g. ``spin_texture=True``
    arrow_color : str, optional
        Defines the color of the arrows when
        ``spin_texture=True``. The default will select the colors
        based on the color map specified. If arrow_color is selected,
        all arrows will have the same color.
        e.g. ``arrow_color='red'``
    arrow_size : int, optional
        As the name suggests defines the size of the arrows, when spin
        texture is selected.
        e.g. ``arrow_size=3``
    only_spin : bool, optional
        If ``only_spin=True`` is selected, the fermi surface is not
        plotted and only the spins in the spin texture is plotted.
    fermi_shift : float, optional
        This parameter is useful when one wants to plot the iso-surface
        above or belove the fermi level.
        e.g. ``fermi_shift=0.6``
    projection_accuracy : str, optional (default ``'normal'``)
        Selected the accuracy of projected properties. ``'normal'`` and
        ``'high'`` are the only two options. ``'normal'`` uses the fast
        but rather inaccurate nearest neighbor interpolation, while
        ``'high'`` uses the more accurate linear interpolation for the
        projection of the properties.
        e.g. ``projection_accuracy='high'``
    code : str, optional (default ``'vasp'``)
        The DFT code in which the calculation is performed with.
        Also, if you want to read a .bxsf file set code ="bxsf"
        e.g. ``code='vasp'``
    vmin : float, optional
        The maximum value in the color bar. cmap is only relevant in
        ``mode='parametric'``.
        e.g. vmin=-1.0
    vmax : float, optional
        The maximum value in the color bar. cmap is only relevant in
        ``mode='parametric'``.
        e.g. vmax=1.0
    savegif : str, optional
        pyprocar can save the fermi surface in a gif
        format. ``savegif`` is the path to which the gif is saved.
        e.g. ``savegif='fermi.gif'`` or ``savegif='~/MgB2/fermi.gif'``
    savemp4 : str, optional
        pyprocar can save the fermi surface in a mp4 video format.
        ``savemp4`` is the path to which the video is saved.
        e.g. ``savegif='fermi.mp4'`` or ``savegif='~/MgB2/fermi.mp4'``
    save3d : str, optional
        pyprocar can save the fermi surface in a 3d file format.
        pyprocar uses the `trimesh <https://github.com/mikedh/trimesh>`_
        to save the 3d file. trimesh can export files with the
        following formats STL, binary PLY, ASCII OFF, OBJ, GLTF/GLB
        2.0, COLLADA. ``save3d`` is the path to which the file is saved.
        e.g. ``save3d='fermi.glb'``
    save_meshio : bool, optional
        pyprocar can use meshio to save any 3d format supported by it.
    perspective : bool, optional
        To create the illusion of depth, perspective is used in 2d
        graphics. One can turn this feature off by ``perspective=False``
        e.g.  ``perspective=False``
    save2d : str, optional
        The fermi surface can be saved as a 2D image. This parameter
        turns this feature on and selects the path at which the file
        is going to be saved.
        e.g. ``save2d='fermi.png'``
    show_slice : bool, optional
        Creates a widget which slices the fermi surface
    show_curvature : bool, optional
        plots the curvature of the fermi surface
    camera_pos : list float, optional (default ``[1, 1, 1]``)
        This parameter defines the position of the camera where it is
        looking at the fermi surface. This feature is important when
        one chooses to use the ``save2d``, ``savegif`` or ``savemp4``
        option.
        e.g. ``camera_pos=[0.5, 1, -1]``
    widget : , optional
        .. todo::
    show : bool, optional (default ``True``)
        If set to ``False`` it will not show the 3D plot.
    Returns
    -------
    s : pyprocar surface object
        The whole fermi surface added bands
    surfaces : list pyprocar surface objects
        list of fermi surface of each band
    """

    welcome()
    ##########################################################################
    # Code dependencies
    ##########################################################################
    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(procar, procar)
            print("PROCAR repaired. Run with repair=False next time.")

    if show:
        p = pyvista.Plotter()

    if code == "vasp":
        outcarparser = UtilsProcar()
        if fermi is None:
            e_fermi = outcarparser.FermiOutcar(outcar)
        else:
            e_fermi = fermi
        reciprocal_lattice = outcarparser.RecLatOutcar(outcar)
        procarFile = ProcarParser()
        procarFile.readFile(procar, False)
        data = ProcarSelect(procarFile, deepCopy=True)

    elif code == "abinit":
        procarFile = ProcarParser()
        procarFile.readFile(procar, False)
        abinitFile = AbinitParser(abinit_output=abinit_output)
        if fermi is None:
            e_fermi = abinitFile.fermi
        else:
            e_fermi = fermi
        reciprocal_lattice = abinitFile.reclat
        data = ProcarSelect(procarFile, deepCopy=True)

        # Converting Ha to eV
        data.bands = 27.211396641308 * data.bands

    elif code == "qe":
        procarFile = QEFermiParser()
        reciprocal_lattice = procarFile.reclat
        data = ProcarSelect(procarFile, deepCopy=True)
        if fermi is None:
            e_fermi = procarFile.fermi
        else:
            e_fermi = fermi

    elif code == "lobster":
        procarFile = LobsterFermiParser()
        reciprocal_lattice = procarFile.reclat
        data = ProcarSelect(procarFile, deepCopy=True)
        if fermi is None:
            e_fermi = 0
        else:
            e_fermi = fermi


    elif code == "bxsf":
        e_fermi = fermi
        data = BxsfParser(infile= infile)
        reciprocal_lattice = data.reclat

    elif code == "frmsf":
        e_fermi = fermi
        data = FrmsfParser(infile=infile)
        reciprocal_lattice = data.rec_lattice
        bands = np.arange(len(data.bands[0, :]))


    ##########################################################################
    # Data Formating
    ##########################################################################


    band_numbers = bands
    if band_numbers is None:
        band_numbers = np.arange(len(data.bands[0, :]))


    spd = []
    if mode == "parametric":
        if orbitals is None:
            orbitals = [-1]
        if atoms is None:
            atoms = [-1]
        if spin is None:
            spin = [0]
        data.selectIspin(spin)
        data.selectAtoms(atoms, fortran=False)
        data.selectOrbital(orbitals)

        for iband in band_numbers:
            spd.append(data.spd[:, iband])
    else:
        for iband in band_numbers:
            spd.append(None)

    spd_spin = []

    if spin_texture:
        dataX = ProcarSelect(procarFile, deepCopy=True)
        dataY = ProcarSelect(procarFile, deepCopy=True)
        dataZ = ProcarSelect(procarFile, deepCopy=True)

        dataX.kpoints = data.kpoints
        dataY.kpoints = data.kpoints
        dataZ.kpoints = data.kpoints

        dataX.spd = data.spd
        dataY.spd = data.spd
        dataZ.spd = data.spd

        dataX.bands = data.bands
        dataY.bands = data.bands
        dataZ.bands = data.bands

        dataX.selectIspin([1])
        dataY.selectIspin([2])
        dataZ.selectIspin([3])

        dataX.selectAtoms(atoms, fortran=False)
        dataY.selectAtoms(atoms, fortran=False)
        dataZ.selectAtoms(atoms, fortran=False)

        dataX.selectOrbital(orbitals)
        dataY.selectOrbital(orbitals)
        dataZ.selectOrbital(orbitals)
        for iband in band_numbers:
            spd_spin.append(
                [dataX.spd[:, iband], dataY.spd[:, iband], dataZ.spd[:, iband]]
            )
    else:
        for iband in band_numbers:
            spd_spin.append(None)


    ##########################################################################
    # Initialization of the Fermi Surface
    ##########################################################################

    fermi_surface3D = FermiSurface3D(
                                    kpoints=data.kpoints,
                                    bands=data.bands,
                                    band_numbers = band_numbers,
                                    spd=spd,
                                    spd_spin=spd_spin,
                                    fermi=e_fermi,
                                    fermi_shift = fermi_shift,
                                    reciprocal_lattice=reciprocal_lattice,
                                    interpolation_factor=interpolation_factor,
                                    projection_accuracy=projection_accuracy,
                                    supercell=supercell,
                                    cmap=cmap,
                                    vmin = vmin,
                                    vmax=vmax,
                                    extended_zone_directions = extended_zone_directions
                                )
    band_surfaces = fermi_surface3D.band_surfaces
    fermi_surface = fermi_surface3D.fermi_surface
    colors = fermi_surface3D.colors
    brillouin_zone = fermi_surface3D.brillouin_zone

    fermi_surface_area = fermi_surface3D.fermi_surface_area
    band_surfaces_area = fermi_surface3D.band_surfaces_area

    fermi_surface_curvature = fermi_surface3D.fermi_surface_curvature
    band_surfaces_curvature = fermi_surface3D.band_surfaces_curvature

    test = fermi_surface_curvature

    # coloring variables
    nsurface = len(band_surfaces)
    # # norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap)
    scalars = np.arange(nsurface + 1) / nsurface

    if save_colors is not False or save3d is not None:
        for i in range(nsurface):
            if band_surfaces[i].scalars is None:
                band_surfaces[i].set_scalars([scalars[i]] * band_surfaces[i].nfaces)



    fermi_surfaces = band_surfaces.copy()


    ##########################################################################
    # Plotting the surface
    ##########################################################################


    if show or save2d:
        # sargs = dict(interactive=True)


        p.add_mesh(
            brillouin_zone.pyvista_obj,
            style="wireframe",
            line_width=3.5,
            color="black",
        )

        if show_slice:
            if mode == "plain":
                text = "Plain"
                p.add_mesh_slice(fermi_surface, cmap=cmap, clim=[vmin, vmax])
                p.remove_scalar_bar()
            elif mode == "parametric":
                text = "Projection"
                p.add_mesh_slice(fermi_surface, cmap=cmap, clim=[vmin, vmax])
                p.remove_scalar_bar()
            else:
                text = "Spin Texture"
                for isurface in range(nsurface):
                    arrows = band_surfaces[isurface].glyph(
                        orient="vectors", factor=arrow_size)

                    if arrow_color is None:
                        p.add_mesh(arrows, cmap=cmap, clim=[vmin, vmax])
                        p.remove_scalar_bar()
                    else:
                        p.add_mesh(arrows, color=arrow_color)


            # p.add_mesh_slice(fermi_surface, cmap=cmap, clim=[vmin, vmax])
            # p.remove_scalar_bar()

            # p.add_mesh(fermi_surface,cmap=cmap, clim=[vmin, vmax] , opacity = 0.5 )
            # slicesx = fermi_surface.slice_along_axis(n=5, axis="x")
            # slicesz = fermi_surface.slice_along_axis(n=5, axis="z")
            # slicesy = fermi_surface.slice_along_axis(n=5, axis="y")
            # p.add_mesh(slicesx, cmap=cmap, clim=[vmin, vmax])
            # p.add_mesh(slicesz, cmap=cmap, clim=[vmin, vmax])
            # p.add_mesh(slicesy, cmap=cmap, clim=[vmin, vmax])
            # fermi_surface.save("MgB2_fermi.vtk")

        elif show_curvature == True:

            cmin = np.percentile(fermi_surface_curvature, 10)
            cmax = np.percentile(fermi_surface_curvature, 90)
            p.add_mesh(fermi_surface, scalars = fermi_surface_curvature,  cmap=cmap, clim=[cmin,  cmax])

        else:
            for isurface in range(nsurface):
                if not only_spin:
                    if mode == "plain":
                        p.add_mesh(band_surfaces[isurface], color=colors[isurface])
                        text = "Plain"
                    elif mode == "parametric":
                        p.add_mesh(
                            band_surfaces[isurface], cmap=cmap, clim=[vmin, vmax]
                        )
                        p.remove_scalar_bar()
                        text = "Projection"

                else:
                    text = "Spin Texture"

                if spin_texture:
                    # Example dataset with normals
                    # create a subset of arrows using the glyph filter
                    arrows = band_surfaces[isurface].glyph(
                    orient="vectors", factor=arrow_size)

                    if arrow_color is None:
                        p.add_mesh(arrows, cmap=cmap, clim=[vmin, vmax])
                        p.remove_scalar_bar()
                    else:
                        p.add_mesh(arrows, color=arrow_color)


        if mode != "plain" or spin_texture:
            p.add_scalar_bar(
                title=text,
                n_labels=6,
                italic=False,
                bold=False,
                title_font_size=None,
                label_font_size=None,
                position_x=0.9,
                position_y=0.01,
                color="black",
            )

        p.add_axes(
            xlabel="Kx", ylabel="Ky", zlabel="Kz", line_width=6, labels_off=False
        )

        if not perspective:
            p.enable_parallel_projection()

        p.set_background(background_color)
        # p.set_position(camera_pos)
        if not widget:
            p.show(cpos=camera_pos, screenshot=save2d)
            # p.show()
        # p.screenshot('1.png')
        # p.save_graphic('1.pdf')
        if savegif is not None:
            path = p.generate_orbital_path(n_points=36)
            p.open_gif(savegif)
            p.orbit_on_path(path)  # ,viewup=camera_pos)
        if savemp4:
            path = p.generate_orbital_path(n_points=36)
            p.open_movie(savemp4)
            p.orbit_on_path(path)  # ,viewup=camera_pos)
            # p.close()
    # p.show()
    s = boolean_add(band_surfaces)
    # s.set_color_with_cmap(cmap=cmap, vmin=vmin, vmax=vmax)
    # s.pyvista_obj.plot()
    # s.trimesh_obj.show()

    if save3d is not None:
        if save_meshio == True:
            pyvista.save_meshio(save3d,  fermi_surface)
        else:
            extention = save3d.split(".")[-1]
            s.export(save3d, extention)
    return s, fermi_surfaces, fermi_surface
    # return test



