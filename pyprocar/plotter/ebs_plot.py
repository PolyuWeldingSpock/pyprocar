__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import os 
import yaml
from typing import List

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from ..utils import ROOT
from ..core import ElectronicBandStructure, KPath


class EBSPlot:
    """
    A class to plot an electronic band structure.

    Parameters
    ----------
    ebs : ElectronicBandStructure
        An electronic band structure object pyprocar.core.ElectronicBandStructure.
    kpath : KPath, optional
        A kpath object pyprocar.core.KPath. The default is None.
    ax : mpl.axes.Axes, optional
        A matplotlib Axes object. If provided the plot will be located at that ax.
        The default is None.
    spin : List[int], optional
        A list of the spins
        The default is None.

    Returns
    -------
    None.

    """
    def __init__(self, 
                    ebs:ElectronicBandStructure, 
                    kpath:KPath=None, 
                    ax:mpl.axes.Axes=None, 
                    spins:List[int]=None, 
                    **kwargs):

        with open(os.path.join(ROOT,'pyprocar','cfg','band_structure.yml'), 'r') as file:
            self.plot_opt = yaml.safe_load(file)
        self.update_config(kwargs)

        self.ebs = ebs
        self.kpath = kpath
        self.spins = spins
        if self.spins is None:
            self.spins = range(self.ebs.nspins)
        self.nspins = len(self.spins)
        if self.ebs.is_non_collinear:
            self.spins = [0]
        self.handles = []

        
        figsize=tuple(self.plot_opt['figure_size']['value'])
        if ax is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax
        # need to initiate kpath if kpath is not defined.
        self.x = self._get_x()

        self._initiate_plot_args()
        return None
        
    def _initiate_plot_args(self):
        """Helper method to initialize the plot options
        """
        self.set_xticks()
        self.set_yticks()
        self.set_xlabel()
        self.set_ylabel()
        self.set_xlim()
        self.set_ylim()
        
    def _get_x(self):
        """
        Provides the x axis data of the plots

        Returns
        -------
        np.ndarray
            x-axis data.

        """
        pos = 0
        if self.kpath is not None and self.kpath.nsegments == len(self.kpath.ngrids):
            for isegment in range(self.kpath.nsegments):
                kstart, kend = self.kpath.special_kpoints[isegment]

                distance = np.linalg.norm(kend - kstart)
                if isegment == 0:
                    x = np.linspace(pos, pos + distance,
                                    self.kpath.ngrids[isegment])
                else:
                    x = np.append(
                        x,
                        np.linspace(pos, pos + distance,
                                    self.kpath.ngrids[isegment]),
                        axis=0,
                    )
                pos += distance
        else :
            x = np.arange(0, self.ebs.nkpoints)

        return np.array(x).reshape(-1,)

    def plot_bands(self):
        """
        Plot the plain band structure.

        Returns
        -------
        None.

        """

        
        for ispin in self.spins:
            if len(self.spins)==1:
                color=self.plot_opt['color']['value']
            else:
                color=self.plot_opt['color']['value'][ispin],
            for iband in range(self.ebs.nbands):
                handle = self.ax.plot(
                    self.x, self.ebs.bands[:, iband, ispin], 
                    color=color, 
                    alpha=self.plot_opt['opacity']['value'][ispin], 
                    linestyle=self.plot_opt['linestyle']['value'][ispin], 
                    label=self.plot_opt['label']['value'][ispin], 
                    linewidth=self.plot_opt['linewidth']['value'][ispin],
                )
                self.handles.append(handle)

    def plot_scatter(self,
                     width_mask:np.ndarray=None,
                     color_mask:np.ndarray=None,
                     spins:List[int]=None,
                     width_weights:np.ndarray=None,
                     color_weights:np.ndarray=None,
                     ):
        """A method to plot a scatter plot

        Parameters
        ----------
        width_mask : np.ndarray, optional
            The width mask, by default None
        color_mask : np.ndarray, optional
            The color mask, by default None
        vmin : float, optional
            Value to normalize the minimum projection value., by default None
        vmax : float, optional
            Value to normalize the maximum projection value., by default None
        spins : List[int], optional
            A list of spins, by default None
        width_weights : np.ndarray, optional
            The width weight of each point, by default None
        color_weights : np.ndarray, optional
            The color weights at each point, by default None
        """
        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]
        
        if width_weights is None:
            width_weights = np.ones_like(self.ebs.bands)
            markersize = self.plot_opt['makersize']['value']
        else:
            markersize =[l*30 for l in self.plot_opt['makersize']['value']]


        if width_mask is not None or color_mask is not None:
            if width_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(width_weights) < width_mask)
            if color_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(color_weights) < color_mask)
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.ebs.bands, False)

        if color_weights is not None:
            vmin=self.plot_opt['clim']['value'][0]
            vmin=self.plot_opt['clim']['value'][1]
            if vmin is None: 
                vmin = color_weights.min()
            if vmax is None:
                vmax = color_weights.max()

        for ispin in spins:
            for iband in range(self.ebs.nbands):
                if len(self.spins)==1:
                    color=self.plot_opt['color']['value']
                else:
                    color=self.plot_opt['color']['value'][ispin],
                if color_weights is None:
                    sc = self.ax.scatter(
                        self.x,
                        mbands[:, iband, ispin],
                        c=color,
                        s=width_weights[:, iband, ispin].round(
                            2)*markersize[ispin],
                        # edgecolors="none",
                        linewidths=self.plot_opt['linewidth']['value'][ispin],
                        cmap=self.plot_opt['cmap']['value'],
                        vmin=vmin,
                        vmax=vmax,
                        marker=self.plot_opt['maker']['value'][ispin],
                        alpha=self.plot_opt['opacity']['value'][ispin],
                    )
                else:
                    sc = self.ax.scatter(
                        self.x,
                        mbands[:, iband, ispin],
                        c=color_weights[:, iband, ispin].round(2),
                        s=width_weights[:, iband, ispin].round(2)*markersize[ispin],
                        # edgecolors="none",
                        linewidths=self.plot_opt['linewidth']['value'][ispin],
                        cmap=self.plot_opt['cmap']['value'],
                        vmin=vmin,
                        vmax=vmax,
                        marker=self.plot_opt['maker']['value'][ispin],
                        alpha=self.plot_opt['opacity']['value'][ispin],
                    )
        if self.plot_opt['plot_color_bar']['value'] and color_weights is not None:
            cb = self.fig.colorbar(sc, ax=self.ax)
            cb.ax.tick_params(labelsize=20)

    def plot_parameteric(
        self,
        spins:List[int]=None,
        width_mask:np.ndarray=None,
        color_mask:np.ndarray=None,
        width_weights:np.ndarray=None,
        color_weights:np.ndarray=None,
        ):
        """A method to plot a scatter plot

        Parameters
        ----------
        spins : List[int], optional
            A list of spins, by default None
        vmin : float, optional
            Value to normalize the minimum projection value., by default None
        vmax : float, optional
            Value to normalize the maximum projection value., by default None
        width_mask : np.ndarray, optional
            The width mask, by default None
        color_mask : np.ndarray, optional
            The color mask, by default None
        width_weights : np.ndarray, optional
            The width weight of each point, by default None
        color_weights : np.ndarray, optional
            The color weights at each point, by default None
        """
        if width_weights is None:
            width_weights = np.ones_like(self.ebs.bands)
            linewidth = self.plot_opt['linewidth']['value']
        else:
            linewidth = [l*5 for l in self.plot_opt['linewidth']['value']]

        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]
        
        
        if width_mask is not None or color_mask is not None:
            if width_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(width_weights) < width_mask)
            if color_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(color_weights) < color_mask)
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.ebs.bands, False)

        if color_weights is not None:
            vmin=self.plot_opt['clim']['value'][0]
            vmax=self.plot_opt['clim']['value'][1]
            if vmin is None:
                vmin = color_weights.min()
            if vmax is None:
                vmax = color_weights.max()
                
            norm = mpl.colors.Normalize(vmin, vmax)

        for ispin in spins:
            for iband in range(self.ebs.nbands):
                if len(self.spins)==1:
                    color=self.plot_opt['color']['value']
                else:
                    color=self.plot_opt['color']['value'][ispin],
                points = np.array(
                    [self.x, mbands[:, iband, ispin]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # this is to delete the segments on the high sym points
                x = self.x
                # segments = np.delete(
                #     segments, np.where(x[1:] == x[:-1])[0], axis=0)
                if color_weights is None:
                    lc = LineCollection(
                        segments, colors=color, 
                        linestyle=self.plot_opt['linestyle']['value'][ispin])
                else:
                    lc = LineCollection(
                        segments, cmap=plt.get_cmap(self.plot_opt['cmap']['value']), norm=norm)
                    lc.set_array(color_weights[:, iband, ispin])
                lc.set_linewidth(
                    width_weights[:, iband, ispin]*linewidth[ispin])
                lc.set_linestyle(self.plot_opt['linestyle']['value'][ispin])
                handle = self.ax.add_collection(lc)
            # if color_weights is not None:
            #     handle.set_color(color_map[iweight][:-1].lower())
            handle.set_linewidth(linewidth)
            self.handles.append(handle)

        if self.plot_opt['plot_color_bar']['value'] and color_weights is not None:
            cb = self.fig.colorbar(lc, ax=self.ax)
            cb.ax.tick_params(labelsize=20)

    def plot_parameteric_overlay(self,
                                 spins:List[int]=None,
                                 vmin:float=None,
                                 vmax:float=None,
                                 weights:np.ndarray=None,
                                 ):
        """A method to plot the parametric overlay

        Parameters
        ----------
        spins : List[int], optional
            A list of spins, by default None
        vmin : float, optional
            Value to normalize the minimum projection value, by default None
        vmax : float, optional
            Value to normalize the maximum projection value, by default None
        weights : np.ndarray, optional
            The weights of each point, by default None
        """
        
        linewidth = [l*7 for l in self.plot_opt['linewidth']['value']]
        if type(self.plot_opt['cmap']['value']) is str:
            color_map = ['Reds', "Blues", "Greens",
                     "Purples", "Oranges", "Greys"]
        else:
            color_map=self.plot_opt['cmap']['value']
        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]
        for iweight, weight in enumerate(weights):
            vmin=self.plot_opt['clim']['value'][0]
            vmax=self.plot_opt['clim']['value'][1]
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 1
            norm = mpl.colors.Normalize(vmin, vmax)
            for ispin in spins:
                # plotting
                for iband in range(self.ebs.nbands):
                    points = np.array(
                        [self.x, self.ebs.bands[:, iband, ispin]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1)
                    # this is to delete the segments on the high sym points
                    x = self.x
                    segments = np.delete(
                        segments, np.where(x[1:] == x[:-1])[0], axis=0)
                    lc = LineCollection(
                        segments, cmap=plt.get_cmap(color_map[iweight]), norm=norm, 
                        alpha=self.plot_opt['opacity']['value'][ispin])
                    lc.set_array(weight[:, iband, ispin])
                    lc.set_linewidth(weight[:, iband, ispin]*linewidth[ispin])
                    handle = self.ax.add_collection(lc)
            handle.set_color(color_map[iweight][:-1].lower())
            handle.set_linewidth(linewidth)
            self.handles.append(handle)

            if self.plot_opt['plot_color_bar']['value']:
                cb = self.fig.colorbar(lc, ax=self.ax)
                cb.ax.tick_params(labelsize=20)

    def set_xticks(self, 
        tick_positions:List[int]=None, 
        tick_names:List[str]=None, 
        color:str="black"):
        """A method to set the x ticks

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None
        color : str, optional
            A color for the ticks, by default "black"
        """

        if self.kpath is not None:
            if tick_positions is None:
                tick_positions = self.kpath.tick_positions
            if tick_names is None:
                tick_names = self.kpath.tick_names
            for ipos in tick_positions:
                self.ax.axvline(self.x[ipos], color=color)

            self.ax.set_xticks(self.x[tick_positions])
            self.ax.set_xticklabels(tick_names)
            self.ax.tick_params(
                which='major',
                axis='x',
                direction='in')

    def set_yticks(self, 
        major:float=None, 
        minor:float=None, 
        interval:List[float]=None):
        """A method to set the y ticks

        Parameters
        ----------
        major : float, optional
            A float to set the major tick locator, by default None
        minor : float, optional
            A float to set the the minor tick Locator, by default None
        interval : List[float], optional
            The interval of the ticks, by default None
        """
        if (major is None or minor is None):
            if interval is None:
                interval = (self.ebs.bands.min()-abs(self.ebs.bands.min())
                            * 0.1, self.ebs.bands.max()*1.1)

            interval = abs(interval[1] - interval[0])
            if interval < 30 and interval >= 20:
                major = 5
                minor = 1
            elif interval < 20 and interval >= 10:
                major = 4
                minor = 0.5
            elif interval < 10 and interval >= 5:
                major = 2
                minor = 0.2
            elif interval < 5 and interval >= 3:
                major = 1
                minor = 0.1
            elif interval < 3 and interval >= 1:
                major = 0.5
                minor = 0.1
            else:
                pass
        if major is not None and minor is not None:
            self.ax.yaxis.set_major_locator(MultipleLocator(major))
            self.ax.yaxis.set_minor_locator(MultipleLocator(minor))
        self.ax.tick_params(
            which='major',
            axis="y",
            direction="inout",
            width=1,
            length=5,
            labelright=False,
            right=True,
            left=True) 

        self.ax.tick_params(
            which='minor',
            axis="y",
            direction="in",
            left=True,
            right=True)

    def set_xlim(self, interval:List[float]=None):
        """A method to set the x limit

        Parameters
        ----------
        interval : List[float], optional
            A list containing the begining and the end of the interval, by default None
        """
        if interval is None:
            interval = (self.x[0], self.x[-1])
        self.ax.set_xlim(interval)

    def set_ylim(self, interval:List[float]=None):
        """A method to set the y limit

        Parameters
        ----------
        interval : List[float], optional
            A list containing the begining and the end of the interval, by default None
        """
        if interval is None:
            interval = (self.ebs.bands.min()-abs(self.ebs.bands.min())
                        * 0.1, self.ebs.bands.max()*1.1)
        self.ax.set_ylim(interval)

    def set_xlabel(self, label:str="K vector"):
        """A method to set the x label

        Parameters
        ----------
        label : str, optional
            String fo the x label name, by default "K vector"
        """
        self.ax.set_xlabel(label)

    def set_ylabel(self, label:str=r"E - E$_F$ (eV)"):
        """A method to set the y label

        Parameters
        ----------
        label : str, optional
            String fo the y label name, by default r"E - E$ (eV)"
        """
        self.ax.set_ylabel(label)

    def set_title(self, title:str="Band Structure"):
        """A method to set the title

        Parameters
        ----------
        title : str, optional
            String for the title, by default "Band Structure"
        """
        self.ax.set_title(label=title)

    def legend(self, labels:List[str]=None):
        """A methdo to plot the legend

        Parameters
        ----------
        labels : List[str], optional
            A list of strings for the labels of each element for the legend, by default None
        """
        if labels == None:
            labels = self.plot_opt['label']['value']
        self.ax.legend(self.handles, labels)

    def draw_fermi(self, 
                fermi_level:float=0, 
                color:str="blue", 
                linestyle:str="dotted", 
                linewidth:float=1):
        """A method to draw the fermi line

        Parameters
        ----------
        color : str, optional
            The color of the fermi line, by default "blue"
        linestyle : str, optional
            The linestyle, by default "dotted"
        linewidth : float, optional
            The linewidth, by default 1
        """
        self.ax.axhline(y=fermi_level, color=color, linestyle=linestyle, linewidth=linewidth)

    def grid(self):
        """A method to plot a grid
        """
        self.ax.grid(
            self.plot_opt['grid']['value'],
            which=self.plot_opt['grid_which']['value'],
            color=self.plot_opt['grid_color']['value'],
            linestyle=self.plot_opt['grid_linestlye']['value'],
            linewidth=self.plot_opt['grid_linewidth']['value'])
    
    def show(self):
        """A method to show the plot
        """
        plt.show()

    def save(self, filename:str='bands.pdf'):
        """A method to save the plot

        Parameters
        ----------
        filename : str, optional
            A string for the file name, by default 'bands.pdf'
        """
        plt.savefig(filename, dpi=self.plot_opt['dpi']['value'], bbox_inches="tight")
        plt.clf()
    
    def update_config(self, config_dict):
        for key,value in config_dict.items():
            self.plot_opt[key]['value']=value
     
