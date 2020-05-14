# -*- coding: utf-8 -*-
"""Discretization module

This module contains the Domain and Axis classes, that are used for discretizing the modeled domain

"""

# Importing dependencies
import numpy as np
from typing import List


class Domain:
    """Domain class for containing discretizations

    Attributes:
        axis: List of axis
        axis_counter: Number of axis
        name: Domain name
    """
    def __init__(self, name: str) -> None:
        """
        Creating domain object

        Args:
            name: Domain name (optional)
        """
        self.axis: List[LinearAxis or GeometricAxis] = []
        self.axis_counter = 0
        self.name = name or 'Domain'
        print(self.name, 'has been successfully created')

    def add_axis(self, x_min: float, x_max: float, m: int, disc_by: str, name: str = None, disc_type: str = None) -> None:
        """
        Adding axis to domain object

        Args:
            x_min: Lower boundary
            x_max: Upper boundary
            m: Number of discretizations
            disc_by: Discretized dimension
            disc_type: Discretization type
            name: Axis name (optional)
        """
        if disc_type == None or disc_type == 'linear':
            self.axis.append(LinearAxis(x_min, x_max, m, disc_by, name))
        elif disc_type == 'nonlinear':
            self.axis.append(GeometricAxis(x_min, x_max, m, disc_by, name))
        print('Axis', name, 'has been successfully added to', self.name)
        self.axis_counter += 1


class LinearAxis:
    """Axis class for containing axis properties

    Attributes:
        x_min: Lower boundary
        x_max: Upper boundary
        m: Number of discretizations
        disc_by: Discretized dimension
        name: Axis name
        r: Discretization width factor
    """
    def __init__(self, x_min: float, x_max: float, m: int, disc_by: str, name: str) -> None:
        """
        Creating axis object

        Args:
            x_min: Lower boundary
            x_max: Upper boundary
            m: Number of discretizations
            disc_by: Discretized dimension
            name: Axis name (optional)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.m = m
        self.disc_by = disc_by
        self.name = name or disc_by
        # Linear grid
        self.r = (self.x_max - self.x_min) / (self.m - 1)

    def midpoints(self) -> np.ndarray:
        """
        Obtain midpoints of axis

        Returns:
            list: List of midpoints (length: m)
        """
        midpoints = np.zeros(self.m)
        # Discretization of length domain
        for i in range(self.m):
            if i == 0:
                midpoints[i] = self.x_min
            else:
                midpoints[i] = midpoints[i - 1] + self.r
        return midpoints

    def edges(self) -> np.ndarray:
        """
        Obtain edges of axis

        Returns:
            list: List of edges (length: m+1)
        """
        edges = np.zeros(self.m + 1)
        # Bin boundaries
        for i in range(self.m):
            if i == 0:
                edges[i] = self.x_min - self.r / 2
            else:
                edges[i] = edges[i - 1] + self.r
            edges[i + 1] = edges[i] + self.r
        return edges

    def widths(self) -> np.ndarray:
        """
        Obtain widths of axis

        Returns:
            list: List of widths (length: m)
        """
        widths = np.zeros(self.m)
        # Bin widths
        for i in range(self.m):
            widths[i] = self.r
        return widths


class GeometricAxis:
    """GeometricAxis class for containing axis properties

    Attributes:
        x_min: Lower boundary
        x_max: Upper boundary
        m: Number of discretizations
        disc_by: Discretized dimension
        name: Axis name
        r: Discretization width factor
    """
    def __init__(self, x_min: float, x_max: float, m: int, disc_by: str, name: str) -> None:
        """
        Creating axis object

        Args:
            x_min: Lower boundary
            x_max: Upper boundary
            m: Number of discretizations
            disc_by: Discretized dimension
            name: Axis name (optional)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.m = m
        self.disc_by = disc_by
        self.name = name or disc_by
        # Geometric grid
        self.r = (self.x_max/self.x_min)**(1/(self.m-1))-1

    def midpoints(self) -> np.ndarray:
        """
        Obtain midpoints of axis

        Returns:
            list: List of midpoints (length: m)
        """
        midpoints = np.zeros(self.m)
        # Discretization of length domain
        for i in range(self.m):
            if i == 0:
                midpoints[i] = self.x_min
            else:
                midpoints[i] = midpoints[i - 1] + self.r*midpoints[i - 1]
        return midpoints

    def edges(self) -> np.ndarray:
        """
        Obtain edges of axis

        Returns:
            list: List of edges (length: m+1)
        """
        edges = np.zeros(self.m + 1)
        # Get midpoints
        midpoints = self.midpoints()
        # Bin boundaries
        for i in range(self.m):
            if i == 0:
                edges[i] = midpoints[i] - midpoints[i]*self.r/(2+self.r)
            edges[i + 1] = midpoints[i] + midpoints[i]*self.r/(2+self.r)
        return edges

    def widths(self) -> np.ndarray:
        """
        Obtain widths of axis

        Returns:
            list: List of widths (length: m)
        """
        widths = np.zeros(self.m)
        # Get midpoints
        midpoints = self.midpoints()
        # Bin widths
        for i in range(self.m):
            widths[i] = self.r*midpoints[i]
        return widths
