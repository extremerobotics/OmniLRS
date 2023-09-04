from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from typing import List, Tuple
import numpy as np
import datetime
import pickle
import cv2

class CraterGenerator:
    """
    Generates craters DEM from a set of spline profiles and randomizes their 2D appearance"""

    def __init__(self, profiles_path,
                       seed:int = 42,
                       min_xy_ratio:float = 0.85,
                       max_xy_ratio:float = 1,
                       resolution:float = 0.01,
                       pad_size:int = 500,
                       random_rotation:bool = True,
                       z_scale:float = 1):
        """
        Args:
            profiles_path (str): path to the pickle file containing the spline profiles.
            seed (int): random seed.
            min_xy_ratio (float): minimum xy ratio of the crater.
            max_xy_ratio (float): maximum xy ratio of the crater.
            resolution (float): resolution of the DEM (in meters per pixel).
            pad_size (int): size of the padding to add to the DEM.
            random_rotation (bool): whether to randomly rotate the craters.
            z_scale (float): scale of the craters."""

        self._profiles_path = profiles_path
        self._resolution = resolution
        self._min_xy_ratio = min_xy_ratio
        self._max_xy_ratio = max_xy_ratio
        self._z_scale = z_scale
        self._random_rotation = random_rotation
        self._pad_size = pad_size
        self._profiles = None
        self._rng = np.random.default_rng(seed)

        self.loadProfiles()

    def loadProfiles(self) -> None:
        """
        Loads the half crater spline profiles from a pickle file."""

        with open(self._profiles_path,"rb") as handle:
            self._profiles = pickle.load(handle)

    def sat_gaussian(self, x:np.ndarray, mu1:float, mu2:float, std:float) -> np.ndarray:
        """
        Saturates a gaussian function to its maximum between mu1 and mu2 with a standard deviation of std.

        Args:
            x: input array.
            mu1: gaussian mu lower bound.
            mu2: gaussian mu upper bound.
            std: standard deviation.
        
        Returns:
            np.ndarray: saturated gaussian."""

        shape = x.shape
        x = x.flatten()
        x[x<mu1] = np.exp(-0.5*((x[x<mu1] - mu1)/std)**2)
        x[x>mu2] = np.exp(-0.5*((x[x>mu2] - mu2)/std)**2)
        x[(x>=mu1) & (x<=mu2)] = 1.0
        x = x  / (std * np.sqrt(2*np.pi))
        x = x.reshape(shape)
        return x

    def centeredDistanceMatrix(self, n:int) -> Tuple[np.ndarray, int]:
        """
        Generates a distance matrix centered at the center of the matrix.
        
        Args:
            n (int): size of the matrix
        
        Returns:
            list: distance matrix, size of the matrix."""

        # Makes sure the matrix size is odd
        n = n + ((n % 2) == 0)

        # Generates a profile to deform the crater
        tmp_y = self._rng.uniform(0.95,1,9)
        tmp_y = np.concatenate([tmp_y,[tmp_y[0]]],axis=0)
        tmp_x = np.linspace(0,1,tmp_y.shape[0])
        s = CubicSpline(tmp_x,tmp_y,bc_type=((1, 0.0), (1, 0.0)))

        # Generates a profile to add marks that converges toward the center of the crater
        tmp_y = self._rng.uniform(0.0,0.01,45)
        tmp_y = np.concatenate([tmp_y,[tmp_y[0]]],axis=0)
        tmp_x = np.linspace(0,1,tmp_y.shape[0])
        s2 = CubicSpline(tmp_x,tmp_y,bc_type=((1, 0.0), (1, 0.0)))

        # Generates the deformation matrix
        m = np.zeros([n,n])
        x,y = np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n))
        theta = np.arctan2(y,x)
        fac = s(theta/(2*np.pi) + 0.5)
        # Generates the marks matrix
        marks = s2(theta/(2*np.pi) + 0.5) * n/2 * self._rng.uniform(0,1)

        # Generates the distance matrix
        sx = self._rng.uniform(self._min_xy_ratio, self._max_xy_ratio)
        sy = 1.0
        x,y = np.meshgrid(range(n),range(n))
        m = np.sqrt(((x-(n/2)+1)*1/sx)**2+((y-(n/2)+1)*1/sy)**2)

        # Deforms the distance matrix 
        m = m*fac

        # Smoothes the marks such that they are not present on the outer edge of the crater
        sat = self.sat_gaussian(m, 0.15*n/2, 0.45*n/2, 0.05*n/2)
        sat = (sat - sat.min()) / (sat.max() - sat.min())

        # Adds the marks to the distance matrix
        m = m + marks*sat

        # Rotate the matrix
        theta = int(self._rng.uniform(0,360))
        m = rotate(m, theta, reshape=False, cval=n/2)

        # Saturate the matrix such that the distance is not greater than the radius of the crater
        m[m > n/2] = n/2
        return m, n

    def applyProfile(self, profile: CubicSpline, distance: np.ndarray, size:int) -> np.ndarray:
        """
        Applies a profile to the distance matrix.
        
        Args:
            profile (CubicSpline): profile to apply.
            distance (np.ndarray): distance matrix.
            size (int): size of the matrix.
        
        Returns:
            np.ndarray: crater DEM."""

        crater = profile(2*distance/size)
        return crater

    def getProfile(self, index:int) -> CubicSpline:
        """
        Gets a random profile from the list of profiles.
        
        Args:
            index (int): index of the profile to get.
        
        Returns:
            CubicSpline: profile."""

        profile = None
        if index < -1:
            raise ValueError("Unknown profile")
        elif index == -1:
            profile = self._rng.choice(self._profiles)
        elif index < len(self._profiles):
            profile = self._profiles[i]
        else:
            raise ValueError("Unknown profile")
        return profile

    def generateCrater(self, size:int, s_index:int = -1) -> Tuple[np.ndarray, int]:
        """
        Generates a crater DEM.
        
        Args:
            size (float): size of the crater.
            s_index (int): index of the profile to use.
        
        Returns:
            tuple: crater DEM, size of the crater."""

        distance, size = self.centeredDistanceMatrix(size)
        profile = self.getProfile(s_index)
        crater = self.applyProfile(profile, distance, size) * size/2.0 * self._z_scale * self._resolution
        return crater, size

    def generateCraters(self, DEM:np.ndarray, coords:np.ndarray, radius:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a DEM with craters.
        
        Args:
            DEM (np.ndarray): DEM to add craters to.
            coords (np.ndarray): coordinates of the craters (in meters).
        
        Returns:
            np.ndarray: unpadded DEM with craters, and matching mask."""

        DEM_padded = np.zeros((self._pad_size*2 + DEM.shape[0], self._pad_size*2 + DEM.shape[1]))
        mask_padded = np.ones_like(DEM_padded)
        DEM_padded[self._pad_size:-self._pad_size, self._pad_size:-self._pad_size] = DEM
        for coord, rad in zip(coords, radius):
            rad = int(rad*2/self._resolution)
            coord = coord/self._resolution
            c,rad = self.generateCrater(int(rad))
            coord2 = (coord + self._pad_size).astype(np.int64)
            coord = (coord - rad/2 + self._pad_size).astype(np.int64)
            DEM_padded[coord[0]:coord[0]+rad,coord[1]:coord[1]+rad] += c
            mask_padded = cv2.circle(mask_padded, (coord2[1], coord2[0]), int(rad/4), 0, -1)
        mask_padded[:self._pad_size+1, :] = 0
        mask_padded[:, :self._pad_size+1] = 0
        mask_padded[-self._pad_size-1:, :] = 0
        mask_padded[:, -self._pad_size-1:] = 0
        return DEM_padded[self._pad_size:-self._pad_size,self._pad_size:-self._pad_size], mask_padded[self._pad_size:-self._pad_size,self._pad_size:-self._pad_size]

class Distribute:
    """
    Distributes craters on a DEM using a Poisson process with hardcore rejection."""

    def __init__(self, x_size:float = 10,
                       y_size:float = 10,
                       densities:List[float] = [0.25,1.5,5],
                       radius:List[Tuple[float]] = [(1.5,2.5),(0.75,1.5),(0.25,0.5)],
                       num_repeat:int = 0,
                       seed:int = 42):
        """
        Args:
            x_size (float): size of the DEM in the x direction (in meters).
            y_size (float): size of the DEM in the y direction (in meters).
            densities (float): densities of the craters (in units per square meters).
            radius (list): min and max radii of the craters (in meters).
            num_repeat (int): number of times to repeat the hardcore rejection.
            seed (int): random seed."""
        
        self._x_max = x_size
        self._y_max = y_size
        self._densities = densities
        self._radius = radius
        self._area = self._x_max*self._y_max
        self._num_repeat = num_repeat
        self._rng = np.random.default_rng(seed)

    def sampleFromPoisson(self, l:float, r_minmax:Tuple[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples from a Poisson process.
        
        Args:
            l (float): density of the Poisson process (in units per square meters).
            r_minmax (tuple): minimum and maximum radius of the craters (in meters).
        
        Returns:
            tuple: coordinates and radius of the craters"""

        num_points = self._rng.poisson(self._area*l)
        radius = self._rng.uniform(r_minmax[0], r_minmax[1], num_points)
        x_coords = self._rng.uniform(0, self._x_max, num_points)
        y_coords = self._rng.uniform(0,self._y_max, num_points)
        return np.stack([x_coords,y_coords]).T, radius

    def hardcoreRejection(self, coords:np.ndarray, radius:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs hardcore rejection on the craters.
        
        Args:
            coords (np.ndarray): coordinates of the craters (in meters).
            radius (np.ndarray): radii of the craters (in meters).
        
        Returns:
            tuple: coordinates of the craters (in meters)."""

        mark_age = self._rng.uniform(0,1,coords.shape[0])
        boole_keep = np.zeros(mark_age.shape[0], dtype=bool)
        for i in range(mark_age.shape[0]):
            dist_tmp = np.linalg.norm(coords[i] - coords, axis=-1)
            in_disk = (dist_tmp < radius[i]) & (dist_tmp >0)
            if len(mark_age[in_disk]) == 0:
                boole_keep[i] = True
            else:
                boole_keep[i] = all(mark_age[i] < mark_age[in_disk])
        return coords[boole_keep], radius[boole_keep]

    def checkPrevious(self, new_coords: np.ndarray, radius: np.ndarray, prev_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Checks if the new craters are not in the previous craters.
        
        Args:
            new_coords (np.ndarray): coordinates of the new craters.
            radius (np.ndarray): radii of the new craters.
            prev_coords (np.ndarray): coordinates of the previous craters.
        
        Returns:
            tuple: the coordinates of the new craters, the radii of the new craters."""

        boole_keep = np.ones(new_coords.shape[0], dtype=bool)
        if prev_coords is None:
            boole_keep = np.ones(new_coords.shape[0], dtype=bool)
        else:
            for i in range(prev_coords[0].shape[0]):
                dist_tmp = np.linalg.norm(prev_coords[0][i] - new_coords, axis=-1)
                in_disk = (dist_tmp < prev_coords[1][i]) & (dist_tmp > 0)
                boole_keep[in_disk] = False
        return new_coords[boole_keep], radius[boole_keep]

    def simulateHCPoissonProcess(self, l:float, r_minmax:Tuple[float], prev_coords:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates a hardcore Poisson process.
        
        Args:
            l (float): density of the Poisson process (in units per square meters).
            r_minmax (tuple): minimum and maximum radius of the craters (in meters).
            prev_coords (np.ndarray): coordinates of the previous craters (in meters).
        
        Returns:
            tuple: coordinates of the craters, radii of the craters."""

        coords, radius = self.sampleFromPoisson(l, r_minmax)
        for _ in range(self._num_repeat):
            coords, radius = self.hardcoreRejection(coords, radius)
            new_coords, new_radius = self.sampleFromPoisson(l, r_minmax)
            coords = np.concatenate([coords, new_coords])
            radius = np.concatenate([radius, new_radius])
            self.checkPrevious(coords, radius, prev_coords)
        coords, radius = self.hardcoreRejection(coords, radius)
        coords, radius = self.checkPrevious(coords, radius, prev_coords)
        return coords, radius

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the hardcore Poisson process for all the densities and radius in order.
        
        Returns:
            tuple: coordinates of the craters, radii of the craters."""

        prev_coords = None
        for d, r_minmax in zip(self._densities, self._radius):
            new_coords, new_radius = self.simulateHCPoissonProcess(d, r_minmax, prev_coords)
            if prev_coords is not None:
                prev_coords = (np.concatenate([prev_coords[0], new_coords],axis=0), np.concatenate([prev_coords[1], new_radius],axis=0))
            else:
                prev_coords = (new_coords, new_radius)
        return prev_coords

class BaseTerrainGenerator:
    """
    Generates a random terrain DEM."""

    def __init__(self, x_size:float = 10,
                       y_size:float = 10,
                       resolution:float = 0.01,
                       max_elevation:float = 0.5,
                       min_elevation:float = -0.25,
                       seed:int = 42,
                       z_scale:float = 50):
        """
        Args:
            x_size (float): size of the DEM in the x direction (in meters).
            y_size (float): size of the DEM in the y direction (in meters).
            resolution (float): resolution of the DEM (in meters per pixel).
            max_elevation (float): maximum elevation of the DEM (in meters).
            min_elevation (float): minimum elevation of the DEM (in meters).
            seed (int): random seed.
            z_scale (float): scale of the DEM."""

        self._min_elevation = min_elevation
        self._max_elevation = max_elevation
        self._x_size = int(x_size / resolution)
        self._y_size = int(y_size / resolution)
        self._DEM = np.zeros((self._x_size, self._y_size),dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._z_scale = z_scale

    def generateRandomTerrain(self) -> np.ndarray:
        """
        Generates a random terrain DEM.
        
        Returns:
            DEM (np.ndarray): random terrain DEM."""

        # Generate low frequency noise to simulate large scale terrain features.
        lr_noise = np.zeros((4,4))
        lr_noise[:-1,1:] = self._rng.uniform(self._min_elevation, self._max_elevation, [3,3])
        hr_noise = cv2.resize(lr_noise, (self._y_size, self._x_size), interpolation=cv2.INTER_CUBIC)
        self._DEM += hr_noise
        # Generate high frequency noise to simulate small scale terrain features.
        lr_noise = self._rng.uniform(self._min_elevation*0.01, self._max_elevation*0.01, [100,100])
        hr_noise = cv2.resize(lr_noise, (self._y_size, self._x_size), interpolation=cv2.INTER_CUBIC)
        self._DEM += hr_noise
        # Normalize the DEM between 0 and 1 and scale it to the desired elevation range.
        self._DEM = (self._DEM - self._DEM.min()) / (self._DEM.max() - self._DEM.min())
        # Scale the DEM to the desired elevation range.
        self._DEM = self._DEM * (self._max_elevation - self._min_elevation) - self._min_elevation
        return self._DEM * self._z_scale

class GenerateProceduralMoonYard:
    """
    Generates a random terrain DEM with craters."""

    def __init__(self, crater_profiles_path,
                       x_size:float = 10,
                       y_size:float = 6.5,
                       resolution:float = 0.01,
                       max_elevation:float = 0.25,
                       min_elevation:float = -0.025,
                       z_scale:float = 1,
                       pad:int = 500,
                       num_repeat:float = 0,
                       densities:List[float] = [0.025,0.05,0.5],
                       radius:List[Tuple[float,float]] = [(1.5,2.5),(0.75,1.5),(0.25,0.5)],
                       seed:int = 42):
        """
        Args:
            crater_profiles_path (str): path to the pickle file containing the crater profiles.
            x_size (float): size of the DEM in the x direction (in meters).
            y_size (float): size of the DEM in the y direction (in meters).
            resolution (float): resolution of the DEM (in meters per pixel).
            max_elevation (float): maximum elevation of the DEM (in meters).
            min_elevation (float): minimum elevation of the DEM (in meters).
            z_scale (float): scale of the DEM.
            pad (int): size of the padding to add to the DEM.
            num_repeat (int): number of times to repeat the hardcore rejection.
            densities (list): densities of the craters (in units per square meters).
            radius (list): radii of the craters (in meters).
            seed (float): random seed."""

        self.T = BaseTerrainGenerator(x_size=x_size, y_size=y_size, resolution=resolution, max_elevation=max_elevation, min_elevation=min_elevation, z_scale=z_scale, seed=seed)
        self.D = Distribute(x_size=x_size, y_size=y_size, densities=densities, radius=radius, num_repeat=num_repeat, seed=seed)
        self.G = CraterGenerator(crater_profiles_path, resolution=resolution, pad_size=pad, seed=seed)

    def randomize(self) -> np.ndarray:
        """
        Generates a random terrain DEM with craters.
        
        Returns:
            np.ndarray: random terrain DEM with craters"""

        DEM = self.T.generateRandomTerrain()
        coords, radius = self.D.run()
        DEM, mask = self.G.generateCraters(DEM, coords, radius)
        return DEM, mask

if __name__ == "__main__":
    craters_profiles_path = "crater_spline_profiles.pkl"
    start = datetime.datetime.now()
    G = GenerateProceduralMoonYard(craters_profiles_path)
    DEMs = [G.randomize() for i in range(4)]
    end = datetime.datetime.now()

    print((end - start).total_seconds())
    for DEM, mask in DEMs:
        plt.figure()
        plt.imshow(DEM, cmap="jet")
        plt.figure()
        plt.imshow(mask, cmap="jet")
    plt.show()
