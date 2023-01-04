import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Region:
    def __init__(self,cords:list, is_square:bool) -> None:
        self.sq = is_square
        self.cords = np.array(cords,dtype=float)
        self.poly = Polygon(cords)
        self.is_in_region = np.vectorize(lambda p: self.poly.contains(Point(p)), signature='(n)->()')
    
    def get_min_x(self) -> float:
        return self.cords[:,0].min()
    def get_max_x(self) -> float:
        return self.cords[:,0].max()
    def get_min_y(self) -> float:
        return self.cords[:,1].min()
    def get_max_y(self) -> float:
        return self.cords[:,1].max()
        
        
class Partition:
    def __init__(self,func, region : Region, method:str, *method_params) -> None:
        self.rg = region
        self.f = func
        self.__measures = None
        self.__vals = None
        self.__fineness = None
        self.mask = None
        self.grid = None
        if method == "grid":
            self.__method_grid(*method_params)
    
    def __method_grid(self, n_x, n_y) -> None: 
        measure = (self.rg.get_max_x() - self.rg.get_min_x()) * (self.rg.get_max_y() - self.rg.get_min_y())/(n_x*n_y)
        self.__fineness = ((self.rg.get_max_x() - self.rg.get_min_x())/n_x)**2 + ((self.rg.get_max_y() - self.rg.get_min_y())/n_y)**2
        self.__measures = np.array([measure]*(n_x*n_y))
        self.grid = np.meshgrid(np.linspace(self.rg.get_min_x(),self.rg.get_max_x(),n_x+1)[:-1], np.linspace(self.rg.get_min_y(),self.rg.get_max_y(),n_y+1)[:-1])
        self.grid = np.concatenate([self.grid[0].reshape(-1,1), self.grid[1].reshape(-1,1)],axis=1)
        self.__vals = self.f(self.grid[:,0], self.grid[:,1])
        self.mask = np.ones((n_x*n_y),dtype=bool)
        if not self.rg.sq:
            self.mask = self.rg.is_in_region(self.grid)
        self.__vals[~self.mask] = 0
        
        
    def get_measures(self) -> np.array:
        return self.__measures
    def get_vals(self) -> np.array:
        return self.__vals
    def get_fineness(self)->float:
        return self.__fineness
    
class Integrator_Riemann:
    def __init__(self, func,info_dict=None) -> None: 
        self.info_dict = info_dict
        self.f = func
        self.__err = None
        
    def generate_riemann_sum(self, part:Partition) -> dict:
        if self.info_dict != None:
            n_x = self.info_dict["n_x"]
            n_y = self.info_dict["n_y"]
            diff_x = self.info_dict["diff_x"]
            diff_y = self.info_dict["diff_y"]
            max_f = self.info_dict["max_func"]
            max_f_y = self.info_dict["max_dfunc_dy"]
            max_f_x = self.info_dict["max_dfunc_dx"]
            max_f_xy = self.info_dict["max_dfunc_dx_dy"]
            err_in = 0.5*(max_f_y*(diff_y**2)*diff_x/n_y + max_f_x*(diff_x**2)*diff_y/n_x) + 0.25*max_f_xy*((diff_y*diff_x)**2)/(n_x*n_y)
            err_border = max_f*diff_x*diff_y*(1/n_x + 1/n_y)
            self.__err = err_in # + err_border
        out_dict = {"riemann_sum": np.sum(part.get_measures()*part.get_vals()),
                    "fineness":part.get_fineness(),
                    "error": self.__err}
        return out_dict
        