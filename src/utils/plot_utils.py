from abc import ABC, abstractmethod

class McChainPlotter(ABC):
    @abstractmethod
    def plot(self,*args,**kwargs):
        pass


class DramPlotter(McChainPlotter):
    def __init__(self, fit_dict:dict):
        if 'dram_results' in fit_dict:
            result_dict = fit_dict['dram_results'] 
        else:
            result_dict = fit_dict
            
        self.chain = result_dict['chain']
        self.map = result_dict['map']
        self.y_map = result_dict['y_map']
        self.theta_mean = result_dict['theta_mean']
        self.theta_credible_intervals = result_dict['theta_credible_intervals']
        self.all_trajectories = result_dict['all_trajectories']
        self.y_mean = result_dict['y_mean']
        self.y_up = result_dict['y_up']
        self.y_dw = result_dict['y_dw']
        self.dic = result_dict['dic']
    
    def plot(self, ax, t, mean_color='blue', fill_color='lightsteelblue',vert_line=None,label='Prediction', width=5):
        ax.plot(t, self.y_mean, lw=width, label=label, color=mean_color)
        ax.fill_between(t, self.y_up, self.y_dw, color=fill_color, alpha=0.5)
        if vert_line:
            ax.vlines(vert_line, min(self.y_dw), max(self.y_up), linestyles ="dotted")
            ax.text(vert_line+0.2, max(self.y_mean), "Fit limit", rotation=90, verticalalignment='center')
        return ax
