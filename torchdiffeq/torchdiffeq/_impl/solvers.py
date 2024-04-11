import abc
import torch
from .event_handling import find_event
from .misc import _handle_unused_kwargs


class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        """
        Initialize the object.

        Parameters:
            dtype (dtype): The data type.
            y0 (array-like): The initial condition.
            norm (float): The norm for the optimization.

        Returns:
            None
        """
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype

        self.norm = norm

    def _before_integrate(self, t):
        """
        This function is called before integrating the system. It takes a parameter t and does not return anything.
        """
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        """
        A description of the _advance function, its parameters, and its return types.
        """
        return set()
    
    def integrate(self, t):
        """
        Integrate the differential equation.

        Args:
            t: Time points at which to solve for y.

        Returns:
            solution: Tensor containing the solution values at each time point.
        """
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution
    

class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _advance_until_event(self, event_fn):
        """
        A description of the entire function, its parameters, and its return types.
        """
        raise NotImplementedError
    
    def integrate_until_event(self, t0, event_fn):
        """
        Integrate the solution until a specified event occurs.

        Parameters:
            t0: The initial time.
            event_fn: The function that defines the event.

        Returns:
            event_time: The time at which the event occurs.
            solution: The solution stacked at the initial and event times.
        """
        t0 = t0.to(self.y0.device, self.dtype)
        self._before_integrate(t0.reshape(-1))
        event_time, y1 = self._advance_until_event(event_fn)
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution
    


