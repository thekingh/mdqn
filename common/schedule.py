def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

def PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        ''' PWS
        endpoints: [(int, int)]
            list of pairs '(time, value)'
        interpolation: lambda float, float, float: float
        outside_value = float
        '''

        indices = [e[0] for e in endpoints]
        assert indices == sorted(indices)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints     = endpoints


    def value(self, t):
        # See schedule value

        for(l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t <= r_t:
                alpha = float(t-l_t)/(r_t-l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps=schedule_timesteps
        self.final_p   = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min( float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)