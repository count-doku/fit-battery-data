import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P


class batteryModel_static():
    """
    ECM model with two RC circuits.
    The parameters are independent of temperature and state of charge (static).
    
    EXAMPLE:
        import numpy as np
        import matplotlib.pyplot as plt
        from battery_models import batteryModel_static

        bm = batteryModel_static(0.001,0.01,0.02,10,100,[2.5, 4.2],[0,1],1,
                                 soc_init=0.8)
        log = {'V': [],
               'I': [],
               'SOC': [],
               'V1': [],
               'V2': []}
        # 300 s charge -> 300 s discharge -> 100 s rest
        current = np.array([1]*300 + [-1]*200 + [0]*100)
        x = bm.state
        for u in current:
            y = bm.output_equation(u, x)
            x = bm.state_equation(u, x)
            bm.update_parameter(x[0])
            log['V'].append(y.item())
            log['I'].append(u)
            log['SOC'].append(x[0].item())
            log['V1'].append(x[1].item())
            log['V2'].append(x[2].item())

        # CC charging phase
        for u in [1]*10000:
            y, flag = bm.output_equation(u, x, flag=True)
            if flag:
                break
            x = bm.state_equation(u, x)
            bm.update_parameter(x[0])
            log['V'].append(y.item())
            log['I'].append(u)
            log['SOC'].append(x[0].item())
            log['V1'].append(x[1].item())
            log['V2'].append(x[2].item())

        # CV charging phase
        voltage_cv = bm.ocv_interp(1)  # Get CV value (Uoc(soc=1))
        for step in range(10000):
            u, num_, info_, U1_dot_, U2_dot_ = bm.calc_CV_current(voltage_cv=voltage_cv,
                                                                  i_max=1)
            y, flag = bm.output_equation(u, x, flag=True)
            x = bm.state_equation(u, x)
            if x[0] >= 0.999:
                break
            bm.update_parameter(x[0])
            log['V'].append(y.item())
            log['I'].append(u)
            log['SOC'].append(x[0].item())
            log['V1'].append(x[1].item())
            log['V2'].append(x[2].item())
            
        for key in log:
            fig, ax = plt.subplots(figsize=(200/25.4, 150/25.4))
            ax.plot(log[key])
            ax.set(title=key, xlabel='time in s')
            ax.grid()
            plt.tight_layout()
            
    
    ARGUMENTS:
        R0:
            description: Internal resistance
            unit: Ohm
            type: Scalar
            
        R1:
            description: Resistance of first RC element
            unit: Ohm
            type: Scalar
            
        R2:
            description: Resistance of second RC element
            unit: Ohm
            type: Scalar
            
        tau1:
            description: Time-constant of first RC element
            unit: s
            type: Scalar
            
        tau2:
            description: Time-constatn of second RC element
            unit: s
            type: Scalar
            
        ocv:
            description: Open-circuit-potential as a function of state-of-charge 
                            and temperature
            unit: Volt
            type: array-like
            
        grid_soc:
            description: State-of-charge grid for open-circuit-potential
            unit: 1
            type: array-like
            
        eta:
            description: Coulombic efficiency
            unit: 1
            type: Scalar
        
    OPTIONAL:
        Cref:
            description: Capacity of the cell
            default: 3600
            unit: As
            type: Scalar
        
        soc_init:
            description: Initial state-of-charge
            default: 0.01
            unit: 1
            type: Scalar
        
        u1_init:
            description: Initial C1 voltage
            default: 0
            unit: V
            type: Scalar
        
        u2_init:
            description: Initial C1 voltage
            default: 0
            unit: V
            type: Scalar
        
        ts:
            description: Sample time
            default: 1
            unit: s
            type: Scalar
            
        eoc:
            description: End-of-charge voltage
            default: 4.2
            unit: V
            type: Scalar
            
        eod:
            description: End-of-discharge voltage
            default: 2.5
            unit: V
            type: Scalar
    """
    def __init__(self, R0, R1, R2, tau1, tau2, ocv, grid_soc, eta,
                 Cref=3600, soc_init=0.01, u1_init=0, u2_init=0, ts=1,
                 eoc=4.2, eod=2.5, T_init=25):
        self.R0 = R0  # R0 f(soc, T) (Ohm)
        self.R1 = R1  # R1 ...
        self.R2 = R2  # R2 ...
        self.tau1 = tau1  # tau1 ... (s)
        self.tau2 = tau2  # tau2 ...
        self.Cref = Cref  # Cref scalar (As)
        self.ocv_map = np.array(ocv)  # ocv map ocv = f(soc) (V)
        self.grid_soc = np.array(grid_soc)  # Interpolation grid
        self.eta = eta  # Efficiency
        self.eoc = eoc  # End of charge (V)
        self.eod = eod  # End of discharge (V)
        self.ts = ts  # Sample time (s)
        
        # Initial model state (soc, U1, U2)
        self.state = np.array([[soc_init], [u1_init], [u2_init]])  
        
        # Keep track of previous states
        # for numerical derivative U1_dot and U2_dot
        self.state_previous = np.copy(self.state)
        self.state_previous_2 = np.copy(self.state)
        
        # KF parameters
        self.P = np.eye(3, 3)*1e-3
        self.Q = np.eye(3, 3)*[1e-5, 1e-9, 1e-9]
        self.R = 0.001
        self.H = np.array([[1, 1, 1]])

        # OCV interpolation
        self.ocv_interp = spi.interp1d(self.grid_soc, self.ocv_map)
        
        self.update_parameter(self.state[0])
        
        self.parameter = {'r0': self.R0,
                          'r1': self.R1,
                          'r2': self.R2,
                          'tau1': self.tau1,
                          'tau2': self.tau2}  # Dictionary of parameters
        self.unit = {'r0': ' ($\Omega$)',
                     'r1': ' ($\Omega$)',
                    'r2': ' ($\Omega$)',
                    'tau1': ' (s)',
                    'tau2': ' (s)',
                    'ocv': ' (V)'}  # Dictionary of parameter units
    
    
    def update_parameter(self, soc):
        self.ocv = self.ocv_interp(soc)
      
        
    def state_equation(self, u, x, kf=False):
        if u >= 0:
            eta = self.eta
        else:
            eta = 1
            
        A = np.array([[1, 0, 0],
                     [0, np.exp(-self.ts*self.tau1**-1), 0],
                     [0, 0, np.exp(-self.ts*self.tau2**-1)]])
        B = np.array([[self.ts*eta/self.Cref],
                      [-self.R1*(np.exp(-self.ts/self.tau1) - 1)],
                      [-self.R2*(np.exp(-self.ts/self.tau2) - 1)]])
        xp = A@x + B*u
        
        self.state_previous = np.copy(self.state)
        self.state_previous_2 = np.copy(self.state_previous)
        self.state = np.copy(xp)
        if kf: return self.state, A, B
        else: return self.state
    
    
    def output_equation(self, u, x, flag=False):
        flag_voltage_limit = False
        y = u*self.R0 + x[1] + x[2] + self.ocv
        if y >= self.eoc or y <= self.eod:
            flag_voltage_limit = True
        if flag: return y, flag_voltage_limit
        else: return y 
    
    
    def kf_predict(self, u, x):
        xp, A, B = self.stateEquation(u, x, kf=True)
        Pp = A@self.P@np.transpose(A) + self.Q
        return xp, Pp
    
    
    def kf_update(self, xp, Pp, y, z):
        K = Pp@np.transpose(self.H)/(self.H@Pp@np.transpose(self.H) + self.R)
        x = xp + K*(z - y)
        P = (np.eye(3, 3) - K@self.H)@Pp
        self.P = np.copy(P)
        return x, P
    
    
    def print_parameter(self):
        for key in self.parameter:
            print(f'{key}: {self.parameter[key]} {self.unit[key]}')
            
            
    def calc_CV_current(self, voltage_cv=4.2, i_max=1, i_min=-1):
        U1_dot = self.state[1] - self.state_previous[1]
        U2_dot = self.state[2] - self.state_previous[2]
        i = (voltage_cv - self.ocv + self.tau1*U1_dot + self.tau2*U2_dot)\
            /(self.R0 + self.R1 + self.R2)
        current = min(i_max, i.item())
        num = (voltage_cv - self.ocv + self.tau1*U1_dot + self.tau2*U2_dot)
        info = (voltage_cv - self.ocv)
        
        return (current, num, info, U1_dot, U2_dot)


class batteryModel_lookuptables():
    """
    """
    def __init__(self, R0, R1, R2, tau1, tau2, ocv, grid_soc, grid_T, eta,
                 Cref=3600, soc_init=0.01, u1_init=0, u2_init=0, ts=1,
                 eoc=4.2, eod=2.5):
        self.R0_map = R0  # R0 map f(soc, T) (Ohm)
        self.R1_map = R1  # R1 map ...
        self.R2_map = R2  # R2 map ...
        self.tau1_map = tau1  # tau1 map ... (s)
        self.tau2_map = tau2  # tau2 map ...
        self.Cref = Cref  # Cref scalar (As)
        self.ocv_map = ocv  # ocv map ocv = f(soc, T) (V)
        self.grid_soc = grid_soc  # Interpolation grid
        self.grid_T = grid_T  # ...
        self.eta = eta  # Efficiency
        self.eoc = eoc  # End of charge (V)
        self.eod = eod  # End of discharge (V)
        self.ts = ts  # Sample time (s)
        
        self.state = np.array([[soc_init], [u1_init], [u2_init]])  # Initial model state (soc, U1, U2)
        
        self.P = np.eye(3, 3)*1e-3
        self.Q = np.eye(3, 3)*[1e-5, 1e-9, 1e-9]
        self.R = 0.001
        self.H = np.array([[1, 1, 1]])
        
        self.parameter = {'r0': self.R0_map,
                          'r1': self.R1_map,
                          'r2': self.R2_map,
                          'tau1': self.tau1_map,
                          'tau2': self.tau2_map}  # Dictionary of parameter (only for vizParam)
        self.unit = {'r0': ' ($\Omega$)',
                     'r1': ' ($\Omega$)',
                    'r2': ' ($\Omega$)',
                    'tau1': ' (s)',
                    'tau2': ' (s)',
                    'ocv': ' (V)'}  # Dictionary of parameter units (only for vizParam)
        
        self.initializeParameterMaps()  # Create look-up-tables
        
    def initializeParameterMaps(self):
        self.R0_interp = spi.interp2d(self.grid_soc, self.grid_T, self.R0_map)  # Create LUT
        self.R1_interp = spi.interp2d(self.grid_soc, self.grid_T, self.R1_map)  # ...
        self.R2_interp = spi.interp2d(self.grid_soc, self.grid_T, self.R2_map)  # ...
        self.tau1_interp = spi.interp2d(self.grid_soc, self.grid_T, self.tau1_map)  # ...
        self.tau2_interp = spi.interp2d(self.grid_soc, self.grid_T, self.tau2_map)  # ...
        self.ocv_interp = spi.interp2d(self.grid_soc, self.grid_T, self.ocv_map)  # ...
    
    def updateParameter(self, soc, T):
        self.R0 = self.R0_interp(soc, T).item()
        self.R1 = self.R1_interp(soc, T).item()
        self.R2 = self.R2_interp(soc, T).item()
        self.tau1 = self.tau1_interp(soc, T).item()
        self.tau2 = self.tau2_interp(soc, T).item()
        self.ocv = self.ocv_interp(soc, T).item()
        
    def stateEquation(self, u, x, kf=False):
        if u >= 0:
            eta = self.eta
        else:
            eta = 1
            
        A = np.array([[1, 0, 0],
                     [0, np.exp(-self.ts*self.tau1**-1), 0],
                     [0, 0, np.exp(-self.ts*self.tau2**-1)]])
        B = np.array([[self.ts*eta/self.Cref],
                      [-self.R1*(np.exp(-self.ts/self.tau1) - 1)],
                      [-self.R2*(np.exp(-self.ts/self.tau2) - 1)]])
        xp = A@x + B*u
        
        self.state = np.copy(xp)
        if kf: return self.state, A, B
        else: return self.state
    
    def outputEquation(self, u, x, flag=False):
        flag_voltage_limit = False
        y = u*self.R0 + x[1] + x[2] + self.ocv
        if y >= self.eoc or y <= self.eod:
            flag_voltage_limit = True
        if flag: return y, flag_voltage_limit
        else: return y 
    
    def kfPredict(self, u, x):
        xp, A, B = self.stateEquation(u, x, kf=True)
        Pp = A@self.P@np.transpose(A) + self.Q
        return xp, Pp
    
    def kfUpdate(self, xp, Pp, y, z):
        K = Pp@np.transpose(self.H)/(self.H@Pp@np.transpose(self.H) + self.R)
        x = xp + K*(z - y)
        P = (np.eye(3, 3) - K@self.H)@Pp
        self.P = np.copy(P)
        return x, P
    
    def vizParam(self, p, extrema=False, mode='3d'):
        """
        Plots the parameter map in dependency of state of charge and temperature. Afterwards a plt.show() is necessary.

        Example:
            vizParam(['r0'])
            plt.show()

        :param p: p need to be a list. String is not supported! E.g: p = ['r0']
        :return:
        """
        
        fig = plt.figure()  # Create figure
        if mode == '3d':
            ax = fig.gca(projection='3d')  # Set axes to 3d-projection
            X = self.grid_soc
            Y = self.grid_T
            X, Y = np.meshgrid(X, Y)  # Create meshgrid for 3d-plot

            for i in p:
                surf = ax.plot_surface(X, Y, self.parameter[i])  # Plot surface to axis <ax>
                if extrema:
                    ax.plot()
            ax.set_xlabel('SOC (1)')
            ax.set_ylabel('T (°C)')
            if len(p) == 1:
                ax.set_zlabel(p[0] + self.unit[p[0]])
                ax.set_title(p[0] + ' map')
            # plt.show()
        elif mode == '2d':
            ax = fig.gca()
            ax.imshow(self.parameter[p[0]])
            

class batteryModel_functions():
    """
    """
    def __init__(self, theta_R0, theta_R1, theta_R2, theta_tau1, theta_tau2, ocv, grid_soc, grid_T, eta,
                 Cref=3600, soc_init=0.01, u1_init=0, u2_init=0, ts=1,
                 eoc=4.2, eod=2.5, T_init=25):
        self.theta_R0 = theta_R0  # R0 map f(soc, T) (Ohm)
        self.theta_R1 = theta_R1  # R1 map ...
        self.theta_R2 = theta_R2  # R2 map ...
        self.theta_tau1 = theta_tau1  # tau1 map ... (s)
        self.theta_tau2 = theta_tau2  # tau2 map ...
        self.Cref = Cref  # Cref scalar (As)
        self.ocv_map = ocv  # ocv map ocv = f(soc, T) (V)
        self.grid_soc = grid_soc  # Interpolation grid
        self.grid_T = grid_T  # ...
        self.eta = eta  # Efficiency
        self.eoc = eoc  # End of charge (V)
        self.eod = eod  # End of discharge (V)
        self.ts = ts  # Sample time (s)
        
        self.state = np.array([[soc_init], [u1_init], [u2_init]])  # Initial model state (soc, U1, U2)
        
        self.P = np.eye(3, 3)*1e-3
        self.Q = np.eye(3, 3)*[1e-5, 1e-9, 1e-9]
        self.R = 0.001
        self.H = np.array([[1, 1, 1]])
        
        self.ocv_interp = spi.interp2d(self.grid_soc, self.grid_T, self.ocv_map)  # ...
        self.updateParameter(self.state[0], T_init)
        
        self.parameter = {'r0': self.theta_R0,
                          'r1': self.theta_R1,
                          'r2': self.theta_R2,
                          'tau1': self.theta_tau1,
                          'tau2': self.theta_tau2}  # Dictionary of parameter (only for vizParam)
        self.unit = {'r0': ' ($\Omega$)',
                     'r1': ' ($\Omega$)',
                    'r2': ' ($\Omega$)',
                    'tau1': ' (s)',
                    'tau2': ' (s)',
                    'ocv': ' (V)'}  # Dictionary of parameter units (only for vizParam)
    
    def updateParameter(self, soc, T):
        self.R0 = self.surface2ndOrder(soc, T, self.theta_R0).item()
        self.R1 = self.surface2ndOrder(soc, T, self.theta_R1).item()        
        self.R2 = self.surface2ndOrder(soc, T, self.theta_R2).item()
        self.tau1 = self.surface2ndOrder(soc, T, self.theta_tau1).item()
        self.tau2 = self.surface2ndOrder(soc, T, self.theta_tau2).item()
        self.ocv = self.ocv_interp(soc, T).item()
        
    def surface2ndOrder(self, soc, T, theta):
        """
        Function: f(soc, T, theta)
        
        """
        return theta[0] + theta[1]*soc + theta[2]*T + theta[3]*soc**2 + theta[4]*T**2 + theta[5]*soc*T + theta[6]*soc**2*T**2
        
    def stateEquation(self, u, x, kf=False):
        if u >= 0:
            eta = self.eta
        else:
            eta = 1
            
        A = np.array([[1, 0, 0],
                     [0, np.exp(-self.ts*self.tau1**-1), 0],
                     [0, 0, np.exp(-self.ts*self.tau2**-1)]])
        B = np.array([[self.ts*eta/self.Cref],
                      [-self.R1*(np.exp(-self.ts/self.tau1) - 1)],
                      [-self.R2*(np.exp(-self.ts/self.tau2) - 1)]])
        xp = A@x + B*u
        
        self.state = np.copy(xp)
        if kf: return self.state, A, B
        else: return self.state
    
    def outputEquation(self, u, x, flag=False):
        flag_voltage_limit = False
        y = u*self.R0 + x[1] + x[2] + self.ocv
        if y >= self.eoc or y <= self.eod:
            flag_voltage_limit = True
        if flag: return y, flag_voltage_limit
        else: return y 
    
    def kfPredict(self, u, x):
        xp, A, B = self.stateEquation(u, x, kf=True)
        Pp = A@self.P@np.transpose(A) + self.Q
        return xp, Pp
    
    def kfUpdate(self, xp, Pp, y, z):
        K = Pp@np.transpose(self.H)/(self.H@Pp@np.transpose(self.H) + self.R)
        x = xp + K*(z - y)
        P = (np.eye(3, 3) - K@self.H)@Pp
        self.P = np.copy(P)
        return x, P
    
    def vizParam(self, p, extrema=False, mode='3d'):
        """
        Plots the parameter map in dependency of state of charge and temperature. Afterwards a plt.show() is necessary.

        Example:
            vizParam(['r0'])
            plt.show()

        :param p: p need to be a list. String is not supported! E.g: p = ['r0']
        :return:
        """
        
        fig = plt.figure()  # Create figure
        if mode == '3d':
            ax = fig.gca(projection='3d')  # Set axes to 3d-projection
            X = self.grid_soc
            Y = self.grid_T
            X, Y = np.meshgrid(X, Y)  # Create meshgrid for 3d-plot

            for i in p:
                surf = ax.plot_surface(X, Y, self.surface2ndOrder(X, Y, self.parameter[i]))  # Plot surface to axis <ax>
                if extrema:
                    ax.plot()
            ax.set_xlabel('SOC (1)')
            ax.set_ylabel('T (°C)')
            if len(p) == 1:
                ax.set_zlabel(p[0] + self.unit[p[0]])
                ax.set_title(p[0] + ' map')
            # plt.show()
        elif mode == '2d':
            ax = fig.gca()
            ax.imshow(self.parameter[p[0]])
        
             
class batteryModel_FA_thermal():
    """
    Used a polynomial for OCV curve
    """
    def __init__(self, theta_R0, theta_R1, theta_R2, theta_tau1, theta_tau2, ocv, grid_soc, grid_T, eta,
                 C_init=3600, soc_init=0.01, u1_init=0, u2_init=0, ts=1,
                 eoc=4.2, eod=2.5, T_init=298, C_th=30, R_th=24, T_amb=298,
                 SoH=1):
        self.theta_R0 = theta_R0  # theta for surface for R0 approximation
        self.theta_R1 = theta_R1
        self.theta_R2 = theta_R2
        self.theta_tau1 = theta_tau1
        self.theta_tau2 = theta_tau2
        self.C_init = C_init  # Initial capacity (As)
        self.C_ref = C_init  # Cref, scalar (As)
        self.ocv_map = ocv  # OCV array, size of grid_soc (V)
        self.grid_soc = grid_soc  # Interpolation grid
        self.grid_T = grid_T
        self.eta = eta  # Coulomb efficiency
        self.eoc = eoc  # End of charge (V)
        self.eod = eod  # End of discharge (V)
        self.ts = ts  # Sample time (s)
        self.C_th = C_th  # Heatcapacity (\DeltaQ/\deltaT in Ws/K)
        self.R_th = R_th  # Heatresistance (K/W)
        self.T = T_init  # Initial battery cell temperature (K)
        self.T_amb = T_amb  # Ambient temperature (K)
        self.SoH = SoH  # SoH
        
        self.state = np.array([[soc_init], [u1_init], [u2_init]])  # Initial model state (soc, U1, U2)
        
        self.P = np.eye(3, 3)*1e-3
        self.Q = np.eye(3, 3)*[1e-5, 1e-9, 1e-9]
        self.R = 0.001
        self.H = np.array([[1, 1, 1]])
        
        """
        OCV Polynomial replaces interpolation object
        """
        # self.ocv_interp = spi.interp2d(self.grid_soc, self.grid_T, self.ocv_map)  # Interpolation object
        deg = 9
        c, stats = P.polyfit(self.grid_soc.squeeze(), np.mean(self.ocv_map, axis=0), deg, full=True)
        # print(f'\nCoefficients: {c}\n')
        # print(f'Stats:\n\nResiduals: {stats[0]}\nRank: {stats[1]}\nSingularValues: {stats[2]}\nrcond: {stats[3]}')
        self.poly_ocv = P.Polynomial(c)
        
        self.updateParameter(self.state[0], T_init)
        
        
        
        self.parameter = {'r0': self.theta_R0,
                          'r1': self.theta_R1,
                          'r2': self.theta_R2,
                          'tau1': self.theta_tau1,
                          'tau2': self.theta_tau2}  # Dictionary of parameter (only for vizParam)
        self.unit = {'r0': ' ($\Omega$)',
                     'r1': ' ($\Omega$)',
                    'r2': ' ($\Omega$)',
                    'tau1': ' (s)',
                    'tau2': ' (s)',
                    'ocv': ' (V)'}  # Dictionary of parameter units (only for vizParam)
    
    def updateParameter(self, soc, T):
        T -= 273
        self.R0 = (-1/0.2*self.SoH+6)*self.surface2ndOrder(soc, T, self.theta_R0).item()
        self.R1 = self.surface2ndOrder(soc, T, self.theta_R1).item()        
        self.R2 = self.surface2ndOrder(soc, T, self.theta_R2).item()
        self.tau1 = self.surface2ndOrder(soc, T, self.theta_tau1).item()
        self.tau2 = self.surface2ndOrder(soc, T, self.theta_tau2).item()
        # self.ocv = self.ocv_interp(soc, T).item()
        self.ocv = self.poly_ocv(soc)
        self.C_ref = self.SoH*self.C_init
        
    def surface2ndOrder(self, soc, T, theta):
        """
        Function: f(soc, T, theta)
        
        """
        return theta[0] + theta[1]*soc + theta[2]*T + theta[3]*soc**2 + theta[4]*T**2 + theta[5]*soc*T + theta[6]*soc**2*T**2
        
    def stateEquation(self, u, x, kf=False):
        if u >= 0:
            eta = self.eta
        else:
            eta = 1
            
        A = np.array([[1, 0, 0],
                     [0, np.exp(-self.ts*self.tau1**-1), 0],
                     [0, 0, np.exp(-self.ts*self.tau2**-1)]])
        B = np.array([[self.ts*eta/self.C_ref],
                      [-self.R1*(np.exp(-self.ts/self.tau1) - 1)],
                      [-self.R2*(np.exp(-self.ts/self.tau2) - 1)]])
        xp = A@x + B*u
        
        self.state = np.copy(xp)
        
        """
        Added thermal model
        """
        self.T = (1 - self.ts/self.R_th/self.C_th)*self.T + self.T_amb*\
            self.ts/self.R_th/self.C_th + self.R0*self.ts/self.C_th*u**2
                    
        if kf: return self.state, A, B
        else: return self.state, self.T
    
    def outputEquation(self, u, x, flag=False):
        flag_voltage_limit = False
        y = u*self.R0 + x[1] + x[2] + self.ocv
        if y >= self.eoc or y <= self.eod:
            flag_voltage_limit = True
        if flag: return y, flag_voltage_limit
        else: return y 
    
    def kfPredict(self, u, x):
        xp, A, B = self.stateEquation(u, x, kf=True)
        Pp = A@self.P@np.transpose(A) + self.Q
        return xp, Pp
    
    def kfUpdate(self, xp, Pp, y, z):
        K = Pp@np.transpose(self.H)/(self.H@Pp@np.transpose(self.H) + self.R)
        x = xp + K*(z - y)
        P = (np.eye(3, 3) - K@self.H)@Pp
        self.P = np.copy(P)
        return x, P
    
    def vizParam(self, p, extrema=False, mode='3d'):
        """
        Plots the parameter map in dependency of state of charge and temperature. Afterwards a plt.show() is necessary.

        Example:
            vizParam(['r0'])
            plt.show()

        :param p: p need to be a list. String is not supported! E.g: p = ['r0']
        :return:
        """
        
        fig = plt.figure()  # Create figure
        if mode == '3d':
            ax = fig.gca(projection='3d')  # Set axes to 3d-projection
            X = self.grid_soc
            Y = self.grid_T
            X, Y = np.meshgrid(X, Y)  # Create meshgrid for 3d-plot

            for i in p:
                surf = ax.plot_surface(X, Y, self.surface2ndOrder(X, Y, self.parameter[i]))  # Plot surface to axis <ax>
                if extrema:
                    ax.plot()
            ax.set_xlabel('SOC (1)')
            ax.set_ylabel('T (°C)')
            if len(p) == 1:
                ax.set_zlabel(p[0] + self.unit[p[0]])
                ax.set_title(p[0] + ' map')
            # plt.show()
        elif mode == '2d':
            ax = fig.gca()
            ax.imshow(self.parameter[p[0]])