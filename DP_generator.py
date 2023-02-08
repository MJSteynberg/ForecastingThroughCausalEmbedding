
import numpy
import scipy, scipy.integrate, scipy.optimize
import pylab


def DoublePendulumDerivArray(vars,t,l1=1,l2=1,m1=1,m2=1,g=1):
    """ variables are (theta1,theta2,omega1,omega2)
    parameters are the following: l1,l2,m1,m2,g
    if any are unspecified, they are taken to be equal to 1.
    
    returns array([dt_theta1,dt_theta2,dt_omega1,dt_omega2])
    
    Sample usage:
    DoublePendulumDerivArray((1,0,0,0),0,l1=2,l2=2,m1=1,m2=2,g=9.8)
    """
    (theta1,theta2,omega1,omega2)=vars        # Unpacks the variables
    dt_theta1=omega1
    dt_theta2=omega2
    s1=numpy.sin(theta1)
    s2=numpy.sin(theta2)
    cd=numpy.cos(theta2-theta1)
    sd=numpy.sin(theta2-theta1)
    mu = m2/(m1+m2)
    lamda = l1/l2
    g1 = g/l1
    g2 = g/l2
    
    den=1-mu*cd**2
    
    num1 = mu*g1*s2*cd+mu*omega1**2*sd*cd-g1*s1+mu/lamda*omega2**2*sd
    num2 = g2*s1*cd-mu*omega2**2*sd*cd-g2*s2-lamda*omega1**2*sd
    
    dt_omega1=num1/den
    dt_omega2=num2/den
    return numpy.array([dt_theta1,dt_theta2,dt_omega1,dt_omega2])
    

def DoublePendulumTrajectory(initial_theta1=0,initial_theta2=0,initial_omega1=0,initial_omega2=0,
    timestep=0.1,numstep=500,l1=1,l2=1,m1=1,m2=1,g=1) :
    """ Runs ODEint for the double pendulum 
    
    sample usage:
    DoublePendulumTrajectory(initial_theta1=0.1,l1=2,l2=2,m1=1,m2=2,g=9.8)
    
    Returns a dictionary which has the following keys:
    "times", "theta1", "theta2", "omega1", "omega2", "parameters"
    
    The values associated with the first five keys are a time series.
    The parameters key returns a dictionary which stores the parameters, plus the energy of the system
    """
    p={"l1":l1,"l2":l2,"m1":m1,"m2":m2,"g":g}                # Stores parameters in a dictionary
    # Create a tuple containing (l1,l2,m1,m2,g)
    partuple=(l1,l2,m1,m2,g)        # Converts parameters to a tuple in the right order
    # Create a tuple containing the inital values
    initial=(initial_theta1,initial_theta2,initial_omega1,initial_omega2)
    # Create the list of times
    tvals=numpy.arange(numstep)*timestep
    # Run odeint
    traj=scipy.integrate.odeint(DoublePendulumDerivArray,initial,tvals,partuple)
    # Store the results of odeint in a dictionary
    data={}
    data["times"]=tvals
    data["theta1"]=traj[:,0]
    data["theta2"]=traj[:,1]
    data["omega1"]=traj[:,2]
    data["omega2"]=traj[:,3]
    data["parameters"]=p
    # Return the dictionary
    return data


def ShowDoublePendulumAttractor(data, coord):
    """ Call with a dictionary containing 
    "times", "theta1", "theta2", "omega1", "omega2","parameters"
    """
    parameters=data["parameters"]
    l1=parameters["l1"]
    l2=parameters["l2"]
    theta1=data["theta1"]
    theta2=data["theta2"]
    x1=l1*numpy.sin(theta1)
    y1=-l1*numpy.cos(theta1)
    x2=l2*numpy.sin(theta2)+x1
    y2=l2*numpy.sin(theta2)+y1
    fig = pylab.figure()
    ax0 = fig.add_subplot( projection='3d')
    i=8
    if coord == 'x1':
    
        ax0.plot(x1[1000:4000], x1[1008:8008],x1[1016:8016],lw=0.2,color='blue')
    elif coord == 'x2':
        ax0.plot(x2[1000:4000], x2[1008:8008],x2[1016:8016],lw=0.2,color='blue')
    elif coord == 'y1':
        ax0.plot(y1[1000:4000], y1[1008:8008],y1[1016:8016],lw=0.2,color='blue')
    elif coord == 'y2':
        ax0.plot(y2[1000:8000], y2[1000+i:8000+i],y2[1000+2*i:8000+2*i],lw=0.2,color='blue')
        
    pylab.show()
    numpy.savetxt('dp_data.txt', list(zip(x2, y2)))
    return    


data = DoublePendulumTrajectory(initial_theta1=numpy.pi/4.5,initial_theta2=0,initial_omega1=-1.5,initial_omega2=0.3,
            timestep=0.05,numstep=10000,l1=1+1*.15,l2=1,m1=0.25*3,m2=2.5,g=9.81)

ShowDoublePendulumAttractor(data, 'y2')
        