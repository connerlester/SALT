#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import importlib
def reload(module_name):
    importlib.reload(module_name)

#blah

### HERE IS THE MAIN ROUTINE NEEDED TO RUN THE MODEL: will explain functions below

'''
import *THE NAME OF THIS .py FILE* as salt # import this .py module 

salt.action() # init transport (must do)

RUN_TIME=100 #sqrt{d/g'} run time 


while salt.T< RUN_TIME: ## time loop

    dead = [] ##init if ejections = 1 (to remove "dead" grains)

    for p in salt.pars: # iterate thru particle list

        if not p.new:

            salt.trajectory_integration() ## integrate grains trajectory thru spacetime
            salt.save_trajectory() ## save traj info if you want
        
        else: # if youre a "new" (ejected) particle then you dont get updated until next time step
            p.new = 0

        # this gives the info on if a particle rebounded off the surface or didnt
        if p.dead: dead.append(1) ## if grain p didnt rebound 
        else: dead.append(0) # still alive

    salt.T+=salt.dt # advance time forward after iterated thru pars

    # delete all the "dead" particle from the pars list
    salt.pars = list(np.delete(salt.pars,np.where(np.array(dead)==1)))

    ## ... and repeat

'''

## note that I am leaving out the code where I did the data analysis,
# where I save particle trajectory distributions and profiles with Z and did fourier transforms etc to get some of the 
# plots in my slides. Adding all that extra code made this file more cluttered than it already is so I took it out...



############################################################################################
## UNIT RESCALINGS: ( g' == (1 - 1/s)g ) = buoyancy reduced gravity
#  velocity -> sqrt[ g' d ] (grains fall velocity) , length -> d (grain diameter), 
#  time -> sqrt[ d/g' ] , acceleration -> g'
#  PARTICLE force -> rho_p V_p g' (=mg') , FLUID stress -> rho_f g' d 

### TRANSPORT PARAMETERS:
# There are 3 fundemental (dimensionless) parameters in this model: 
## 1.) Density ratio between grains and fluid (s)
## 2.) Galileo number (Ga)--the grains fall velocity relative to the "viscous transition velocity" (nu/d)
    # aka the grain Reynolds Number
## 3.) Shields Number--turbulent shear stress over the normal force needed to remove a grain from a 1d deep pocket in the bed

## GLOBAL VARS
T=0 #init run time
dt = .01 # time step
dt0 = dt/1000 # bed impact time step
TSAVE = int(1/dt) # save trajectory time
Zmax = 800 # domain height
Zmin = 0.5 # min elevation (grain radius)
NZ = 85 # number of Z bins
zfact = np.log(Zmax/Zmin)/NZ # factor
Z = lambda i: Zmin*np.exp(i*zfact) ## fxn to get Z for a given bin index
dZ = lambda z: z*(np.exp(zfact)-1) ## fxn to get dZ at a given Z
#### elevation Z is log-binned ####
def Zindex(z):
    ## returns index for a given Z value  
    if z>=Zmin:return np.log(z/Zmin)/zfact
    else:return 0 ## any z<Zmin gets index 0

## STATISTICS:
mod, Lhop0 ,Lhop, dLhop= [], [],[],[]


pars = [] ## particle list


## initialize model run with this class and arbuments in __init__ ( ex: salt.action(A=1,k=10) for transport over a ripple of amplitude 1d and wavenumber k=L/lambda=10)
class action:

    def __init__(self,
        s = 2000, ## = rho_p/rho_f 
        Ga = 22, ## = sqrt[sg'd]/(nu/d)
        u_ = 3, ## = u*/uth -- ratio of shear velocity u_* at z-->infinity and at transport threshold (z-->surface)
        uth = 2.83, ## rough transport threshold for s=2000
        L = 1000, ## domain length
        A = 0, k = 3, ## ripples: amplitude and wavenumber
        delx=1, ## surface gradient step 
        NOWIND=0, NODRAG=0, NOVDRAG=0, ELASTIC=0, EJECTIONS=1, FLUIDGRAIN_FEEDBACK=0,DEM_FLOW=1, ## all flags
        N0 =0 ## number of pars in transport at t=0 (if N0=0 it is set to default value for a given u*/uth)
        ):  
        
        self.s,self.Ga = s,Ga
        self.u_ = u_
        self.uth = uth/(1-1/s)**.5
        self.ustar = u_*self.uth ## = shear velocity
        self.shields = (self.ustar)**2/s  ## = rho_f u*^2 (shear stress) / rho_p g'd
        self.shieldsth = (self.uth)**2/s  ## " " threshold
        self.NOWIND,self.NODRAG,self.NOVDRAG,self.ELASTIC,self.EJECTIONS,self.FLUIDGRAIN_FEEDBACK,self.DEM_FLOW=NOWIND,NODRAG,NOVDRAG,ELASTIC,EJECTIONS,FLUIDGRAIN_FEEDBACK,DEM_FLOW
        self.A,self.k ,self.delx= A,k,delx
        self.NSS = int((1/.63)*(self.shields-self.shieldsth)*L)
        self.N0 = self.NSS if N0==0 else N0
        self.L = L
        self.set() ## setting params above---see below

    def set(self):
        m=self

        ## MAKING EVERYING GLOBAL---proabably a better way to do this
        global s,Ga,ustar,uth,shields,shieldsth,A,k,NSS,L
        global NOWIND,NODRAG,NOVDRAG,ELASTIC,EJECTIONS,FLUIDGRAIN_FEEDBACK,DEM_FLOW
        global ZZ, dZZ, XX, Zsurf, F, uf, delx
        s,Ga,ustar,uth,shields,shieldsth,A,k,NSS,L,delx = m.s,m.Ga,m.ustar,m.uth,m.shields,m.shieldsth,m.A,m.k,m.NSS,m.L,m.delx
        NOWIND,NODRAG,NOVDRAG,ELASTIC,EJECTIONS,FLUIDGRAIN_FEEDBACK,DEM_FLOW = m.NOWIND,m.NODRAG,m.NOVDRAG,m.ELASTIC,m.EJECTIONS,m.FLUIDGRAIN_FEEDBACK,m.DEM_FLOW

        # init fields 
        ZZ = Z(np.arange(NZ)) # Z-array (vertical)
        dZZ = dZ(ZZ) # dZ-array
        XX = np.arange(L) # X-array (horizontal) -- note dX=1d
        field = fields()
        Zsurf, F, uf = field.Zsurf, field.F, field.uf

        #init pars
        for i in range(m.N0): 
            x = np.random.randint(0,L) # random x-position
            z = 50+np.random.exponential(20)  + Zsurf[int(x)] + A # exponential decaying particle consentration with Z
            um = ustar*np.sqrt(z)/2 ## giving them some initial x-velocity
            vm = 5 # " " and z-velocity
            # init particle 
            particle( u = np.random.normal(um), ## random around given value
                      v = np.random.normal(vm)*np.random.randint(-1,2), # +- random around given value
                      x = x, 
                      z = z)


 ## flow field and surface elevation field
class fields:
    def __init__(self):
        self.Zsurf = A*np.cos(2*np.pi*k*XX/L) + Zmin ## surface at grain raduis above z=0 (Zsurf=0+Zmin by default)
        self.F = np.zeros(NZ) ## fluid force(Z) on grains to integrate wind
        # -- I couldnt get integration to work so I dont use this (I explain below)
        if DEM_FLOW: self.uf = DEM_flow(ustar/uth)
        else: self.uf = fluid_integration()
        if NOWIND: self.uf*=0

## mixing length model for flow profile
def fluid_integration():
        ## the particle born shear stress is the sum of fluid forces/area above elevation Z
        taup = np.array([np.sum((F)[y:]) for y in range(NZ)]) # sum
        tauf = ustar**2 - taup  ## RESCALED fluid shear stress
        ## turbulent mixing length model
        dd=1
        RR = (ZZ+dd)/26
        l = (0.4*(ZZ+dd)* ( 1- np.exp(-( RR*Ga*ustar/np.sqrt(s)))) )
        ## wind gradient
        a,b,c= l**2, np.sqrt(s)/Ga, -tauf
        dufdz = (-b + np.sqrt(b**2 - 4*a*c))/(2*a) 
        ## integration of wind uf(z) plus uth (surface must be at uth)
        uf = np.array([np.sum(dufdz[:y]*dZZ[:y]) for y in range(NZ) ])+uth
        return uf 

# drag coeff for sphere
def Cdrag( u_rel ):
    Ru = Ga/np.sqrt(s) 
    C = (np.sqrt(.5*u_rel) + np.sqrt(24/Ru))**2 
    return C   



## grain functions
class particle: ## all particles and their properties go here

    def __init__(self, u,v,x,z, new=0 ):
        p = self
        p.u,p.v,p.x,p.z,p.ax,p.az= u,v,x,z,0,0 ## set initial x,u,a info
        p.up,p.vp,p.xp,p.zp= [u],[v],[x],[z] ## saving x,u lists
        p.zup, p.xup , p.l0 = -100, 0, 0
        p.xcross = 0
        p.dt=dt ## time step for impacts 
        p.Nrebs,p.dead= 0,0 ## for rebounds
        p.new = new ## for ejections
        pars.append(p)



     ## this is the main function for running the model
    def trajectory_integration(self):
        p = self
        
        p.ax,p.az = p.forces() ## find forces/mass on particle p

        ## Verlet Integration
        p.x += p.u*dt + .5*p.ax*dt**2 # step position forward
        p.z += p.v*dt + .5*p.az*dt**2
        ## EVERY TIME YOU STEP p.x CHECK BOUNDARY CONDITIONS!
        p.xbounds() 

        ## check for bed impact
        if p.Zrel() < 0: # if pars Z-position relative to Zsurf is < 0...

            p.bedimpact() ## impact! (see below)
            
            if not p.dead: ## if par sucessfully rebounded (not dead)
                p.x += p.u*p.dt + .5*p.ax*p.dt**2 ## advance the rest of the time step (see bedimpact fxn below)
                p.z += p.v*p.dt + .5*p.az*p.dt**2
                p.xbounds() # check bounds

        
        if not p.dead: 
            ax1,az1 = p.forces() ## calc forces again 

            p.u += (ax1+p.ax)*p.dt/2 ## ... advance velocties (verlet)
            p.v += (az1+p.az)*p.dt/2

            p.dt = dt ## reset partice dt (from impact)

        # FLAG: this never happened but just incase
        if p.z>Zmax: raise ValueError('Zmax exceeded !!!')




    ## second major function for model run
    def bedimpact(self):
        p=self

        p.dt = dt ## init par dt
        zrel0 = p.Zrel() # init flag
        zrel=zrel0 ## init zrel

        ## impact scheme--needs to be updated: instead of stepping back with const dt0, step back with dt/2, then forward dt/4 etc. (iterate until min error is reached--less steps)
        while zrel < 0: ## IMACT !!! 

            p.x = p.x - (p.u*dt0 + .5*p.ax*dt0**2) ## step back the way you came
            p.z = p.z - (p.v*dt0 + .5*p.az*dt0**2) # but with MUCH smaller time step dt0

            p.dt-=dt0 ## keep track of how many times you step back

            p.xbounds() ## once again check boundary after changing p.x

            zrel=p.Zrel() ## check Zrel until you get back to surface

            ## IMPORTANT FLAG!! 
            if zrel<zrel0: ## THIS SHOULD NOT HAPPEN!!! zrel should be getting less negative
                raise ValueError('Zrel error !!!',"zrel=",zrel,"zrel0=",zrel0)

        ## after back at Zrel~0
        if p.dt <0: p.dt=0 ## correction if over step
        epsilon = 1e-4 ## something to fix weird bugs
        p.z = p.Zrel(1)+epsilon # place particle at surface

        ax1,az1 = p.forces() ## get forces at surface
        p.u += (ax1+p.ax)*p.dt/2 ## advance velocties TO the surface from z(t)
        p.v += (az1+p.az)*p.dt/2

        p.rebound() ## calculate rebound/ejection info (see below)

        p.ax,p.az = p.forces() ## get forces
        p.dt = dt-p.dt ## set rest of time step... finish time step in traj._integration fxn 



    ## 3rd major function
    def rebound( self ):
        p = self
        u , v , x  = p.u , p.v , p.x
        ## on impact change coords to local slope coords to calculate rebound:

        x1 = int((L+x+delx)%L) ## x+-dx for gradient (includes BC's for x)
        x_1 = int((L+x-delx)%L)
        
        dzdx = (Zsurf[x1] - Zsurf[x_1])/(2*delx) # ~ surface slope
        norm = np.sqrt(dzdx**2 +1) ## normalizing factor

        vit = (-u*dzdx + v)/norm ##transform velocties to local coord system
        uit = (u + v*dzdx)/norm
        
        ## MAJOR FLAG!!! 
        if vit>0: ## THIS SHOULD NOT HAPPEN
            raise ValueError('v_imp>0 !!!')
            

        p.v = vit
        p.u = uit
            
        _vi_ = np.sqrt(p.v**2 + p.u**2) ## mag of impact velocity

        if ELASTIC: ## if elastic--just flip sign of p.v
            p.u =  u
            p.v = -v
        else:  
            p.e_coeffs() ## else get Coeff Of Rest's (COR's)
            
            v0t = -p.ez*p.v ## COR corrected velocity rebound in z-dir
            vup = p.e*_vi_ # ... rebound magnitude
            thetaup = np.abs(np.arcsin(v0t/vup)) #rebound angle
            u0t = vup*np.cos(thetaup) # x-dir rebound
            
            p.u = (u0t - v0t*dzdx)/norm  ##transform back to normal x,z coords
            p.v = (u0t*dzdx + v0t)/norm

        # FLAG: if ejections are on...
        if EJECTIONS: 
            p.rebound_ejection(_vi_)
        elif .5*_vi_**2  < 10:
            p.dead = 1  
            x = np.random.randint(0,L) # random x-position
            z = np.random.randint(50,100)
            um = ustar*np.sqrt(z) 
            vm = 5 
            particle( u = np.random.normal(um), ## random around given value
                      v = np.random.normal(vm)*np.random.randint(-1,2), # +- random around given value
                      x = x, 
                      z = z,
                      new=1) 
        else:
            p.Nrebs += 1




    # if ejections=1: 4th most important function for model run
    def rebound_ejection(self,Vimp):
        p=self
        ## need to input impact velocity from rebound fxn

        ## const params (arb!)
        ErebLimit, NejLimit, EejLimit, Pmax = 10, 30, 40, 0.8

        Eimp = .5*Vimp**2 # impact energy

        #rebound probability!
        Preb = Pmax*( 1 - np.exp(- Vimp/p.Nrebs/10 )) if p.Nrebs>0 else Pmax
        ## as Vimp increases Preb increases
        ## but as number of rebounds increases, Preb decreases

        ## rebound?
        if Preb < np.random.rand() or Eimp < ErebLimit: ## note Eimp can be too small
            p.dead = 1 ## no-rebound (=dead particle) flag
            p.e = 0 ## COR=0 if no reb... keeps all Eimp in bed for ejections
        else:
            p.Nrebs+=1 ## if rebound add 1


        ## NEEDS UPDATE!!! THIS IS WEIRD!!!
        r = .5*np.random.rand()
        factor = r*(NSS/len(pars)-1) if not FLUIDGRAIN_FEEDBACK else .1/Eimp**(r)
        Ebed = (1-p.e**2)*Eimp*factor if factor > 0 else 0 ## bed energy for ejections!
        ## len(pars)^r is a feedback that keeps from exponential growth of grains in transport
        # and gets me to a steady state---not a super great/realistic feedback probably but I could argue for why its no so unreasonable

        while Ebed > EejLimit: ## If there is enough energy to eject a grain (> arb limit):
            Eej = 2*Ebed ##init statement

            while Eej>Ebed: ## Eej must always be < Ebed:

                ## pull ejection velocities randomly from distributions
                # note the means are around sqrt[Ebed]... which gets smaller after every ejection (see below)
                unew = np.random.normal(np.sqrt(Ebed),.5) 
                vnew = np.random.lognormal(np.log(np.sqrt(Ebed)),.5) 

                Eej = .5*(unew**2+vnew**2) ## ejection energy
            ## if Eej was pulled to be < Ebed init new grain
            
            particle(unew,vnew,p.x,p.z,new=1 ) ## init with velocity info above

            Ebed-=Eej ##REDUCE ENERGY IN BED FOR NEXT EJECTION!!!

 
    ## get forces/mass on grain p
    def forces(self):
        p = self
        u,v,x,z = p.u,p.v,p.x,p.z
     
        ufz = p.fluid_interp() # interpolate wind speed at particles zrel position

        u_rel = np.sqrt( (ufz-u)**2 + v**2) ## grain-wind relative velocity mag
        Cd = Cdrag(u_rel) ## drag coefficient 
        
        # FLAGS
        if NODRAG: Cd = 0
        if NOVDRAG: v = 0

        ax = (3/4)*(Cd/s)*(ufz-u) ## drag force in x
        az = (3/4)*(Cd/s)*(-v) - 1 ## drag force in z and -g' (rescaled = -1)

        return ax,az

    # boundary conditions
    def xbounds(self):
        if self.x < 0:
            self.x += L ## add or subtract domain length L for situation " ... "
            self.xcross-=1
        elif self.x > L:
            self.x -= L
            self.xcross += 1

    # particle Z-pos relative to Zsurf (interpolation)
    def Zrel(self,get=0):
        # "get" just returns actual surface elevation (for impacts)
        a,b = int(np.floor(self.x)),int(np.ceil(self.x)) # bounds for interpolation
        d_ab = b-a # step size
        if a<0: a+=L # BC's
        if b>=L: b-=L

        if a==b: 
            if get: return Zsurf[a]
            else: return self.z - Zsurf[a] # no interp if a=b
        else:
            zx = Zsurf[a]+ (self.x-a)*(Zsurf[b]-Zsurf[a])/(d_ab) #linear interpolation of surface points
            if get: return zx
            else: return self.z - zx ## return relative z-pos


    ## COR's from experiment, Beladjine et al 2007 and Pahtz et al 2021
    def e_coeffs(self):
        p=self
        _vi_ = np.sqrt(p.v**2 +p.u**2) ## vel mag
        sin0 = -p.v/_vi_
        p.e = 0.87 - 0.72*sin0 ## COR mag
        p.ez = 0.87/(sin0)**.5 - 0.72  ## COR z-dir

    ## interp fluid velocity at particles zrel
    def fluid_interp(self):
        p=self
        zrel = p.Zrel()
        zi = Zindex(zrel) ## get Z-index
        a,b = int(np.floor(zi)),int(np.ceil(zi)) # bounds for interpolation
        if a==b:
            return uf[a]
        else:
            ufz = uf[a]+ (p.z - ZZ[a])*(uf[b]-uf[a])/(Z(b)-Z(a)) #linear interpolation uf(zrel)
            return ufz

    ##########################################################################################
    ## I couldnt get the fluid integration/grain-fluid feedback to work but here are the functions
    def fluid_force(self):
        p=self

        zrel = p.Zrel()
        ufz = p.fluid_interp() # wind speed at zrel
        u_rel = np.sqrt( (ufz-p.u)**2 + p.v**2)
        Cd = Cdrag(u_rel)

        zi = int(round(Zindex(zrel))) # Z-index

        Fdragx = .75*(Cd)*(ufz-p.u) ## drag force/area

        F[zi] += (np.pi/6)*(Fdragx)/L  ## Particle force (per unit area) on fluid

    ## save par info !!  
    def save_trajectory(self):
        p=self 
        # save info for a full time unit sqrt[d/g']
        p.xp.append(p.x)
        p.zp.append(p.z)
        p.up.append(p.u)
        p.vp.append(p.v)
        p.xp = p.xp[-TSAVE:]
        p.zp = p.zp[-TSAVE:]
        p.up = p.up[-TSAVE:]
        p.vp = p.vp[-TSAVE:]



### FLOW PROFILES FOR STEADY STATE TRANSPORT! OBTAINED FROM GRAIN SCALE DEM MODEL
# ALL FOR s=2000:
u_DEM = np.array([1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5.]) #u*/uth 

Z_DEM = np.array([-4.00000e+00, -3.90000e+00, -3.80000e+00, -3.70000e+00, ## elevations
       -3.60000e+00, -3.50000e+00, -3.40000e+00, -3.30000e+00,
       -3.20000e+00, -3.10000e+00, -3.00000e+00, -2.90000e+00,
       -2.80000e+00, -2.70000e+00, -2.60000e+00, -2.50000e+00,
       -2.40000e+00, -2.30000e+00, -2.20000e+00, -2.10000e+00,
       -2.00000e+00, -1.90000e+00, -1.80000e+00, -1.70000e+00,
       -1.60000e+00, -1.50000e+00, -1.40000e+00, -1.30000e+00,
       -1.20000e+00, -1.10000e+00, -1.00000e+00, -9.00000e-01,
       -8.00000e-01, -7.00000e-01, -6.00000e-01, -5.00000e-01,
       -4.00000e-01, -3.00000e-01, -2.00000e-01, -1.00000e-01,
        0.00000e+00,  1.00000e-01,  2.00000e-01,  3.00000e-01,
        4.00000e-01,  5.00000e-01,  6.00000e-01,  7.20000e-01,
        8.64000e-01,  1.03680e+00,  1.24416e+00,  1.49299e+00,
        1.79159e+00,  2.14991e+00,  2.57989e+00,  3.09587e+00,
        3.71504e+00,  4.45805e+00,  5.34966e+00,  6.41960e+00,
        7.70350e+00,  9.24420e+00,  1.10931e+01,  1.33117e+01,
        1.59740e+01,  1.91688e+01,  2.30026e+01,  2.76031e+01,
        3.31237e+01,  3.97484e+01,  4.76981e+01,  5.72377e+01,
        6.86853e+01,  8.24223e+01,  9.89070e+01,  1.18688e+02,
        1.42426e+02,  1.70911e+02,  2.05093e+02,  2.46112e+02,
        2.95334e+02,  3.54401e+02,  4.25281e+02,  5.10337e+02,
        6.12405e+02,  7.34886e+02,  8.81863e+02])
uf_DEM = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ##flow profiles for each u* in order
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.65845358e-04, 6.70922250e-03, 2.26353588e-02, 5.29640160e-02,
        1.04467752e-01, 1.85438156e-01, 3.02498700e-01, 4.62403605e-01,
        6.69058695e-01, 9.24637410e-01, 1.22647389e+00, 1.56994108e+00,
        1.94734281e+00, 2.35159416e+00, 2.77556778e+00, 3.30235106e+00,
        3.95012956e+00, 4.73924535e+00, 5.69504955e+00, 6.84871320e+00,
        8.23721025e+00, 9.90778755e+00, 1.19105786e+01, 1.42879908e+01,
        1.70452881e+01, 2.01076736e+01, 2.33139220e+01, 2.64868897e+01,
        2.95086081e+01, 3.23352263e+01, 3.49723901e+01, 3.74462912e+01,
        3.97893189e+01, 4.20316977e+01, 4.42002135e+01, 4.63163460e+01,
        4.83980940e+01, 5.04586170e+01, 5.25059805e+01, 5.45474010e+01,
        5.65858500e+01, 5.86209030e+01, 6.06504375e+01, 6.26723310e+01,
        6.46836120e+01, 6.66825825e+01, 6.86709405e+01, 7.06503840e+01,
        7.26217620e+01, 7.45876215e+01, 7.65479625e+01, 7.85036340e+01,
        8.04563340e+01, 8.24056380e+01, 8.32593075e+01],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.36586636e-03, 1.08027074e-02, 2.84739233e-02, 5.97487655e-02,
        1.10139140e-01, 1.86497173e-01, 2.95771122e-01, 4.43807992e-01,
        6.33852957e-01, 8.66305337e-01, 1.13965659e+00, 1.44936746e+00,
        1.78958469e+00, 2.15399994e+00, 2.53702731e+00, 3.01448614e+00,
        3.60499994e+00, 4.32918879e+00, 5.21122228e+00, 6.28323930e+00,
        7.58335714e+00, 9.16288429e+00, 1.10816616e+01, 1.34022605e+01,
        1.61610291e+01, 1.93204816e+01, 2.27304468e+01, 2.61930288e+01,
        2.95591941e+01, 3.27641529e+01, 3.58047768e+01, 3.87055254e+01,
        4.14994477e+01, 4.42196497e+01, 4.68938333e+01, 4.95456618e+01,
        5.21898112e+01, 5.48376011e+01, 5.74956299e+01, 6.01674244e+01,
        6.28517332e+01, 6.55462809e+01, 6.82459481e+01, 7.09450464e+01,
        7.36373188e+01, 7.63176458e+01, 7.89854585e+01, 8.16401881e+01,
        8.42846788e+01, 8.69200681e+01, 8.95486316e+01, 9.21709379e+01,
        9.47881247e+01, 9.74013297e+01, 9.85452469e+01],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.94402950e-04,
        7.23604483e-04, 1.63419315e-03, 2.98987253e-03, 4.85987645e-03,
        7.33321882e-03, 1.05676541e-02, 1.47226703e-02, 2.00025512e-02,
        2.66511127e-02, 3.51385105e-02, 4.60165717e-02, 5.99437401e-02,
        7.80725033e-02, 1.02181378e-01, 1.35111312e-01, 1.80790840e-01,
        2.43710824e-01, 3.29167564e-01, 4.43333265e-01, 5.90590974e-01,
        7.73163924e-01, 9.91012655e-01, 1.24343250e+00, 1.52659004e+00,
        1.83586970e+00, 2.16650087e+00, 2.51349447e+00, 2.94616890e+00,
        3.48266235e+00, 4.14258581e+00, 4.94982757e+00, 5.93480825e+00,
        7.13809570e+00, 8.60846012e+00, 1.04128381e+01, 1.26295185e+01,
        1.53262906e+01, 1.85110467e+01, 2.20661773e+01, 2.57804224e+01,
        2.94718362e+01, 3.30463453e+01, 3.64837255e+01, 3.98045534e+01,
        4.30444851e+01, 4.62355127e+01, 4.94073733e+01, 5.25790930e+01,
        5.57668791e+01, 5.89801039e+01, 6.22258844e+01, 6.55052071e+01,
        6.88116597e+01, 7.21384772e+01, 7.54786130e+01, 7.88208628e+01,
        8.21567706e+01, 8.54814037e+01, 8.87891247e+01, 9.20813429e+01,
        9.53601724e+01, 9.86270225e+01, 1.01884712e+02, 1.05135355e+02,
        1.08378951e+02, 1.11616909e+02, 1.13034705e+02],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.31304642e-03, 5.08787022e-03, 1.22974254e-02,
        2.38638618e-02, 4.15968399e-02, 6.84976596e-02, 1.08451260e-01,
        1.65990537e-01, 2.46032559e-01, 3.52871568e-01, 4.89426426e-01,
        6.57677001e-01, 8.57337180e-01, 1.08598986e+00, 1.34086815e+00,
        1.61863548e+00, 1.91518269e+00, 2.22606102e+00, 2.61418137e+00,
        3.09552192e+00, 3.68847201e+00, 4.41528393e+00, 5.30536704e+00,
        6.39780081e+00, 7.74294792e+00, 9.41252340e+00, 1.14973278e+01,
        1.40967111e+01, 1.72686600e+01, 2.09390568e+01, 2.48924253e+01,
        2.89119309e+01, 3.28807512e+01, 3.67568607e+01, 4.05524850e+01,
        4.42954713e+01, 4.80078936e+01, 5.17195518e+01, 5.54509068e+01,
        5.92197876e+01, 6.30358728e+01, 6.69029829e+01, 7.08197595e+01,
        7.47759297e+01, 7.87633431e+01, 8.27712174e+01, 8.67873270e+01,
        9.08022480e+01, 9.48069810e+01, 9.87938850e+01, 1.02762111e+02,
        1.06714206e+02, 1.10651868e+02, 1.14577644e+02, 1.18494930e+02,
        1.22403726e+02, 1.26305730e+02, 1.28013918e+02],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.13703030e-04, 2.43555576e-03, 7.22573075e-03, 1.53198306e-02,
        2.78892297e-02, 4.66407175e-02, 7.41603291e-02, 1.13997531e-01,
        1.70186236e-01, 2.47129577e-01, 3.48268451e-01, 4.75656070e-01,
        6.30343378e-01, 8.11890329e-01, 1.01887845e+00, 1.24874494e+00,
        1.49840851e+00, 1.76436268e+00, 2.04277321e+00, 2.38980298e+00,
        2.82015327e+00, 3.35035309e+00, 4.00113324e+00, 4.79822097e+00,
        5.77748541e+00, 6.98656669e+00, 8.49505749e+00, 1.04008604e+01,
        1.28216056e+01, 1.58608974e+01, 1.95141665e+01, 2.35926802e+01,
        2.78582378e+01, 3.21469401e+01, 3.63982932e+01, 4.06058404e+01,
        4.47949117e+01, 4.89931216e+01, 5.32289787e+01, 5.75206610e+01,
        6.18849557e+01, 6.63279221e+01, 7.08441963e+01, 7.54265269e+01,
        8.00651793e+01, 8.47417770e+01, 8.94390359e+01, 9.41449367e+01,
        9.88464670e+01, 1.03533793e+02, 1.08201450e+02, 1.12847255e+02,
        1.17474186e+02, 1.22083237e+02, 1.26679375e+02, 1.31263593e+02,
        1.35837877e+02, 1.40405209e+02, 1.42403789e+02],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 6.47070444e-05, 4.62246540e-04, 1.28789904e-03,
        2.58697092e-03, 4.46184592e-03, 6.91881796e-03, 1.00486734e-02,
        1.39663896e-02, 1.88729304e-02, 2.49285644e-02, 3.23517676e-02,
        4.15042140e-02, 5.29429608e-02, 6.72368380e-02, 8.52583912e-02,
        1.08277385e-01, 1.38178712e-01, 1.77491940e-01, 2.29245848e-01,
        2.96748140e-01, 3.83444624e-01, 4.92341892e-01, 6.25533012e-01,
        7.83225140e-01, 9.64322500e-01, 1.16717124e+00, 1.38919040e+00,
        1.62764620e+00, 1.87962940e+00, 2.14238924e+00, 2.46875616e+00,
        2.87183872e+00, 3.36652272e+00, 3.97142956e+00, 4.71054632e+00,
        5.61501432e+00, 6.72701188e+00, 8.11350812e+00, 9.87013440e+00,
        1.21225880e+01, 1.50119048e+01, 1.85858552e+01, 2.27117688e+01,
        2.71386812e+01, 3.16789068e+01, 3.62324900e+01, 4.07838092e+01,
        4.53576552e+01, 4.99785924e+01, 5.46656384e+01, 5.94441500e+01,
        6.43215984e+01, 6.93038700e+01, 7.43879084e+01, 7.95588844e+01,
        8.48055912e+01, 9.01090112e+01, 9.54504664e+01, 1.00810147e+02,
        1.06169714e+02, 1.11515131e+02, 1.16838248e+02, 1.22136008e+02,
        1.27411128e+02, 1.32665872e+02, 1.37905900e+02, 1.43131212e+02,
        1.48346336e+02, 1.53551272e+02, 1.55829988e+02],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.46469056e-03,
        5.11233840e-03, 1.13853702e-02, 2.10206457e-02, 3.50379329e-02,
        5.48458245e-02, 8.23200588e-02, 1.20019989e-01, 1.70778897e-01,
        2.37543408e-01, 3.23018181e-01, 4.29273927e-01, 5.57563770e-01,
        7.07720882e-01, 8.78741743e-01, 1.06897335e+00, 1.27622529e+00,
        1.49800532e+00, 1.73152701e+00, 1.97389953e+00, 2.27380878e+00,
        2.64337848e+00, 3.09668080e+00, 3.64937981e+00, 4.32225900e+00,
        5.14380659e+00, 6.15341192e+00, 7.41151530e+00, 9.00870080e+00,
        1.10806471e+01, 1.37844914e+01, 1.72238328e+01, 2.13256489e+01,
        2.58450458e+01, 3.05727872e+01, 3.53946402e+01, 4.02797862e+01,
        4.52345927e+01, 5.02808364e+01, 5.54432234e+01, 6.07338517e+01,
        6.61639284e+01, 7.17408396e+01, 7.74616563e+01, 8.32945410e+01,
        8.92159339e+01, 9.52046951e+01, 1.01230388e+02, 1.07275183e+02,
        1.13315011e+02, 1.19336247e+02, 1.25329847e+02, 1.31294029e+02,
        1.37229813e+02, 1.43143947e+02, 1.49040252e+02, 1.54920002e+02,
        1.60788289e+02, 1.66645116e+02, 1.69208672e+02],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.33851115e-04,
        7.80818449e-04, 1.69816267e-03, 3.02465631e-03, 4.74465113e-03,
        6.95428765e-03, 9.72067220e-03, 1.29968077e-02, 1.69050871e-02,
        2.15135626e-02, 2.69635503e-02, 3.33137964e-02, 4.04931274e-02,
        4.86866790e-02, 5.81976440e-02, 6.92211842e-02, 8.20294249e-02,
        9.69915072e-02, 1.14579520e-01, 1.35786359e-01, 1.61675107e-01,
        1.93837279e-01, 2.34269118e-01, 2.85293644e-01, 3.49800158e-01,
        4.30286787e-01, 5.29230478e-01, 6.48208625e-01, 7.87705602e-01,
        9.47186196e-01, 1.12535404e+00, 1.32039731e+00, 1.53029802e+00,
        1.75253121e+00, 1.98452250e+00, 2.22423836e+00, 2.51997500e+00,
        2.88304391e+00, 3.32660650e+00, 3.86639436e+00, 4.52137293e+00,
        5.31785093e+00, 6.29318027e+00, 7.50520341e+00, 9.03982267e+00,
        1.10234555e+01, 1.36333716e+01, 1.70018207e+01, 2.10892056e+01,
        2.56779108e+01, 3.05452371e+01, 3.55467196e+01, 4.06443709e+01,
        4.58456754e+01, 5.11761935e+01, 5.66678402e+01, 6.23259817e+01,
        6.81636101e+01, 7.41880685e+01, 8.03896131e+01, 8.67584998e+01,
        9.32645082e+01, 9.98703570e+01, 1.06537212e+02, 1.13237252e+02,
        1.19934327e+02, 1.26612903e+02, 1.33261117e+02, 1.39876003e+02,
        1.46460387e+02, 1.53021329e+02, 1.59559676e+02, 1.66081077e+02,
        1.72588357e+02, 1.79084339e+02, 1.81927037e+02]])


## set the above data to the SALT grid
def DEM_flow(u_):
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    i = np.where(np.abs(u_DEM-u_)==np.min(np.abs(u_DEM-u_)))[0][0]
    uf_ref = uf_DEM[i]
    Z_ref = Z_DEM+Zmin
    f = interp1d(Z_ref,uf_ref)
    nuf = np.zeros(NZ)
    nuf[ZZ<=800] = f(ZZ[ZZ<=800])
    logwall = lambda z,z0: np.sqrt(1-1/s)*ustar*np.log(z/z0)/0.4
    z0 = curve_fit(logwall,ZZ[(ZZ<=800) & (ZZ>30)],nuf[(ZZ<=800) & (ZZ>30)])[0]
    nuf[ZZ>=800] = logwall(ZZ[ZZ>=800],z0)
    return nuf/np.sqrt(1-1/s)









import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from celluloid import Camera
def fig_init__gif():

    vf = np.zeros([L+1,len(ZZ[ZZ<200])])
    eXX = np.append(XX,L)
    eX,eZ = np.meshgrid(eXX,ZZ[ZZ<200]-.5)
    for i in range(len(Zsurf)):
        eZ[:,i]+= Zsurf[i]
    for x in eXX:
        vf[x] = uf[(ZZ-.5)<200]


    fig,ax= plt.subplots(1,1,figsize=(50,10),dpi=200)
    ax.set_xlim(0,L)
    ax.set_ylim(-3-A,120)
    ax.tick_params('both',labelsize=15)
    ax.set_ylabel(r'$z/d$',fontsize=25)
    ax.set_xlabel(r'$x/d$',fontsize=25)

    ax.set_aspect(1)
    fig.canvas.draw()
    grainsize = ((ax.get_window_extent().width  / (L+1) * 72./fig.dpi) ** 2)

    camera = Camera(fig)
    return camera,eX,eZ,vf,ax,grainsize

def loop__gif(camera,eX,eZ,vf,ax,grainsize):
    ax.pcolormesh(eX,eZ,vf.T,shading="gouraud")
    ax.plot(Zsurf,'k')
    for p in pars:            
        ax.scatter(p.xp ,p.zp ,c=np.arange(len(p.zp)),s=grainsize,cmap='binary')
    camera.snap()
    return camera
def save_gif(camera,gif_name,dpi=400):
    animation = camera.animate()
    animation.save(gif_name+'.gif', writer = 'imagemagick',dpi=dpi)
