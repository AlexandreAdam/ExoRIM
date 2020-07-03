''' --------------------------------------------------------------------
                PYSCO: PYthon Self Calibrating Observables
    --------------------------------------------------------------------
    ---
    pysco is a python module to create, and extract Kernel-phase data 
    structures, using the theory of Martinache, 2010, ApJ, 724, 464.
    ----

    This file contains the definition of the kpi class:
    --------------------------------------------------

    an object that contains the linear model for the optical system
      of interest. Properties of this model are:
      --> name   : name of the model (HST, Keck, Annulus_19, ...)
      --> mask   : array of coordinates for pupil sample points
      --> uv     : matching array of coordinates in uv plane (baselines)
      --> RED   : vector coding the redundancy of these baselines
      --> TFM   : transfer matrix, linking pupil-phase to uv-phase
      --> KerPhi : array storing the kernel-phase relations
      --> uvrel :  matrix storing the relations between sampling points and uv-points
      -------------------------------------------------------------------- '''

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gzip
import time
from scipy.sparse.linalg import svds



class kpi(object):
    ''' Fundamental kernel-phase relations

    -----------------------------------------------------------------------
    This object condenses all the knowledge about a given instrument pupil 
    geometry into a series of arrays useful for kernel-phase analysis as 
    well as for other purposes, such as wavefront sensing.
    ----------------------------------------------------------------------- '''


    # =========================================================================
    # =========================================================================

    def __init__(self, mask, bsp_mat='sparse', verbose=False, ns=3):
        '''
        :param file: Coordinate of apertures .txt form
        :param bsp_mat: Bispectrum matrix format
        :param ns: Number of samples, set < 2 if undersampled data

        Creation of the KerPhase_Relation object from a pupil mask file:

        ----------------------------------------------------------------
        This is the core function of this class, really...

        Input is a pupil coordinates file, containing one set of (x,y) 
        coordinates per line. Coordinates are in meters. From this, all 
        the intermediate products that lead to the kernel-phase matrix 
        KerPhi are calculated.
                                
        Set Ns < 2 for undersampled data [AL, 20.02.2014]
        ----------------------------------------------------------------
        '''
        array_name = ""  # default array name. Should be descriptive of the array geometry
        self.mask = mask
        self.nbh  = self.mask.shape[0]   # number of sub-Ap

        ndgt = 6 # number of digits of precision for rounding
        prec = 10**(-ndgt)

        # ================================================
        # Determine all the baselines in the array.
        # ================================================

        # 1. Start by doing all the possible combinations of coordinates 
        # --------------------------------------------------------------
        # in the array to calculate the baselines. The intent here, is 
        # to work with redundant arrays of course, so there will be plenty 
        # of duplicates.

        nbh = self.nbh # local representation of the class variable
        uvx = np.zeros(nbh * (nbh-1)) # prepare empty arrays to store
        uvy = np.zeros(nbh * (nbh-1)) # the baselines

        k = 0 # index for possible combinations (k = f(i,j))
        
        uvi = np.zeros(nbh * (nbh-1), dtype=int) # arrays to store the possible
        uvj = np.zeros(nbh * (nbh-1), dtype=int) # combinations k=f(i,j) !!


        for i in range(nbh):     # do all the possible combinations of
            for j in range(nbh): # sub-apertures
                if i != j:
                    uvx[k] = self.mask[i,0] - self.mask[j,0]
                    uvy[k] = self.mask[i,1] - self.mask[j,1]
                    # ---
                    uvi[k], uvj[k] = i, j
                    k+=1

        a = np.unique(np.round(uvx, ndgt)) # distinct u-component of baselines
        nbx = a.shape[0]                    # number of distinct u-components
        uv_sel = np.zeros((0,2))           # array for "selected" baselines

        for i in range(nbx):     # identify distinct v-coords and fill uv_sel
            b = np.where(np.abs(uvx - a[i]) <= prec)
            c = np.unique(np.round(uvy[b], ndgt))
            nby = np.shape(c)[0] # number of distinct v-compoments
            app = np.ones(nby)*a[i]
            uv_sel = np.append(uv_sel, np.array([app,c]).T, axis=0)

        self.nbuv = int(np.shape(uv_sel)[0]/2) # actual number of distinct uv points
        self.uv = uv_sel[:self.nbuv,:]  # discard second half (symmetric)
        print("%d distinct baselines were identified" % (self.nbuv,))

        # 1.5. Special case for undersampled data
        # ---------------------------------------
        if (ns < 2):
            uv_sampl = self.uv.copy()   # copy previously identified baselines
            uvm = np.abs(self.uv).max() # max baseline length
            keep = (np.abs(uv_sampl[:,0]) < (uvm * ns / 2.)) * \
                   (np.abs(uv_sampl[:,1]) < (uvm * ns / 2.))
            self.uv = uv_sampl[keep]
            self.nbuv = (self.uv.shape)[0]

            print("%d baselines were kept (undersampled data)" % (self.nbuv,))

        # 2. Calculate the transfer matrix and the redundancy vector
        # [AL, 2014.05.22] keeping relations between uv points and sampling points
        # --------------------------------------------------------------
        self.TFM = np.zeros((self.nbuv, self.nbh), dtype=float) # matrix
        self.RED = np.zeros(self.nbuv, dtype=float)    # Redundancy
        # relations matrix (-1 = not connected. NB: only positive baselines are saved)
        self.uvrel=-np.ones((nbh,nbh),dtype='int')                  
        for i in range(self.nbuv):
            a=np.where((np.abs(self.uv[i,0]-uvx) <= prec) *
                       (np.abs(self.uv[i,1]-uvy) <= prec))
            for k in range(len(a[0])) :
                 self.uvrel[uvi[a][k],uvj[a][k]]=i  
                 #self.uvrel[uvj[a][k],uvi[a][k]]=i
                                                                                        
            self.TFM[i, uvi[a]] +=  1.0
            self.TFM[i, uvj[a]] -=  1.0
            self.RED[i]   = np.size(a)
        # converting to relations matrix
                                                    

        # 3. Determine the kernel-phase relations
        # ----------------------------------------

        # One sub-aperture is taken as reference: the corresponding
        # column of the transfer matrix is discarded. TFM is now a
        # (nbuv) x (nbh - 1) array.
        
        # The choice is up to the user... but the simplest is to
        # discard the first column, that is, use the first aperture
        # as a reference?

        self.TFM = self.TFM[:,1:] # cf. explanation
        self.TFM = np.dot(np.diag(1./self.RED), self.TFM) # experiment #[Al, 2014.05.12] Frantz's version                           
        U, S, Vh = np.linalg.svd(self.TFM.T, full_matrices=1) 

        S1 = np.zeros(self.nbuv)
        S1[0:nbh-1] = S

        self.nkphi  = np.size(np.where(abs(S1) < 1e-3)) # number of Ker-phases
        KPhiCol  = np.where(abs(S1) < 1e-3)[0]
        self.KerPhi = np.zeros((self.nkphi, self.nbuv)) # allocate the array

        for i in range(self.nkphi):
            self.KerPhi[i,:] = (Vh)[KPhiCol[i],:]

        if verbose:
            print('-------------------------------')
            print('Singular values for this array:\n', np.round(S, ndgt))
            print('\nRedundancy Vector:\n', self.RED)
        else:
            print('%d Kernel Phases identified.' % self.nkphi)
        self.name = array_name

        if bsp_mat is not None:
            print('Now calculating bispectrum')
            self.generate_bispectrum_matrix2(bsp_mat = bsp_mat)

    # =========================================================================
    # =========================================================================

    def plot_pupil_and_uv(self, xymax = 8.0):
        ''' Nice plot of the pupil sampling and matching uv plane.

        --------------------------------------------------------------------
        xymax just specifies the size of the region represented in the plot,
        expressed in meters. Should typically be slightly larger than the 
        largest baseline in the array.
        --------------------------------------------------------------------'''

        plt.clf()
        f0 = plt.subplot(121)
        f0.plot(self.mask[:,0], self.mask[:,1], 'bo')
        f0.axis([-xymax, xymax, -xymax, xymax], aspect='equal')
        plt.title(self.name+' pupil')
        f1 = plt.subplot(122)

        f1.plot(self.uv[:,0],   self.uv[:,1], 'bo') # plot baselines + symetric
        f1.plot(-self.uv[:,0], -self.uv[:,1], 'ro') # for a "complete" feel
        plt.title(self.name+' uv coverage')
        f1.axis([-2*xymax, 2*xymax, -2*xymax, 2*xymax], aspect='equal')


        # complete previous plot with redundancy of the baseline
        # -------------------------------------------------------
        dy = 0.1*abs(self.uv[0,1]-self.uv[1,1]) # to offset text in the plot.
        for i in range(self.nbuv):
            f1.text(self.uv[i,0]+dy, self.uv[i,1]+dy, 
                    int(self.RED[i]), ha='center')
        
        f0.axis('equal')
        f1.axis('equal')
        #plt.draw()


    # =========================================================================
    # =========================================================================

    def save_to_file(self, file):
        ''' Export the KerPhase_Relation data structure into a pickle
        
        ----------------------------------------------------------------
        To save on disk space, this procedure uses the gzip module.
        While there is no requirement for a specific extension for the
        file, I would recommend that one uses ".kpi.gz", so as to make
        it obvious that the file is a gzipped kpi data structure.
        ----------------------------------------------------------------  '''
        try: 
            try:
                data = {'name'   : self.name,
                        'mask'   : self.mask,
                        'uv'     : self.uv,
                        'TFM'   : self.TFM,
                        'KerPhi' : self.KerPhi,
                        'RED'   : self.RED,
                        'uvrel' : self.uvrel,
                        # 'uv_samp_rev': self.uv_samp_rev,
                        'uv_to_bsp': self.uv_to_bsp,
                        'bsp_s': self.bsp_s} 
                print('KerPhase_Relation data structure was saved.'         )
            except:
                data = {'name'   : self.name,
                        'mask'   : self.mask,
                        'uv'     : self.uv,
                        'TFM'   : self.TFM,
                        'KerPhi' : self.KerPhi,
                        # 'uv_samp_rev': self.uv_samp_rev,
                        'RED'   : self.RED,
                        'uvrel' : self.uvrel}
                print('KerPhase_Relation data structure was saved. No bispectrum!'                                                                                         )
        except:
            print("KerPhase_Relation data structure is incomplete")
            print("File %s wasn't saved!" % (file,))
            return None
        # -------------
        try: myf = gzip.GzipFile(file, "wb")
        except:
            print("File %s cannot be created."+
                  " KerPhase_Relation data structure wasn't saved." % (file,))
            return None
        # -------------
        pickle.dump(data, myf, -1)
        myf.close()

###############################################################################
###############################################################################
###############################################################################
###############################################################################

###############################################################################
        
    def generate_bispectrum_matrix2(self, n=5, verbose=False,bsp_mat='sparse'):
        ''' Calculates the matrix to convert from uv phases to bispectra.
        This version iterates through the sampling points in a vectorized way.
        It saves all of the triangles, then removes the duplicates every 'n'
        iterations. Reduce the number to save on ram but make it much slower.
        
        n_guess_bsp: guess the number of bispectra and pre-allocate the memory
        to save millions of 'append' calls (which was the slowest part). It must
        be large enough to contain all of the bispectra, or you will get an error.
        '''
        n_guess_bsp = self.nbh*(self.nbh-1)*(self.nbh-2) // 6 # int(n_guess_bsp)
        uv_to_bsp = np.zeros((n_guess_bsp,self.nbuv),dtype=np.long)
        bsp_u = np.zeros((n_guess_bsp,3)) # the u points of each bispectrum point
        bsp_v = np.zeros((n_guess_bsp,3)) # the v points of each bispectrum point
        already_done = np.zeros((n_guess_bsp),dtype=np.longlong) # to track the sets of uv points that have already been counted        
        bsp_ix=0
        uvrel=self.uvrel+np.transpose(self.uvrel)+1
        
        nbits=np.longlong(np.ceil(np.log(self.nbuv)/np.log(10)))
        
        print('Calculating bispectrum matrix. Will take a few minutes.')
        
        # Loop over the first pupil sampling point
        tstart=time.time()
        for ix1 in range(self.nbh):
            
            # Loop over the second pupil sampling point
            for ix2 in range(ix1 + 1, self.nbh):
                # Rather than a for loop, vectorize it!
                ix3s=np.arange(ix2 + 1, self.nbh)
                n_ix3s = ix3s.size
                
                if (bsp_ix+n_ix3s) > n_guess_bsp:
                    raise IndexError('Number of calculated bispectra exceeds the initial guess for the matrix size!')
                
                # Find the baseline indices
                b1_ix = uvrel[ix1, ix2]
                b2_ixs = uvrel[ix2, ix3s]
                b3_ixs = uvrel[ix1, ix3s]  # we actually want the negative of this baseline
                b1_ixs = np.repeat(b1_ix, n_ix3s)
                
                # What uv points are these?
                uv1 = self.uv[b1_ixs, :]
                uv2 = self.uv[b2_ixs, :]
                uv3 = self.uv[b3_ixs, :]

                # This is not necessary?
                # Are they already in the array? (any permutation of these baselines is the same)
                # Convert to a single number to find out.
                bl_ixs = np.array([b1_ixs, b2_ixs, b3_ixs])
                bl_ixs = np.sort(bl_ixs, axis=0)
                these_triplet_nums = (10**(2*nbits))*bl_ixs[2, :] + (10**nbits)*bl_ixs[1, :] + bl_ixs[0, :]
                
                # Just add them all and remove the duplicates later.
                already_done[bsp_ix:bsp_ix+n_ix3s] = these_triplet_nums
                    
                # add to all the arrays
                uv_to_bsp_line = np.zeros((n_ix3s, self.nbuv))
                diag = np.arange(n_ix3s)
                # Are we sure this respects the closure relation?
                uv_to_bsp_line[diag, b1_ixs] += 1
                uv_to_bsp_line[diag, b2_ixs] += 1
                uv_to_bsp_line[diag, b3_ixs] += -1
                uv_to_bsp[bsp_ix:bsp_ix + n_ix3s, :] = uv_to_bsp_line

                bsp_u[bsp_ix:bsp_ix+n_ix3s, :] = np.transpose(np.array([uv1[:, 0], uv2[:, 0], uv3[:, 0]]))
                bsp_v[bsp_ix:bsp_ix+n_ix3s, :] = np.transpose(np.array([uv1[:, 1], uv2[:, 1], uv3[:, 1]]))
                bsp_ix += n_ix3s
                
            # remove the duplicates every n loops
            if (ix1 % n) == ((self.nbh-1) % n):
                # the (nbh-1 mod n) ensures we do this on the last iteration as well
                dummy, unique_ix = np.unique(already_done[0:bsp_ix + n_ix3s], return_index=True)
                bsp_ix = len(unique_ix)
                already_done[0:bsp_ix] = already_done[unique_ix]
                already_done[bsp_ix:] = 0
                uv_to_bsp[0:bsp_ix] = uv_to_bsp[unique_ix]
                bsp_u[0:bsp_ix] = bsp_u[unique_ix]
                bsp_v[0:bsp_ix] = bsp_v[unique_ix]
                
                # Only print(the status every 5*n iterations)
                if (ix1 % (5*n)) == ((self.nbh-1) % n):
                    print('Done', ix1, 'of', self.nbh, '. ', bsp_ix, ' bispectra found. Time taken:',np.round(time.time()-tstart,decimals=1),'sec')
            
        print('Done. Total time taken:',np.round((time.time()-tstart)/60.,decimals=1),'mins')
        
        # Remove the excess parts of each array and attach them to the kpi.
        nbsp = bsp_ix
        self.already_done = already_done
        self.nbsp = bsp_ix
        self.uv_to_bsp = uv_to_bsp[0:bsp_ix]
        self.bsp_u = bsp_u[0:bsp_ix]
        self.bsp_v = bsp_v[0:bsp_ix]
        print('Found',nbsp,'bispectra')
        t_start2 = time.time()

        tol = 1e-5

        try:
            if bsp_mat == 'sparse':
                print('Doing sparse svd')
                rank = np.linalg.matrix_rank(uv_to_bsp.astype('double'), tol = tol)
                print('Matrix rank:',rank)
                u, s, vt = svds(uv_to_bsp.astype('double').T, k=rank)

            elif bsp_mat == 'full':
                print('Attempting full svd')
                u, s, vt = np.linalg.svd(uv_to_bsp.astype('double').T,full_matrices=False)

                rank = np.sum(s>tol)
            sys.stdout.flush()

            self.uv_to_bsp_raw = np.copy(uv_to_bsp)
            self.uv_to_bsp = u.T
            self.nbsp = rank 
            self.bsp_s = s

            print('Reduced-rank bispectrum matrix calculated.')
            print('Matrix shape',self.uv_to_bsp.shape)
            print('Time taken:',np.round((time.time()-t_start2)/60.,decimals=1),'mins')

            if verbose:
                print(np.log(s) )
                return s
        except:
            print('SVD failed. Using raw matrix.')
            self.uv_to_bsp = uv_to_bsp 
            self.nbsp = nbsp 
        sys.stdout.flush()

if __name__ == "__main__":
    coords = np.random.randn(21, 2)
    bs = kpi(coords, bsp_mat='sparse')