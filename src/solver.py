import scipy.sparse as sp
import numpy as np

import sys
import matplotlib.pyplot as plt

from src.mesh import Mesh
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve, inv, splu


class Solver:
    def __init__(self, mesh: Mesh, time: np.ndarray):
        self.mesh = mesh
        self.time = time
        self.time_step = time[1] - time[0]

    def get_iteration_matrix(self, method: callable):
        raise NotImplementedError

    def solve(self, initial_condition: np.ndarray, method: str):
        raise NotImplementedError



class HeatEquation(Solver):
    def __init__(self, mesh: Mesh, time: np.ndarray, diffusivity:float):
        super().__init__(mesh, time)
        self.diffusivity = diffusivity


    def get_iteration_matrix(self):
        
        # TODO: Non funziona
        #laplacian = self.mesh.get_cot_laplacian(mtype='csc')
        #mass = self.mesh.get_mass_matrix(mtype='csc')

        # Form the system matrix for the implicit Euler method
        #system_matrix = mass - self.time_step * inv(mass) @ laplacian

        laplacian = get_cot_laplacian(self.mesh)
        system_matrix = sp.eye(laplacian.shape[0], format='csc') - self.diffusivity * self.time_step * laplacian

        # Use splu to factorize the system matrix
        return splu(system_matrix)


    def solve(self, initial_condition:np.ndarray, boundary_points:np.ndarray):
        num_variables = initial_condition.shape[0]   #[0] qui ok ma in generale no
        num_time_steps = self.time.shape[0]
        
        solution = np.zeros((num_time_steps, num_variables))
        solution[0] = initial_condition

        matrix = self.get_iteration_matrix()

        for i in range(num_time_steps-1):
            solution[i+1] = matrix.solve(solution[i])
            solution[i+1][boundary_points] = 1

        return solution



class TuringPattern(Solver):
    def __init__(self, mesh: Mesh, time: np.ndarray,
                 d1:float,
                 d2:float,
                 f:float,
                 k:float
                 ):
        super().__init__(mesh, time)
        self.d1 = d1
        self.d2 = d2
        self.f = f
        self.k = k

    def get_iteration_matrix(self):
        
        laplacian = get_cot_laplacian(self.mesh)

        return laplacian


    def solve(self, initial_phi:np.ndarray, initial_psi:np.ndarray, boundary=np.ndarray):
        num_variables = initial_phi.shape[0]
        num_time_steps = self.time.shape[0]
        
        phi = np.zeros((num_time_steps, num_variables))
        psi = np.zeros((num_time_steps, num_variables))
        phi[0] = initial_phi
        psi[0] = initial_psi

        phi_boundary_values = initial_phi[boundary]
        psi_boundary_values = initial_psi[boundary]
        
        matrix = self.get_iteration_matrix()

        for i in range(num_time_steps-1):
            phi[i+1] = phi[i] + self.time_step * (self.d1 * matrix @ phi[i] - phi[i]*np.power(psi[i],2) + self.f*(1-phi[i]))
            psi[i+1] = psi[i] + self.time_step * (self.d2 * matrix @ psi[i] + phi[i]*np.power(psi[i],2) - (self.f+self.k)*psi[i])
            phi[i+1][boundary] = phi_boundary_values
            psi[i+1][boundary] = psi_boundary_values

        return phi, psi



class WaveEquationCrankNicolson(Solver):
    def __init__(self, mesh: Mesh, time: np.ndarray, wave_speed: float):
        super().__init__(mesh, time)
        self.wave_speed = wave_speed
        self.time_step = time[1] - time[0]  # Assuming uniform time steps

    def get_system_matrices(self):
        # This method computes the Laplacian based on your mesh
        laplacian = get_cot_laplacian(self.mesh)  # Placeholder function to compute Laplacian
        A = splu(sp.identity(laplacian.shape[0], format='csc') - 0.5 * self.wave_speed * self.time_step**2 * laplacian)
        B = sp.identity(laplacian.shape[0], format='csc') + 0.5 * self.wave_speed * self.time_step**2 * laplacian
        return A, B, laplacian

    def solve(self, initial_condition: np.ndarray, initial_velocity: np.ndarray):
        n = len(initial_condition)
        num_time_steps = len(self.time)
        
        # Initialize arrays for the solution
        solution = np.zeros((num_time_steps, n))

        # Set initial conditions
        solution[0] = initial_condition
        velocity = initial_velocity

        A, B, C = self.get_system_matrices()

        # Use a linear solver to solve the system at each step
        for i in range(num_time_steps-1):
            solution[i+1] = A.solve(B @ solution[i] + self.time_step * velocity)
            velocity += 0.5 * self.time_step * C @ (solution[i+1] + solution[i])

        return solution



def get_cot_laplacian(mesh:Mesh, eps:float = 1e-12, format:str = 'csc'):
    '''
        Evaluate the cot-laplacian of a given mesh

        Input
        - mesh: (Mesh), the mesh
        - eps: (float), clip value
        - format: (str), format of data for the laplacian [csc,csr]
    '''
    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    # Get number of elements
    num_of_vertices = vertices.shape[0]
    num_of_faces = faces.shape[0]
    
    # Get vertex position for every face
    face_vertices = vertices[faces]
    v0, v1, v2 = face_vertices[:,0], face_vertices[:,1], face_vertices[:,2]

    # Evaluate side length of each triangle
    # A opposite to v1, B opposite to v2, C opposite to v3
    A = np.linalg.norm((v1-v2),axis=1)
    B = np.linalg.norm((v0-v2),axis=1)
    C = np.linalg.norm((v0-v1),axis=1)

    # Area of every triangle (Heron's formula)
    s = 0.5 * (A + B + C)
    # Area can be negative so we clip it
    area = np.sqrt(np.clip((s * (s - A) * (s - B) * (s - C)), a_min=eps, a_max=None))

    # Cotangent of angles
    A, B, C = A * A, B * B, C * C
    cot_a = (B + C - A) / area
    cot_b = (A + C - B) / area
    cot_c = (A + B - C) / area
    cot = np.stack([cot_a, cot_b, cot_c], axis=1)
    cot /= 4

    # Construct sparse matrix
    ii = faces[:, [1,2,0]]
    jj = faces[:, [2,0,1]]
    idx = np.stack([ii,jj],axis=0).reshape((2, 3*num_of_faces))

    # TODO: da migliorare, mettere qui la csc e fare dopo i calcoli non ha senso
    laplacian = sp.csc_matrix((cot.reshape(-1), idx), (num_of_vertices, num_of_vertices))

    if format == 'csc':
        laplacian = sp.csc_matrix((cot.reshape(-1), idx), (num_of_vertices, num_of_vertices))
    elif format == 'csr':
        laplacian = sp.csr_matrix((cot.reshape(-1), idx), (num_of_vertices, num_of_vertices))
    elif format == 'lil':
        coo = sp.coo_matrix((cot.reshape(-1), (idx[0], idx[1])), shape=(num_of_vertices, num_of_vertices))
        laplacian = coo.tolil()
    else:
        raise NotImplementedError

    # TODO: andrebbero messi prima della conversione
    # Symmetric
    rows, cols = laplacian.nonzero()
    laplacian[cols, rows] = laplacian[rows, cols]

    # Final step
    laplacian -= sp.diags(laplacian.sum(axis=1).A1)

    return laplacian
