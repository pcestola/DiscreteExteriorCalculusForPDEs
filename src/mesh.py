import igl
import numpy as np

from enum import Enum

class StatoMesh(Enum):
    VUOTA = 0
    CARICATA = 1

class Mesh():
    def __init__(self) -> None:
        self.vertices = None
        self.faces = None
        self.num_vertices = 0
        self.num_faces = 0
        self.state = StatoMesh.VUOTA
    
    def load_from_file(self, path:str):
        '''
            Load mesh from file with .obj extension
        '''
        self.vertices, self.faces = igl.read_triangle_mesh(path)
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]
        self.state = StatoMesh.CARICATA

    def get_boundary(self):
        return igl.boundary_loop(self.faces)

    def get_face_normals(self):
        '''
            Evaluate face normals
        '''
        assert self.state != StatoMesh.VUOTA
        return igl.per_face_normals(self.vertices, self.faces)
    
    def get_vertex_normals(self, normalize=False, eps=1e-12):
        '''
            Evaluate face normals
        '''
        assert self.state != StatoMesh.VUOTA
        normals = igl.per_vertex_normals(self.vertices, self.faces)
        if normalize:
            norm = 1 / (eps + np.linalg.norm(normals, axis=1))
            return normals * norm[:,np.newaxis]
        else:
            return normals
    
    def get_gradient(self):
        '''
            Evaluate gradient matrix of the mesh
        '''
        assert self.state != StatoMesh.VUOTA
        return igl.grad(self.vertices, self.faces)
    
    # TODO: Capire perché non funzionano
    def get_cot_laplacian(self, mtype='matrix'):
        '''
            Evaluate cotangent matrix of the mesh
        '''
        assert self.state != StatoMesh.VUOTA
        
        laplacian = -igl.cotmatrix(self.vertices, self.faces)
        
        if mtype == 'matrix':
            return laplacian
        elif mtype == 'csc':
            return laplacian.tocsc()
        elif mtype == 'csr':
            return laplacian.tocsr()
    
    def get_mass_matrix(self, mtype='matrix'):
        '''
            Evaluate mass matrix of the mesh
        '''
        assert self.state != StatoMesh.VUOTA

        mass = igl.massmatrix(self.vertices, self.faces)

        if mtype == 'matrix':
            return mass
        elif mtype == 'csc':
            return mass.tocsc()
        elif mtype == 'csr':
            return mass.tocsr()
