import cv2

import numpy as np
import pandas as pd
import skimage.draw as draw
import scipy.ndimage as ndi
import skimage.measure as measure
import skimage.segmentation as seg

from collections import defaultdict
from scipy.spatial import ConvexHull, cKDTree
from .segment_kernels import build_adj_from_pairs, clear_low_diversity_edges, compute_cell_edges
#from scipy.optimize import minimize, leastsq


CELL_MORPHOLOGY_PROPERTIES = (
    'label',
    'area',
    'area_bbox',
    'area_convex',
    'area_filled',
    'axis_major_length',
    'axis_minor_length',
    'bbox',
    'centroid',
    'centroid_local',
    'eccentricity',
    'equivalent_diameter_area',
    'euler_number',
    'extent',
    'feret_diameter_max',
    'inertia_tensor',
    'inertia_tensor_eigvals',
    'major_axis_length',
    'moments',
    'moments_central',
    'moments_hu',
    'moments_normalized',
    'num_pixels',
    'orientation',
    'perimeter',
    'perimeter_crofton',
    'solidity',
)


class VMSI_obj:
    def __init__(self):
        self.V_df = []
        self.C_df = []
        self.E_df = []

class Segmenter:
    def __init__(self, images = None, masks = None, very_far = 300, labelled=False, padding = None):
        """
        :param: images: (Numpy array) Membrane-stained images to be segmented. WARNING: currently experimental and not working as intended. Default: None.
        :param masks: (Numpy array) Segmented image with edges set to zero and cells set to non-zero. Edges must be 1px wide and 4-connected. Default: None.
        :param very_far: (Int) Maximum distance in pixels between two vertices connected by the same edge. Default: 300.
        :param labelled: (Bool) Whether the segmented cells have been labelled. Default: False.
        :param padding: (List) padding = [original height, original width, min_row, max_row, min_col, max_col]
        """
        self.images = []
        self.masks = []
        self.very_far = very_far
        self.padding = []

        if images is not None:
            raise NotImplementedError()
            self.images = images
        if masks is not None:
            self.masks = masks
        if not labelled:
            self.masks = measure.label(self.masks)
        if padding is not None:
            self.padding = padding
        
        self.dtype = masks.dtype
        print(f'!!! USING {self.dtype} as default dtype of the mask')


    def process_segmented_image(self, holes_mask=None):
        """
        Given a segmented mask, produce VMSI_obj for input into VMSI
        """

        # Process mask
        # Clear border (create external cell from all cells that run into image boundary)
        tmp1 = seg.clear_border(self.masks)
        # If we are specifying holes, also set cells bordering holes as external cell
        if holes_mask is not None:
            hole_kernel = np.ones((5, 5), dtype=np.uint8)
            dilated_holes = cv2.dilate(holes_mask.astype(np.uint8), hole_kernel) > 0
            hole_adj_cells = np.unique(self.masks[dilated_holes])
            hole_adj_cells = hole_adj_cells[hole_adj_cells != 0]
            if hole_adj_cells.size > 0:
                label_cap = int(np.max(self.masks))
                lookup = np.zeros(label_cap + 1, dtype=np.bool_)
                lookup[hole_adj_cells.astype(np.intp, copy=False)] = True
                tmp1[lookup[self.masks.astype(np.intp, copy=False)]] = 0
        tmp3 = tmp1 + ((self.masks - tmp1) > 0).astype(self.dtype) # set to 1 all the holes (same as bg)

        # Find edge pixels that only separate external cells
        padded_tmp3 = np.pad(tmp3, 1, mode='constant', constant_values=0)
        clear_low_diversity_edges(tmp1, padded_tmp3)
        mask_tmp = tmp1
        
        # Relabel mask (may not be necessary in future, just for Matlab compatibility)
        # mask_tmp = self.relabel(mask_tmp)

        # Create VMSI object to store vertex, cell and edge information
        obj = VMSI_obj()

        obj.C_df = self.find_cells(mask_tmp)

        obj.V_df, cc = self.find_vertices(mask_tmp, obj.C_df)
        obj.E_df = self.find_edges(obj, mask_tmp, cc)
        self.identify_holes(obj)

        if self.padding:
            top_pad = self.padding[2]
            left_pad = self.padding[4]
            bottom_pad = self.padding[0] - self.padding[3] - 1
            right_pad = self.padding[1] - self.padding[5] - 1

            mask_tmp = np.pad(
                mask_tmp,
                ((top_pad, bottom_pad), (left_pad, right_pad)),
                mode='constant',
                constant_values=0
            )
            shift = np.array([left_pad, top_pad], dtype=np.int32)

            obj.C_df['centroids'] = list(np.vstack(obj.C_df['centroids']) + shift)
            for column in ('centroid-0', 'bbox-0', 'bbox-2'):
                if column in obj.C_df.columns:
                    obj.C_df[column] = obj.C_df[column] + top_pad
            for column in ('centroid-1', 'bbox-1', 'bbox-3'):
                if column in obj.C_df.columns:
                    obj.C_df[column] = obj.C_df[column] + left_pad
            obj.V_df['coords'] = list(np.vstack(obj.V_df['coords']) + shift)
            obj.E_df['pixels'] = [(pix[0] + left_pad, pix[1] + top_pad) for pix in obj.E_df['pixels']]
        
        return obj, mask_tmp
    
    def find_vertices(self, mask, C_df):
        V_df = pd.DataFrame(columns=['coords', 'ncells', 'nverts', 'edges'])

        branchpoints = self.find_branch_points(mask == 0)

        cc = measure.label(branchpoints, connectivity=2)
        regions = measure.regionprops(cc)

        if len(regions) == 0:
            V_df = pd.DataFrame({'coords': [],'ncells': [],'nverts': [],'edges': []})
            C_df['numv'] = np.zeros(len(C_df), dtype=int)
            C_df['nverts'] = [np.array([], dtype=int) for _ in range(len(C_df))]
            C_df['ncells'] = [np.array([0], dtype=int) for _ in range(len(C_df))]
            return V_df, cc

        v = np.array([np.flip(np.round(r.centroid).astype(int)) for r in regions], dtype=int)
        a = [r.coords for r in regions]

        order = v[:, 0].argsort()
        v = v[order]
        a = [a[i] for i in order]

        label_to_idx = self.label_to_idx

        ncells_list = []
        n = len(v)
        for coords in a:
            # take the region of the vertex and get the labels of surrounding cells
            rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
            cmin, cmax = coords[:, 1].min(), coords[:, 1].max()

            r0 = max(rmin - 1, 0)
            r1 = min(rmax + 2, mask.shape[0])
            c0 = max(cmin - 1, 0)
            c1 = min(cmax + 2, mask.shape[1])

            sub = mask[r0:r1, c0:c1]

            raw_labels = np.unique(sub[sub != 0])
            nc = np.array(
                [label_to_idx[lab] for lab in raw_labels if lab in label_to_idx],
                dtype=np.int32
            )
            ncells_list.append(nc)

        V_df = pd.DataFrame({'coords': list(v),'ncells': ncells_list,'nverts': [np.array([], dtype=int) for _ in range(n)],'edges': [np.array([], dtype=int) for _ in range(n)]})

        # adding the number of vertices per each cell
        cell_vertices = [[] for _ in range(len(C_df))]
        numv = np.zeros(len(C_df), dtype=int)

        for vidx, cells in enumerate(ncells_list):
            for c in cells:
                numv[c] += 1
                cell_vertices[c].append(vidx)

        C_df['numv'] = numv
        C_df['nverts'] = [np.array(vlist, dtype=int) for vlist in cell_vertices]

        # adding the neighbor cells per each cell
        cell_ncells = []
        for c_idx, verts in enumerate(cell_vertices):
            if len(verts) > 0:
                nc = np.unique(np.concatenate([ncells_list[v_idx] for v_idx in verts]))
                nc = nc[nc != c_idx]
            else:
                nc = np.array([0], dtype=int)
            cell_ncells.append(nc)

        C_df['ncells'] = cell_ncells

        R = v.astype(np.float64)
        if n > 1 and self.very_far > 0:
            tree = cKDTree(R)
            try:
                candidate_pairs = tree.query_pairs(r=float(self.very_far), output_type='ndarray')
            except TypeError:
                candidate_pairs = np.array(list(tree.query_pairs(r=float(self.very_far))), dtype=np.int32)
            if candidate_pairs.size == 0:
                candidate_pairs = np.empty((0, 2), dtype=np.int32)
            else:
                candidate_pairs = np.asarray(candidate_pairs, dtype=np.int32)
        else:
            candidate_pairs = np.empty((0, 2), dtype=np.int32)

        max_len = max((len(c) for c in ncells_list), default=0)
        ncells_arr = -np.ones((n, max_len), dtype=np.int32)
        ncells_adj = -np.ones((n, max_len), dtype=np.int32)
        for i, c in enumerate(ncells_list):
            if len(c) > 0:
                ncells_arr[i, :len(c)] = c
                c_adj = c[c != 0]
                if len(c_adj) > 0:
                    ncells_adj[i, :len(c_adj)] = c_adj

        # Match the original segment.py behavior: cell index 0 participates in
        # vertex ncells bookkeeping, but is ignored when deciding whether two
        # vertices share at least two cells and should be adjacent.
        matched_pairs = build_adj_from_pairs(ncells_adj, candidate_pairs)
        vertex_neighbours = [[] for _ in range(n)]
        for i, j in matched_pairs:
            vertex_neighbours[int(i)].append(int(j))
            vertex_neighbours[int(j)].append(int(i))
        V_df['nverts'] = [np.array(neigh, dtype=int) for neigh in vertex_neighbours]

        return V_df, cc

    def find_branch_points(self, skel):
        #!: TG: on GPU may be more efficient a convolution
        # Vectorized branch point finding; faster than convolving with filter
        skel = skel.astype(np.uint8, copy=False)
        neigh = (
            skel[2:, 1:-1] +
            skel[:-2, 1:-1] +
            skel[1:-1, 2:] +
            skel[1:-1, :-2]
        )
        out = np.zeros_like(skel, dtype=bool)
        out[1:-1, 1:-1] = (neigh >= 3) & skel[1:-1, 1:-1]
        return out

    def find_cells(self, mask):
        props = measure.regionprops_table(mask, properties=CELL_MORPHOLOGY_PROPERTIES)
        C_df = pd.DataFrame(props)

        labels_arr = C_df['label'].to_numpy(dtype=np.int32, copy=True)
        self.label_to_idx = {lab: i for i, lab in enumerate(labels_arr)}

        n = len(C_df)
        C_df['centroids'] = list(np.stack([C_df['centroid-1'], C_df['centroid-0']], axis=1))
        C_df['nverts'] = [np.array([], dtype=int) for _ in range(n)]
        C_df['numv'] = np.zeros(n, dtype=int)
        C_df['ncells'] = [np.array([], dtype=int) for _ in range(n)]
        C_df['edges'] = [np.array([], dtype=int) for _ in range(n)]
        C_df['holes'] = np.zeros(n, dtype=bool)
        C_df['inertia'] = list(np.stack([C_df['inertia_tensor-0-0'], C_df['inertia_tensor-0-1'], C_df['inertia_tensor-1-1']], axis=1))
        C_df['polygon_perimeter'] = np.zeros(n, dtype=float)
        C_df['feret_d'] = C_df['feret_diameter_max'].to_numpy(copy=True)
        C_df['moments_hu'] = list(np.stack([C_df[f'moments_hu-{i}'] for i in range(7)], axis=1))
        C_df['bbox'] = list(np.stack([C_df['bbox-3'] - C_df['bbox-1'], C_df['bbox-2'] - C_df['bbox-0']], axis=1))

        self.very_far = C_df['perimeter'].max() / 2 if n > 0 else 0
        return C_df

    def identify_holes(self, obj):
        """
        Filter out labelled objects that have area greater than 2x the median area and are non-convex
        """
        areas = obj.C_df['area'].to_numpy()
        for i in range(obj.C_df.shape[0]):
            vcoords = np.array(obj.V_df.loc[obj.C_df.at[i, 'nverts'], 'coords'].tolist())
            if vcoords.shape[0] >= 3:
                hull = ConvexHull(vcoords)
                if hull.simplices.shape[0] < vcoords.shape[0] and obj.C_df.at[i, 'area'] > 2*np.median(areas):
                    obj.C_df.at[i, 'holes'] = True
            else:
                obj.C_df.at[i, 'holes'] = True
        return


    def find_edges(self, obj, mask, cc):
        E_df = pd.DataFrame(columns = ['pixels','verts','cells'])
        if obj.V_df.empty:
            obj.C_df['edges'] = [np.array([], dtype=int) for _ in range(len(obj.C_df))]
            return E_df

        b_dat = (mask == 0)  # boolean mask of edged pixels

        rv = np.vstack(obj.V_df['coords'].values)  # array of vertex coordinates (x, y)
        b_dat[rv[:,1], rv[:,0]] = False  # remove vertex pixels from edges mask
        b_dat[cc != 0] = False  # remove branch point regions

        b_end = self.endpoints(b_dat)  # endpoints of the cleaned skeleton

        # Match segment.py endpoint scan order exactly: np.argwhere on the
        # transposed endpoint mask yields endpoints in (x, y) order and fixes
        # edge orientation/sign conventions downstream.
        re = np.argwhere(b_end.T != 0)

        # Keep the original scan order from segment.py so downstream edge indices
        # remain as close as possible to the legacy implementation.
        b_l = measure.label(b_dat.T, connectivity=1).T  # labeled connected components of edges
        b_props = measure.regionprops(b_l)  # region properties of edges components
        if re.size > 0:
            endpoint_coords = re.astype(np.float64, copy=False)
            vertex_coords = rv.astype(np.float64, copy=False)
            tree = cKDTree(vertex_coords)
            min_dist, nearest_vertices = tree.query(endpoint_coords, k=1)
            nearest_vertices = np.asarray(nearest_vertices, dtype=np.int32).reshape(-1)

            if vertex_coords.shape[0] > 1:
                for idx, point in enumerate(endpoint_coords):
                    radius = np.nextafter(float(min_dist[idx]), np.inf)
                    tied_vertices = tree.query_ball_point(point, r=radius)
                    if len(tied_vertices) <= 1:
                        continue

                    tied_vertices = np.asarray(tied_vertices, dtype=np.int32)
                    diff = vertex_coords[tied_vertices] - point
                    dist_sq = np.sum(diff * diff, axis=1)
                    min_sq = np.min(dist_sq)
                    nearest_vertices[idx] = int(np.min(tied_vertices[dist_sq == min_sq]))

            end_labels = b_l[re[:,1], re[:,0]]  # component label for each endpoint
        else:
            nearest_vertices = np.empty((0,), dtype=np.int32)
            end_labels = np.empty((0,), dtype=np.int32)

        E_pixels = []
        E_verts = []
        E_cells = []
        label_to_endpoints = defaultdict(list)
        for idx, lbl in enumerate(end_labels):
            label_to_endpoints[lbl].append(idx)

        V_nverts = obj.V_df['nverts'].values
        V_ncells = obj.V_df['ncells'].values
        C0_nverts = obj.C_df.at[0, 'nverts']

        for i in sorted(label_to_endpoints):
            endpoints_idx = label_to_endpoints[i]
            if i == 0:
                continue  # skip background
            if len(endpoints_idx) != 2:
                continue  # only edges with 2 endpoints

            e1, e2 = endpoints_idx
            v1 = int(nearest_vertices[e1])
            v2 = int(nearest_vertices[e2])

            if (v1 != -1) and (v2 != -1):
                if (v2 in V_nverts[v1]) and ((v1 not in C0_nverts) or (v2 not in C0_nverts)):
                    coords = b_props[i-1].coords
                    pix = tuple(np.flip(coords.T))
                    verts = np.array([v1, v2])
                    cells = np.intersect1d(V_ncells[v1], V_ncells[v2])
                    E_pixels.append(pix)
                    E_verts.append(verts)
                    E_cells.append(cells)

        E_df = pd.DataFrame({'pixels': E_pixels,'verts': E_verts,'cells': E_cells})

        # Edit V_df with edge information
        if E_df.empty:
            E_verts = np.empty((0, 2), dtype=np.int32)
        else:
            E_verts = np.vstack(E_df['verts'].values).astype(np.int32, copy=False)
        V_nverts = obj.V_df['nverts'].values
        V_coords = obj.V_df['coords'].values
        V_ncells = obj.V_df['ncells'].values
        C0_nverts = obj.C_df.at[0, 'nverts']

        edge_lookup = {}
        for idx, (a, b) in enumerate(E_verts):
            edge_lookup[(a, b)] = idx
            edge_lookup[(b, a)] = idx  # symmetric
        new_pixels = []
        new_verts = []
        new_cells = []

        V_edges = [[] for _ in range(len(obj.V_df))]
        for v in range(len(obj.V_df)):
            for nv in V_nverts[v]:
                edge_idx = edge_lookup.get((v, nv), None)

                if edge_idx is not None:
                    V_edges[v].append(edge_idx)
                elif (v not in C0_nverts) and (nv not in C0_nverts):
                    r0, c0 = V_coords[v][1], V_coords[v][0]
                    r1, c1 = V_coords[nv][1], V_coords[nv][0]

                    line = draw.line(r0, c0, r1, c1)
                    pix = tuple(np.flip(line, axis=0))

                    verts = np.array([v, nv])
                    cells = np.intersect1d(V_ncells[v], V_ncells[nv])
                    new_idx = len(E_verts) + len(new_verts)
                    new_pixels.append(pix)
                    new_verts.append(verts)
                    new_cells.append(cells)
                    edge_lookup[(v, nv)] = new_idx
                    edge_lookup[(nv, v)] = new_idx
                    V_edges[v].append(new_idx)
                else:
                    V_edges[v].append(-1)
        obj.V_df['edges'] = [np.array(e, dtype=int) for e in V_edges]
        if new_verts:
            E_df = pd.concat([E_df, pd.DataFrame({'pixels': new_pixels,'verts': new_verts,'cells': new_cells})], ignore_index=True)

        # Edit C_df with edge information
        if len(obj.C_df) == 0:
            return E_df
        V_coords = np.vstack(obj.V_df['coords'].values)
        V_nverts = self.pad_list(obj.V_df['nverts'].values)
        V_edges  = self.pad_list(obj.V_df['edges'].values)
        C_nverts = self.pad_list(obj.C_df['nverts'].values)
        
        C0_flag = np.zeros(len(obj.C_df), dtype=np.bool_)
        if len(obj.C_df) > 0:
            for c in obj.C_df.at[0, 'ncells']:
                if 0 <= c < len(C0_flag):
                    C0_flag[c] = True

        C_edges = compute_cell_edges(V_coords, V_nverts, V_edges, C_nverts, C0_flag)
        
        obj.C_df['edges'] = [row[row >= 0] for row in C_edges]
        return E_df
    
    @staticmethod
    def pad_list(arr_list, fill=-1):
        if len(arr_list) == 0:
            return np.empty((0, 0), dtype=np.int32)
        max_len = max(len(a) for a in arr_list)
        out = np.full((len(arr_list), max_len), fill, dtype=np.int32)
        for i, a in enumerate(arr_list):
            out[i, :len(a)] = a
        return out

    def endpoints(self, image):
        # Define endpoint as pixel with only 1 4-connected neighbor
        # This requires the skeletonized image to be 4-connnected
        image = image.astype(np.uint8, copy=False)
        k = np.array([[0,1,0],
                    [1,0,1],
                    [0,1,0]], dtype=np.uint8)
        neigh = ndi.convolve(image, k, mode='constant', cval=0)
        return (image == 1) & (neigh == 1)

