import trimesh
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import gradio as gr
import time
bone_colors = plt.get_cmap('tab10')

from Anymate.utils.utils import load_checkpoint, get_joint, get_connectivity, get_skinning
from Anymate.utils.dataset_utils import obj2mesh
from Anymate.args import anymate_args
# from Anymate.utils.render_utils import empty, add_co, add_mesh, add_joint, add_conn, add_skin, setup_armature

def visualize_results(mesh_file=None, joints=None, conns=None, skins=None):
    # Create a scene with both original and processed meshes
    scene = trimesh.Scene()
    vis_file = mesh_file.replace('object.obj', 'vis.glb')
        
    if mesh_file is not None:
        # Load the original mesh (in blue) with transparency
        original_mesh = trimesh.load(mesh_file)
        # original_mesh = obj2mesh(mesh_file)
        if skins is not None:
            # pdb.set_trace()
            # Get per-vertex colors based on skinning weights
            vertex_colors = np.zeros((len(original_mesh.vertices), 4))
            
            # Convert skinning weights to numpy if needed
            if isinstance(skins, torch.Tensor):
                skins = skins.cpu().numpy()
            
            # For each bone, blend colors based on skinning weights
            for bone_idx in range(skins.shape[1]):
                bone_color = np.array(bone_colors(bone_idx % 10))  # Get base color for this bone
                weights = skins[:, bone_idx]
                vertex_colors += np.outer(weights, bone_color)  # Blend weighted colors
                
            # Normalize and clip colors
            vertex_colors = np.clip(vertex_colors, 0, 1)
            
            # Convert to vertex colors and set alpha
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
            vertex_colors[:, 3] = 255  # Set alpha to 100 for transparency
            # print(vertex_colors.shape)
            # print(vertex_colors.max(axis=0), vertex_colors.min(axis=0), vertex_colors.mean(axis=0))
            
            # Apply colors directly to vertices
            original_mesh.visual.vertex_colors = vertex_colors

            # face_colors = np.zeros((len(original_mesh.faces), 4))

            processed_mesh = trimesh.load(mesh_file)
            # processed_mesh = obj2mesh(mesh_file)
            # Assign vertex colors from original_mesh to processed_mesh
            # Since they might have different number of vertices, we need to find closest vertices
            
            # Get vertices from both meshes
            orig_vertices = original_mesh.vertices
            proc_vertices = processed_mesh.vertices
            
            # For each vertex in processed_mesh, find the closest vertex in original_mesh
            closest_indices = []
            for proc_vertex in proc_vertices:
                # Calculate distances to all original vertices
                distances = np.linalg.norm(orig_vertices - proc_vertex, axis=1)
                # Find index of closest vertex
                closest_idx = np.argmin(distances)
                closest_indices.append(closest_idx)
        
            proc_vertex_colors = original_mesh.visual.vertex_colors[closest_indices]
            processed_mesh.visual.vertex_colors = proc_vertex_colors
            original_mesh = processed_mesh

        else:
            original_mesh.visual.face_colors = [255, 255, 255, 100]  # Blue with alpha=100 for transparency
        scene.add_geometry(original_mesh)

    if joints is not None:
        # create a sphere for each joint
        for position in joints:
            sphere = trimesh.primitives.Sphere(radius=0.02)
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red with transparency
            sphere.apply_translation(position.cpu().numpy())
            scene.add_geometry(sphere)

        if conns is not None:
            # create a line for each connectivity
            for i, conn in enumerate(conns):
                if i == conn:
                    continue
                # Create cylinder between joints
                points = [joints[i].cpu().numpy(), joints[conn].cpu().numpy()]
                direction = points[1] - points[0]
                height = np.linalg.norm(direction)
                cylinder = trimesh.primitives.Cylinder(radius=0.01, height=height)
                
                # Calculate rotation matrix to align cylinder with direction
                direction = direction / height  # Normalize direction vector
                up_vector = np.array([0, 0, 1])
                rotation_matrix = trimesh.geometry.align_vectors(up_vector, direction)
                
                # Apply rotation and translation to cylinder
                cylinder.apply_transform(rotation_matrix)
                cylinder.apply_translation(points[0] + direction * height/2)
                
                cylinder.visual.face_colors = [0, 0, 255, 255]  # Blue
                scene.add_geometry(cylinder)
    
    # Export the scene
    scene.export(vis_file)
    return vis_file


def process_mesh_to_pc(obj_path, sample_num = 8192, save_path = None):
    # mesh_list : list of trimesh
    try :
        mesh = trimesh.load_mesh(obj_path)

        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]

        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)


        if save_path is not None:
            np.save(save_path, pc_normal)
        
        return pc_normal
    except Exception as e:
        print(f"Error: {obj_path} {e}")
        return None
    

def normalize_mesh(mesh):
    # Check if input is a scene with multiple meshes
    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes in the scene into a single mesh
        meshes = []
        for geometry in mesh.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                # Transform mesh to scene coordinates
                transform = mesh.graph[mesh.graph.nodes_geometry[0]][0]
                geometry.apply_transform(transform)
                meshes.append(geometry)
        
        # Combine all meshes
        mesh = trimesh.util.concatenate(meshes)

    # Get vertices and compute bounding box
    vertices = mesh.vertices
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Find center and scale
    center = (bbox_min + bbox_max) * 0.5
    scale = 2.0 / (bbox_max - bbox_min).max()
    
    # Center and scale vertices
    vertices = (vertices - center) * scale
    
    # Create new mesh with normalized vertices
    normalized_mesh = trimesh.Trimesh(vertices=vertices, 
                                    faces=mesh.faces,
                                    face_normals=mesh.face_normals,
                                    vertex_normals=mesh.vertex_normals,
                                    process=False)
    
    # # Copy texture from original mesh if it exists
    # if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
    #     print("copy material")
    #     normalized_mesh.visual.material = mesh.visual.material
    # if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'texture'):
    #     print("copy texture")
    #     normalized_mesh.visual.texture = mesh.visual.texture
    # if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
    #     print("copy uv")
    #     normalized_mesh.visual.uv = mesh.visual.uv
    
    return normalized_mesh


def vis_joint(normalized_mesh_file, joints):
    if normalized_mesh_file is None or joints is None:
        return None, None
    vis_file = visualize_results(mesh_file=normalized_mesh_file, joints=joints)
    return vis_file, vis_file

def vis_connectivity(normalized_mesh_file, joints, conns):
    if normalized_mesh_file is None or joints is None or conns is None:
        return None, None
    vis_file = visualize_results(mesh_file=normalized_mesh_file, joints=joints, conns=conns)
    return vis_file, vis_file

def vis_skinning(normalized_mesh_file, joints, conns, skins):
    if normalized_mesh_file is None or joints is None or conns is None or skins is None:
        return None, None
    vis_file = visualize_results(mesh_file=normalized_mesh_file, joints=joints, conns=conns, skins=skins)
    return vis_file, vis_file

def prepare_blender_file(normalized_mesh_file):
    if normalized_mesh_file is None:
        return None
    
    if not os.path.exists(normalized_mesh_file) or not os.path.exists(normalized_mesh_file.replace('object.obj', 'joints.pt')) or not os.path.exists(normalized_mesh_file.replace('object.obj', 'conns.pt')) or not os.path.exists(normalized_mesh_file.replace('object.obj', 'skins.pt')):
        return None

    folder = normalized_mesh_file.replace('object.obj', '')
    abs_folder = os.path.abspath(folder)
    os.system(f"python Render.py --path '{abs_folder}'")

    blender_file = os.path.join(folder, 'blender_output.blend')
    while not os.path.exists(blender_file):
        time.sleep(1)
    
    return blender_file


def process_input(mesh_file):
    """
    Function to handle input changes and initialize visualization
    
    Args:
        mesh_file: Path to input mesh file
        joint_checkpoint: Path to joint prediction checkpoint
        conn_checkpoint: Path to connectivity prediction checkpoint 
        skin_checkpoint: Path to skinning prediction checkpoint
    
    Returns:
        vis_file: Path to visualization file
    """

    # For now just visualize the input mesh
    if mesh_file is None:
        return None, None, None, None, None, None, None, None
    
    # make folder for tmp files
    if mesh_file.endswith('.obj'):
        os.makedirs(f"Anymate/tmp/{mesh_file.split('/')[-1].replace('.obj', '')}", exist_ok=True)
    else:
        os.makedirs(f"Anymate/tmp/{mesh_file.split('/')[-1].replace('.glb', '')}", exist_ok=True)
        abs_path = os.path.abspath(mesh_file)

        os.system(f"python Convert.py --path '{abs_path}'")

        mesh_file = abs_path.replace('.glb', '.obj')
        while not os.path.exists(mesh_file):
            time.sleep(1)

    normalized_mesh = normalize_mesh(trimesh.load(mesh_file))
    normalized_mesh_file = f"Anymate/tmp/{mesh_file.split('/')[-1].replace('.obj', '')}/object.obj"
    normalized_mesh.export(normalized_mesh_file)

    # normalized_mesh.export(mesh_file)

    vis_file = visualize_results(mesh_file=normalized_mesh_file)
    pc = process_mesh_to_pc(normalized_mesh_file)
    pc = torch.from_numpy(pc).to(anymate_args.device).to(torch.float32)

    # print(pc.shape, pc.max(dim=0), pc.min(dim=0))

    return normalized_mesh_file, None, None, None, pc, None, None, None


def get_model(checkpoint):
    model = load_checkpoint(checkpoint, anymate_args.device, anymate_args.num_joints)
    return model, True

def get_result_joint(mesh_file, model, pc, eps=0.03, min_samples=1):
    return get_joint(pc, model, device=anymate_args.device, save=mesh_file.replace('object.obj', 'joints.pt'), eps=eps, min_samples=min_samples)

def get_result_connectivity(mesh_file, model, pc, joints):
    return get_connectivity(pc, joints, model, device=anymate_args.device, save=mesh_file.replace('object.obj', 'conns.pt'))

def get_result_skinning(mesh_file, model, pc, joints, conns):
    mesh = trimesh.load(mesh_file)
    # mesh = obj2mesh(mesh_file)
    vertices = torch.from_numpy(mesh.vertices).to(anymate_args.device).to(torch.float32)
    vertex_normals = torch.from_numpy(mesh.vertex_normals).to(anymate_args.device).to(torch.float32)
    vertices = torch.cat([vertices, vertex_normals], dim=-1)
    return get_skinning(pc, joints, conns, model, vertices=vertices, device=anymate_args.device, save=mesh_file.replace('object.obj', 'skins.pt'))

def get_all_models(checkpoint_joint, checkpoint_conn, checkpoint_skin):
    model_joint = load_checkpoint(checkpoint_joint, anymate_args.device, anymate_args.num_joints)
    model_connectivity = load_checkpoint(checkpoint_conn, anymate_args.device, anymate_args.num_joints)
    model_skin = load_checkpoint(checkpoint_skin, anymate_args.device, anymate_args.num_joints)
    return model_joint, model_connectivity, model_skin, True, True, True

def get_all_results(mesh_file, model_joint, model_connectivity, model_skin, pc, eps=0.03, min_samples=1):
    joints = get_result_joint(mesh_file, model_joint, pc, eps=eps, min_samples=min_samples)
    conns = get_result_connectivity(mesh_file, model_connectivity, pc, joints)
    skins = get_result_skinning(mesh_file, model_skin, pc, joints, conns)
    return joints, conns, skins

