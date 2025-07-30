import numpy as np
import trimesh
import pyrender
from PIL import Image
import os


# 合并Metallic和Roughness贴图
def merge_mr_maps(roughness_path, metallic_path=None, size=(1024, 1024)):
    """
    将单独的金属度和粗糙度贴图合并为一张符合glTF规范的贴图。
    粗糙度 -> G通道, 金属度 -> B通道
    """
    try:
        roughness_img = Image.open(roughness_path).convert('L').resize(size)
        roughness_data = np.array(roughness_img)
    except FileNotFoundError:
        print(f"警告：找不到粗糙度贴图 '{roughness_path}'。将使用默认值。")
        roughness_data = np.full((size[1], size[0]), 255 * 0.5, dtype=np.uint8)  # 默认0.5的粗糙度

    if metallic_path and os.path.exists(metallic_path):
        metallic_img = Image.open(metallic_path).convert('L').resize(size)
        metallic_data = np.array(metallic_img)
    else:
        print("信息：找不到金属度贴图。假设为非金属材质(金属度=0)。")
        metallic_data = np.zeros((size[1], size[0]), dtype=np.uint8)

    mr_data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    mr_data[:, :, 1] = roughness_data
    mr_data[:, :, 2] = metallic_data
    return Image.fromarray(mr_data)


# --- 主程序 ---
# 1. 定义贴图文件路径 (请确保这些文件在 'textures' 文件夹中)
TEXTURE_DIR = 'textures/'
MODEL_PATH = os.path.join(TEXTURE_DIR, 'model1.obj')
ALBEDO_PATH = os.path.join(TEXTURE_DIR, 'albedo.jpg')
# ALBEDO_PATH = os.path.join(TEXTURE_DIR, 'texture_pbr_v128.png')
ROUGHNESS_PATH = os.path.join(TEXTURE_DIR, 'roughness.jpg')
NORMAL_PATH = os.path.join(TEXTURE_DIR, 'normalGL.jpg')
AO_PATH = os.path.join(TEXTURE_DIR, 'ao.jpg')
DISPLACEMENT_PATH = os.path.join(TEXTURE_DIR, 'displacement.jpg')
METALLIC_PATH = os.path.join(TEXTURE_DIR, 'metallic.jpg')

# 检查必要文件是否存在
if not os.path.exists(ALBEDO_PATH):
    print(f"关键错误：无法找到 Albedo 贴图 '{ALBEDO_PATH}'。程序无法继续。")
    exit()

# 2. 加载外部模型
try:
    # 使用trimesh加载模型文件
    scene_or_mesh = trimesh.load(MODEL_PATH, process=True, force='mesh')
except Exception as e:
    print(f"错误：无法加载模型文件 '{MODEL_PATH}'.")
    print(f"具体错误: {e}")
    exit()

# 如果加载的是一个场景（包含多个网格），我们只取第一个
if isinstance(scene_or_mesh, trimesh.Scene):
    if len(scene_or_mesh.geometry) == 0:
        print("错误：加载的文件不包含任何几何体。")
        exit()
    # 将所有几何体合并成一个
    loaded_mesh = trimesh.util.concatenate(
        tuple(g for g in scene_or_mesh.geometry.values())
    )
else:
    loaded_mesh = scene_or_mesh

# 3. 预处理加载的模型

# a. 将模型的中心移到世界坐标原点 (0, 0, 0)
loaded_mesh.apply_translation(-loaded_mesh.centroid)

# b. 将模型缩放到一个合适的大小
#    我们将其最大边长缩放到2个单位，这样它就能很好地适配我们的相机
scale_factor = 2.0 / loaded_mesh.scale
loaded_mesh.apply_scale(scale_factor)

# c. 检查模型是否有UV坐标，这是贴图所必需的
if not hasattr(loaded_mesh.visual, 'uv') or loaded_mesh.visual.uv is None:
    print("警告：加载的模型没有UV坐标！贴图将无法正确显示。")
    # 对于简单模型，可以尝试自动生成，但效果不一定好
    # loaded_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=trimesh.remesh.subdivide(loaded_mesh).vertices[:,:2])

# d. 细分模型以便应用置换贴图
loaded_mesh = loaded_mesh.subdivide_to_size(max_edge=0.02)
print(f"模型加载并处理完毕。顶点数: {len(loaded_mesh.vertices)}")

# 4. 应用置换贴图
if os.path.exists(DISPLACEMENT_PATH) and hasattr(loaded_mesh.visual, 'uv'):
    print("正在应用置换贴图...")
    disp_img = Image.open(DISPLACEMENT_PATH).convert('L')
    disp_data = np.array(disp_img) / 255.0

    uv_coords = loaded_mesh.visual.uv
    vertex_normals = loaded_mesh.vertex_normals

    uv_coords_flipped_v = uv_coords.copy()
    uv_coords_flipped_v[:, 1] = 1.0 - uv_coords_flipped_v[:, 1]

    img_size = np.array(disp_img.size) - 1
    uv_pixels = np.clip((uv_coords_flipped_v * img_size), 0, img_size).astype(int)
    height_values = disp_data[uv_pixels[:, 1], uv_pixels[:, 0]]

    displacement_strength = 0.05
    loaded_mesh.vertices += vertex_normals * height_values[:, np.newaxis] * displacement_strength
    print(f"置换应用完毕。模型新边界: {loaded_mesh.bounds}")
else:
    print(f"警告：找不到置换贴图 '{DISPLACEMENT_PATH}'。跳过置换步骤。")

# 5. 加载和准备PBR贴图
albedo_img = Image.open(ALBEDO_PATH)
normal_img = Image.open(NORMAL_PATH)
ao_img = Image.open(AO_PATH)
metallic_roughness_img = merge_mr_maps(ROUGHNESS_PATH, METALLIC_PATH, size=albedo_img.size)

albedo_tex = pyrender.Texture(source=albedo_img, source_channels='RGB')
normal_tex = pyrender.Texture(source=normal_img, source_channels='RGB')
ao_tex = pyrender.Texture(source=ao_img, source_channels='R')
metallic_roughness_tex = pyrender.Texture(source=metallic_roughness_img, source_channels='RGB')

# 6. 创建PBR材质和最终的pyrender网格
textured_pbr = pyrender.MetallicRoughnessMaterial(
    baseColorTexture=albedo_tex,
    metallicRoughnessTexture=metallic_roughness_tex,
    normalTexture=normal_tex,
    occlusionTexture=ao_tex,
)
final_mesh = pyrender.Mesh.from_trimesh(loaded_mesh, material=textured_pbr)

# 7. 设置场景、相机、光源和渲染器
# scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[0.2, 0.2, 0.3])
# scene.add(final_mesh)
#
# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# camera_pose = np.eye(4)
# camera_pose[2, 3] = 3.5  # 将相机放在Z轴上，直视原点
# scene.add(camera, pose=camera_pose)
#
# # 使用Viewer自带的光源，方便观察
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0.2, 0.2, 0.3, 1.0])
scene.add(final_mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0]
])
scene.add(camera, pose=camera_pose)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
scene.add(light, pose=camera_pose)

# 8. 渲染
print("启动渲染查看器...")
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, shadows=True)
