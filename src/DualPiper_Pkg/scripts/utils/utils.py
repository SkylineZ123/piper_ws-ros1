import numpy as np
import sensor_msgs.point_cloud2 as pc2
def imgmsg_to_cv2(img_msg):
    """
    纯 Python 实现的 ROS Image 转 OpenCV (Numpy) 格式
    目前支持最常见的 rgb8 和 bgr8 编码
    """
    # 检查图像编码格式
    if img_msg.encoding not in ["rgb8", "bgr8"]:
        raise ValueError(f"暂时不支持此编码格式: {img_msg.encoding}")

    # 假设是 8位 3通道 图像
    dtype = np.uint8
    channels = 3

    # 直接从内存中读取字节流并重组为矩阵
    image = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, channels)

    # 如果原始数据是 rgb8，OpenCV 默认使用 bgr8，需要调换通道
    if img_msg.encoding == "rgb8":
        image = image[:, :, ::-1] # 简单的通道翻转 RGB -> BGR

    return image

def sample_pointcloud(pc_msg, target_points: int = 5000):
    """
    Convert a PointCloud2 message → (target_points, 6) float32 array [x,y,z,r,g,b].
    If the cloud has only XYZ, the RGB columns are zero-filled.
    Points are randomly down-/up-sampled to exactly *target_points*.
    """
    pts = []
    for p in pc2.read_points(pc_msg, skip_nans=True):
        pts.append(list(p[:6]))          # take up to 6 fields
    if not pts:
        return np.zeros((target_points, 6), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)
    # Pad to 6 columns if fewer
    if arr.shape[1] < 6:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)])
    else:
        arr = arr[:, :6]

    # Filter non-finite values
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.shape[0] == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    n = arr.shape[0]
    idx = np.random.choice(n, target_points, replace=(n < target_points))
    return arr[idx]
