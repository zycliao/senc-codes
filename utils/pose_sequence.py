import numpy as np
from scipy.spatial.transform import Rotation as R

def slerp(q0, q1, r):
    """
    Spherical linear interpolation.
    :param q0, q1: (N, 4) or (N, T, 4) (N can be 1 for broadcasting)
    :param r: (N,)
    """
    assert q0.shape == q1.shape and q0.shape[-1] == 4, f"q0 shape: {q0.shape}, q1 shape: {q1.shape}"
    if q0.ndim == 3:
        r = np.expand_dims(r, axis=[-2, -1])
    elif q0.ndim == 2:
        r = np.expand_dims(r, axis=[-1])
    else:
        raise ValueError(f"q0 ndim: {q0.ndim}")
    dot = (q0 * q1).sum(-1, keepdims=True)
    dot = np.clip(dot, -1, 1)
    omega = np.arccos(dot) + 1e-8
    sin_omega = np.sin(omega) + 1e-8
    w0 = np.where(sin_omega, np.sin((1 - r) * omega) / sin_omega, 1 - r)
    w1 = np.where(sin_omega, np.sin(r * omega) / sin_omega, r)
    return w0 * q0 + w1 * q1


# def axis_angle_to_quat(rotvec):
#     angle = np.linalg.norm(rotvec, axis=-1)[..., None] + np.finfo(float).eps
#     axis = rotvec / angle
#     sin = np.sin(angle / 2)
#     w = np.cos(angle / 2)
#     return np.concatenate((w, sin * axis), axis=-1)
#
#
# def quat_to_axis_angle(quat):
#     angle = 2 * np.arccos(quat[..., 0:1])
#     axis = quat[..., 1:] * (1 / (np.sin(angle / 2) + np.finfo(float).eps))
#     return angle * axis


def axis_angle_to_quat(rotvec):
    orig_shape = rotvec.shape
    rotvec = rotvec.reshape([-1, 3])
    quat = R.from_rotvec(rotvec).as_quat()
    return quat.reshape(orig_shape[:-1] + (4,))


def quat_to_axis_angle(quat):
    orig_shape = quat.shape
    quat = quat.reshape([-1, 4])
    rotvec = R.from_quat(quat).as_rotvec()
    return rotvec.reshape(orig_shape[:-1] + (3,))


class PoseSequence:
    def __init__(self, npz_file, is_amass=True, keep_hand=False):
        with np.load(npz_file) as data:
            self.poses = data["poses"].astype(np.float32)
            self.trans = data["trans"].astype(np.float32)
            self.betas = data["betas"].astype(np.float32)
            self.gender = data["gender"]

            if 'fps' in data:
                self.fps = data["fps"]
            elif 'mocap_framerate' in data:
                self.fps = data["mocap_framerate"]
            else:
                raise ValueError('fps not found in npz file')
            # self.source = data["source"]
        # convert self.poses from rotation matrix to axis-angle
        if is_amass:
            assert self.poses.shape[-1] == 156
            num_frames = self.poses.shape[0]
            self.poses = self.poses.reshape([num_frames, 52, 3])
            global_mat = R.from_rotvec(self.poses[:, 0]).as_matrix()

            # correct the global rotation inconsistency
            correction_mat = R.from_rotvec(np.array([-np.pi / 2, 0, 0])).as_matrix()
            global_mat = np.einsum('ij,bjk->bik', correction_mat, global_mat)
            self.trans = np.einsum('ij,bj->bi', correction_mat, self.trans)
            # self.trans = np.einsum('ij,bj->bi', R.from_rotvec(np.array([0, -np.pi / 2, 0])).as_matrix(), self.trans)
            self.poses[:, 0] = R.from_matrix(global_mat).as_rotvec()
            if keep_hand:
                num_joints = 52
            else:
                num_joints = 24
            self.poses = self.poses[:, :num_joints].reshape([num_frames*num_joints, 3])
            #
            # self.poses = R.from_rotvec(self.poses).as_quat().reshape([-1, 24, 4])
            self.poses = axis_angle_to_quat(self.poses).reshape([num_frames, num_joints, 4])
            # self.poses = self.poses[:, :, [3, 0, 1, 2]]

        # convert self.poses from axis-angle to quaternion

        if self.poses.shape[-1] == 3:
            self.poses = axis_angle_to_quat(self.poses)
        # self.poses = self.poses[1000:1200]
        # self.trans = self.trans[1000:1200]
        self.poses = self.poses.astype(np.float32)
        self.trans = self.trans.astype(np.float32)

    @property
    def num_frames(self):
        return self.poses.shape[0]

    @property
    def duration(self):
        return (self.num_frames - 1) / self.fps

    @property
    def dt(self):
        return 1 / self.fps

    @property
    def num_joints(self):
        return self.poses.shape[1]

    @property
    def skeleton_shape(self):
        return self.poses.shape[1:]

    # Reads body pose and translation for the times specified in the nd-array 't'
    # Supports interpolation and extrapolation
    # Given an input 't' with dimensionality [t0, t1, ..., tn]
    # Returns pose as [t0, t1, ..., tn, num_joints, 4]
    # and translation as [t0, t1, ..., tn, 3]
    def get(self, t, extrapolation=None, get_quat=True):
        batch_shape = t.shape
        t = t.reshape(-1)
        t = self.extrapolate(t, extrapolation)
        frame = t * self.fps
        frame, r = np.int32(frame), np.float32(frame % 1)
        next_frame = np.minimum(frame + 1, self.num_frames - 1)
        # Pose interpolation
        prev = self.poses[frame]
        next = self.poses[next_frame]
        pose = np.where(
            np.expand_dims(frame == next_frame, [-2, -1]), prev, slerp(prev, next, r)
        )
        # Translation interpolation
        prev = self.trans[frame]
        next = self.trans[next_frame]
        r = np.expand_dims(r, axis=-1)
        trans = (1 - r) * prev + r * next
        # Reshape into input batch shape
        pose = pose.reshape((*batch_shape, *self.skeleton_shape))
        trans = trans.reshape((*batch_shape, 3))
        if get_quat:
            return pose, trans
        else:
            pose_axis_angle = []
            for i in range(pose.shape[0]):
                pose_axis_angle.append(R.from_quat(pose[i]).as_rotvec())
            pose_axis_angle = np.array(pose_axis_angle, dtype=np.float32)
            return pose_axis_angle, trans.astype(np.float32)

    def get_by_fps(self, target_fps, get_quat=False):
        num_frames = int(self.duration * target_fps) + 1
        t = np.arange(num_frames) / target_fps
        return self.get(t, get_quat=get_quat)

    # Extrapolates by mapping values of 't' outside sequence duration [0, T] to values within
    def extrapolate(self, t, mode=None):
        assert mode in {
            None,
            "clip",
            "mirror",
        }, "Wrong time extrapolation mode, must be in {None, 'clip', 'mirror'}"
        t = np.array(t)
        if (t >= 0.0).all() and (t <= self.duration).all():
            return t
        if mode is None:
            raise Exception("Queried time is outside the length of the sequence.")
        if mode == "clip":
            return np.clip(t, 0.0, self.duration)
        if mode == "mirror":
            t = np.where(t < 0, -t, t)
            t = np.where(t > self.duration, 2 * self.duration - t, t)
            return self.extrapolate(t, mode)

