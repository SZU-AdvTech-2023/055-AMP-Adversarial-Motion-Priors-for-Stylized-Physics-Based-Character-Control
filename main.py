import isaacgym
import torch
from isaacgymenvs.tasks.amp.poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from isaacgymenvs.tasks.amp.poselib.poselib.visualization.common import plot_skeleton_state


# t = SkeletonTree.from_mjcf("./assets/mjcf/nv_ant.xml")  # OK
t = SkeletonTree.from_mjcf("./assets/mjcf/nv_humanoid.xml")  # OK
# t = SkeletonTree.from_mjcf("./assets/mjcf/humanoid_CMU_V2020_v2.xml")  # NOT WORKING
print(t)

print('=======================================')
zero_pose = SkeletonState.zero_pose(t)
print(zero_pose.local_rotation.shape)
print(zero_pose.global_rotation.shape)
print(zero_pose.is_local)
print(zero_pose.tensor.shape)

# plot_skeleton_state(zero_pose)
