"""Small IMU summary and display rotation only; never transforms sensor data."""
from collections import deque
import numpy as np

MAX_IMU_AGE_SECONDS = 1.5

class GravityWindow:
    def __init__(self): self.samples=deque(maxlen=1000)
    def add(self,when,acceleration,angular_velocity):
        if np.isfinite([when,*acceleration,*angular_velocity]).all():
            self.samples.append((when,acceleration,angular_velocity))
    def view(self,when):
        if not self.samples: return {'stable':False}
        latest=self.samples[-1][0]; age=when-latest
        samples=[s for s in self.samples if 0<=latest-s[0]<=2]
        if len(samples)<100: return {'stable':False}
        a=np.array([s[1] for s in samples]);g=np.array([s[2] for s in samples]);mean=a.mean(axis=0);norm=np.linalg.norm(mean)
        stable=bool(samples[-1][0]-samples[0][0]>=1.5 and 0<=age<=MAX_IMU_AGE_SECONDS and norm>.2 and np.linalg.norm(a.std(axis=0))/max(norm,1e-9)<.025 and np.linalg.norm(g,axis=1).max()<.08)
        return dict(stable=stable,first=samples[0][0],last=samples[-1][0],acceleration=mean.tolist(),count=len(samples))

def leveling_quaternion(acceleration,imu_to_lidar):
    a=np.array(acceleration,dtype=float);q=np.array(imu_to_lidar,dtype=float)
    if a.shape!=(3,) or q.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(q).all() or np.linalg.norm(a)<.2 or abs(np.linalg.norm(q)-1)>.01:
        raise ValueError('Invalid gravity or IMU transform')
    q=q/np.linalg.norm(q)
    # Existing fixed extrinsic is read only, to express gravity in LiDAR axes.
    a=a+2*np.cross(q[:3],np.cross(q[:3],a)+q[3]*a)
    a/=np.linalg.norm(a);up=np.array([0.,0.,1.])
    if a[2]<-1+1e-8:return [1.,0.,0.,0.]
    rotation=np.r_[np.cross(a,up),1+a[2]]
    return (rotation/np.linalg.norm(rotation)).tolist()
