# engine/core/math3d.py
"""
Core math types for 3D geometric environment.
Designed for CPU-side transforms - upload to GPU as uniform data.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Union, Iterator

# =============================================================================
# Vector Types
# =============================================================================

@dataclass
class Vec2:
    """2D vector for screen/panel coordinates."""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> Vec2:
        return self.__mul__(scalar)
    
    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)
    
    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y
    
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y
    
    def normalized(self) -> Vec2:
        ln = self.length()
        if ln < 1e-10:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / ln, self.y / ln)
    
    def lerp(self, other: Vec2, t: float) -> Vec2:
        return Vec2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @staticmethod
    def from_tuple(t: Tuple[float, float]) -> Vec2:
        return Vec2(t[0], t[1])


@dataclass
class Vec3:
    """3D vector for simulation space (beats, frequency, intensity)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)
    
    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)
    
    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def normalized(self) -> Vec3:
        ln = self.length()
        if ln < 1e-10:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / ln, self.y / ln, self.z / ln)
    
    def lerp(self, other: Vec3, t: float) -> Vec3:
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def xy(self) -> Vec2:
        return Vec2(self.x, self.y)
    
    @staticmethod
    def from_tuple(t: Tuple[float, float, float]) -> Vec3:
        return Vec3(t[0], t[1], t[2])
    
    @staticmethod
    def unit_x() -> Vec3:
        return Vec3(1.0, 0.0, 0.0)
    
    @staticmethod
    def unit_y() -> Vec3:
        return Vec3(0.0, 1.0, 0.0)
    
    @staticmethod
    def unit_z() -> Vec3:
        return Vec3(0.0, 0.0, 1.0)


@dataclass
class Vec4:
    """4D vector for homogeneous coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def __add__(self, other: Vec4) -> Vec4:
        return Vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
    
    def __sub__(self, other: Vec4) -> Vec4:
        return Vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
    
    def __mul__(self, scalar: float) -> Vec4:
        return Vec4(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)
    
    def dot(self, other: Vec4) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    def xyz(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)
    
    def to_vec3(self) -> Vec3:
        """Perspective divide to get 3D point."""
        if abs(self.w) < 1e-10:
            return Vec3(self.x, self.y, self.z)
        return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)
    
    @staticmethod
    def from_vec3(v: Vec3, w: float = 1.0) -> Vec4:
        return Vec4(v.x, v.y, v.z, w)
    
    @staticmethod
    def point(x: float, y: float, z: float) -> Vec4:
        """Create a point (w=1)."""
        return Vec4(x, y, z, 1.0)
    
    @staticmethod
    def direction(x: float, y: float, z: float) -> Vec4:
        """Create a direction vector (w=0)."""
        return Vec4(x, y, z, 0.0)


# =============================================================================
# Matrix Types
# =============================================================================

class Mat3:
    """3x3 matrix for 2D transforms and rotations."""
    
    __slots__ = ('m',)
    
    def __init__(self, values: Tuple[float, ...] = None):
        """Initialize with row-major values or identity."""
        if values is None:
            self.m = (
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            )
        else:
            assert len(values) == 9
            self.m = tuple(values)
    
    def __getitem__(self, idx: Tuple[int, int]) -> float:
        row, col = idx
        return self.m[row * 3 + col]
    
    def __matmul__(self, other: Union[Mat3, Vec3]) -> Union[Mat3, Vec3]:
        if isinstance(other, Mat3):
            return self._mul_mat(other)
        elif isinstance(other, Vec3):
            return self._mul_vec(other)
        raise TypeError(f"Cannot multiply Mat3 by {type(other)}")
    
    def _mul_mat(self, other: Mat3) -> Mat3:
        result = []
        for row in range(3):
            for col in range(3):
                val = sum(self[row, k] * other[k, col] for k in range(3))
                result.append(val)
        return Mat3(tuple(result))
    
    def _mul_vec(self, v: Vec3) -> Vec3:
        return Vec3(
            self[0,0]*v.x + self[0,1]*v.y + self[0,2]*v.z,
            self[1,0]*v.x + self[1,1]*v.y + self[1,2]*v.z,
            self[2,0]*v.x + self[2,1]*v.y + self[2,2]*v.z
        )
    
    def transpose(self) -> Mat3:
        return Mat3((
            self[0,0], self[1,0], self[2,0],
            self[0,1], self[1,1], self[2,1],
            self[0,2], self[1,2], self[2,2]
        ))
    
    def determinant(self) -> float:
        return (
            self[0,0] * (self[1,1]*self[2,2] - self[1,2]*self[2,1]) -
            self[0,1] * (self[1,0]*self[2,2] - self[1,2]*self[2,0]) +
            self[0,2] * (self[1,0]*self[2,1] - self[1,1]*self[2,0])
        )
    
    def to_tuple(self) -> Tuple[float, ...]:
        return self.m
    
    def to_list_column_major(self) -> list:
        """For GPU upload (OpenGL expects column-major)."""
        return [
            self[0,0], self[1,0], self[2,0],
            self[0,1], self[1,1], self[2,1],
            self[0,2], self[1,2], self[2,2]
        ]
    
    @staticmethod
    def identity() -> Mat3:
        return Mat3()
    
    @staticmethod
    def scale(sx: float, sy: float) -> Mat3:
        return Mat3((
            sx,  0.0, 0.0,
            0.0, sy,  0.0,
            0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def translate(tx: float, ty: float) -> Mat3:
        return Mat3((
            1.0, 0.0, tx,
            0.0, 1.0, ty,
            0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def rotate(angle: float) -> Mat3:
        """Rotation by angle (radians)."""
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat3((
            c,   -s,  0.0,
            s,    c,  0.0,
            0.0, 0.0, 1.0
        ))


class Mat4:
    """4x4 matrix for 3D transforms."""
    
    __slots__ = ('m',)
    
    def __init__(self, values: Tuple[float, ...] = None):
        """Initialize with row-major values or identity."""
        if values is None:
            self.m = (
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            )
        else:
            assert len(values) == 16
            self.m = tuple(values)
    
    def __getitem__(self, idx: Tuple[int, int]) -> float:
        row, col = idx
        return self.m[row * 4 + col]
    
    def __matmul__(self, other: Union[Mat4, Vec4, Vec3]) -> Union[Mat4, Vec4, Vec3]:
        if isinstance(other, Mat4):
            return self._mul_mat(other)
        elif isinstance(other, Vec4):
            return self._mul_vec4(other)
        elif isinstance(other, Vec3):
            # Treat as point (w=1)
            v4 = self._mul_vec4(Vec4.from_vec3(other, 1.0))
            return v4.to_vec3()
        raise TypeError(f"Cannot multiply Mat4 by {type(other)}")
    
    def _mul_mat(self, other: Mat4) -> Mat4:
        result = []
        for row in range(4):
            for col in range(4):
                val = sum(self[row, k] * other[k, col] for k in range(4))
                result.append(val)
        return Mat4(tuple(result))
    
    def _mul_vec4(self, v: Vec4) -> Vec4:
        return Vec4(
            self[0,0]*v.x + self[0,1]*v.y + self[0,2]*v.z + self[0,3]*v.w,
            self[1,0]*v.x + self[1,1]*v.y + self[1,2]*v.z + self[1,3]*v.w,
            self[2,0]*v.x + self[2,1]*v.y + self[2,2]*v.z + self[2,3]*v.w,
            self[3,0]*v.x + self[3,1]*v.y + self[3,2]*v.z + self[3,3]*v.w
        )
    
    def transpose(self) -> Mat4:
        return Mat4(tuple(
            self[col, row] 
            for row in range(4) 
            for col in range(4)
        ))
    
    def to_tuple(self) -> Tuple[float, ...]:
        return self.m
    
    def to_list_column_major(self) -> list:
        """For GPU upload (OpenGL expects column-major)."""
        result = []
        for col in range(4):
            for row in range(4):
                result.append(self[row, col])
        return result
    
    def to_mat3(self) -> Mat3:
        """Extract upper-left 3x3 (rotation/scale)."""
        return Mat3((
            self[0,0], self[0,1], self[0,2],
            self[1,0], self[1,1], self[1,2],
            self[2,0], self[2,1], self[2,2]
        ))
    
    def inverse(self) -> Mat4:
        """Compute inverse matrix."""
        m = self.m
        
        c00 = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10]
        c01 = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10]
        c02 = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9]
        c03 = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9]
        
        det = m[0]*c00 + m[1]*c01 + m[2]*c02 + m[3]*c03
        if abs(det) < 1e-10:
            return Mat4.identity()
        
        inv_det = 1.0 / det
        
        c10 = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10]
        c11 = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10]
        c12 = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9]
        c13 = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9]
        
        c20 = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6]
        c21 = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6]
        c22 = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5]
        c23 = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5]
        
        c30 = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6]
        c31 = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6]
        c32 = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5]
        c33 = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5]
        
        return Mat4((
            c00*inv_det, c10*inv_det, c20*inv_det, c30*inv_det,
            c01*inv_det, c11*inv_det, c21*inv_det, c31*inv_det,
            c02*inv_det, c12*inv_det, c22*inv_det, c32*inv_det,
            c03*inv_det, c13*inv_det, c23*inv_det, c33*inv_det
        ))
    
    @staticmethod
    def identity() -> Mat4:
        return Mat4()
    
    @staticmethod
    def scale(sx: float, sy: float = None, sz: float = None) -> Mat4:
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        return Mat4((
            sx,  0.0, 0.0, 0.0,
            0.0, sy,  0.0, 0.0,
            0.0, 0.0, sz,  0.0,
            0.0, 0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def translate(tx: float, ty: float, tz: float) -> Mat4:
        return Mat4((
            1.0, 0.0, 0.0, tx,
            0.0, 1.0, 0.0, ty,
            0.0, 0.0, 1.0, tz,
            0.0, 0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def translate_vec(v: Vec3) -> Mat4:
        return Mat4.translate(v.x, v.y, v.z)
    
    @staticmethod
    def rotate_x(angle: float) -> Mat4:
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat4((
            1.0, 0.0, 0.0, 0.0,
            0.0, c,   -s,  0.0,
            0.0, s,    c,  0.0,
            0.0, 0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def rotate_y(angle: float) -> Mat4:
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat4((
            c,   0.0, s,   0.0,
            0.0, 1.0, 0.0, 0.0,
            -s,  0.0, c,   0.0,
            0.0, 0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def rotate_z(angle: float) -> Mat4:
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat4((
            c,   -s,  0.0, 0.0,
            s,    c,  0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ))
    
    @staticmethod
    def rotate_axis(axis: Vec3, angle: float) -> Mat4:
        axis = axis.normalized()
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        x, y, z = axis.x, axis.y, axis.z
        
        return Mat4((
            t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,  0.0,
            t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,  0.0,
            t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,    0.0,
            0.0,          0.0,          0.0,          1.0
        ))
    
    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3 = None) -> Mat4:
        if up is None:
            up = Vec3(0.0, 1.0, 0.0)
        
        forward = (target - eye).normalized()
        right = forward.cross(up).normalized()
        up = right.cross(forward)
        
        return Mat4((
            right.x,    right.y,    right.z,    -right.dot(eye),
            up.x,       up.y,       up.z,       -up.dot(eye),
            -forward.x, -forward.y, -forward.z,  forward.dot(eye),
            0.0,        0.0,        0.0,         1.0
        ))
    
    @staticmethod
    def ortho(left: float, right: float, bottom: float, top: float, 
              near: float, far: float) -> Mat4:
        dx = right - left
        dy = top - bottom
        dz = far - near
        
        return Mat4((
            2.0/dx,  0.0,     0.0,      -(right+left)/dx,
            0.0,     2.0/dy,  0.0,      -(top+bottom)/dy,
            0.0,     0.0,    -2.0/dz,   -(far+near)/dz,
            0.0,     0.0,     0.0,       1.0
        ))
    
    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> Mat4:
        f = 1.0 / math.tan(fov_y / 2.0)
        dz = near - far
        
        return Mat4((
            f/aspect, 0.0, 0.0,                    0.0,
            0.0,      f,   0.0,                    0.0,
            0.0,      0.0, (far+near)/dz,          2.0*far*near/dz,
            0.0,      0.0, -1.0,                   0.0
        ))


# =============================================================================
# Quaternion
# =============================================================================

@dataclass
class Quat:
    """Quaternion for rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def __mul__(self, other: Quat) -> Quat:
        return Quat(
            self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
            self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        )
    
    def conjugate(self) -> Quat:
        return Quat(-self.x, -self.y, -self.z, self.w)
    
    def length(self) -> float:
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w)
    
    def normalized(self) -> Quat:
        ln = self.length()
        if ln < 1e-10:
            return Quat(0.0, 0.0, 0.0, 1.0)
        return Quat(self.x/ln, self.y/ln, self.z/ln, self.w/ln)
    
    def rotate_vec(self, v: Vec3) -> Vec3:
        qv = Quat(v.x, v.y, v.z, 0.0)
        result = self * qv * self.conjugate()
        return Vec3(result.x, result.y, result.z)
    
    def to_mat4(self) -> Mat4:
        x, y, z, w = self.x, self.y, self.z, self.w
        
        xx = x*x; yy = y*y; zz = z*z
        xy = x*y; xz = x*z; yz = y*z
        wx = w*x; wy = w*y; wz = w*z
        
        return Mat4((
            1-2*(yy+zz),  2*(xy-wz),    2*(xz+wy),    0.0,
            2*(xy+wz),    1-2*(xx+zz),  2*(yz-wx),    0.0,
            2*(xz-wy),    2*(yz+wx),    1-2*(xx+yy),  0.0,
            0.0,          0.0,          0.0,          1.0
        ))
    
    def slerp(self, other: Quat, t: float) -> Quat:
        dot = self.x*other.x + self.y*other.y + self.z*other.z + self.w*other.w
        
        if dot < 0.0:
            other = Quat(-other.x, -other.y, -other.z, -other.w)
            dot = -dot
        
        if dot > 0.9995:
            return Quat(
                self.x + t*(other.x - self.x),
                self.y + t*(other.y - self.y),
                self.z + t*(other.z - self.z),
                self.w + t*(other.w - self.w)
            ).normalized()
        
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        
        s0 = math.sin((1-t)*theta) / sin_theta
        s1 = math.sin(t*theta) / sin_theta
        
        return Quat(
            s0*self.x + s1*other.x,
            s0*self.y + s1*other.y,
            s0*self.z + s1*other.z,
            s0*self.w + s1*other.w
        )
    
    @staticmethod
    def identity() -> Quat:
        return Quat(0.0, 0.0, 0.0, 1.0)
    
    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> Quat:
        axis = axis.normalized()
        half = angle / 2.0
        s = math.sin(half)
        return Quat(axis.x * s, axis.y * s, axis.z * s, math.cos(half))
    
    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> Quat:
        cp = math.cos(pitch / 2); sp = math.sin(pitch / 2)
        cy = math.cos(yaw / 2);   sy = math.sin(yaw / 2)
        cr = math.cos(roll / 2);  sr = math.sin(roll / 2)
        
        return Quat(
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy
        )


# =============================================================================
# Transform
# =============================================================================

@dataclass
class Transform:
    """Combined position, rotation, scale."""
    position: Vec3 = None
    rotation: Quat = None
    scale: Vec3 = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = Vec3(0.0, 0.0, 0.0)
        if self.rotation is None:
            self.rotation = Quat.identity()
        if self.scale is None:
            self.scale = Vec3(1.0, 1.0, 1.0)
    
    def to_mat4(self) -> Mat4:
        t = Mat4.translate_vec(self.position)
        r = self.rotation.to_mat4()
        s = Mat4.scale(self.scale.x, self.scale.y, self.scale.z)
        return t @ r @ s
    
    def lerp(self, other: Transform, t: float) -> Transform:
        return Transform(
            position=self.position.lerp(other.position, t),
            rotation=self.rotation.slerp(other.rotation, t),
            scale=self.scale.lerp(other.scale, t)
        )


# =============================================================================
# Easing Functions
# =============================================================================

def ease_linear(t: float) -> float:
    return t

def ease_in_quad(t: float) -> float:
    return t * t

def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2

def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0*t + 2.0) ** 2 / 2.0

def ease_in_cubic(t: float) -> float:
    return t * t * t

def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3

def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0*t + 2.0) ** 3 / 2.0

def ease_in_elastic(t: float) -> float:
    if t == 0.0 or t == 1.0:
        return t
    return -math.pow(2.0, 10.0*t - 10.0) * math.sin((t*10.0 - 10.75) * (2.0*math.pi/3.0))

def ease_out_elastic(t: float) -> float:
    if t == 0.0 or t == 1.0:
        return t
    return math.pow(2.0, -10.0*t) * math.sin((t*10.0 - 0.75) * (2.0*math.pi/3.0)) + 1.0

def ease_out_bounce(t: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if t < 1.0/d1:
        return n1 * t * t
    elif t < 2.0/d1:
        t -= 1.5/d1
        return n1 * t * t + 0.75
    elif t < 2.5/d1:
        t -= 2.25/d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625/d1
        return n1 * t * t + 0.984375


# =============================================================================
# Utility Functions
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def inverse_lerp(a: float, b: float, value: float) -> float:
    if abs(b - a) < 1e-10:
        return 0.0
    return (value - a) / (b - a)

def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    t = inverse_lerp(in_min, in_max, value)
    return lerp(out_min, out_max, t)

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def deg_to_rad(degrees: float) -> float:
    return degrees * (math.pi / 180.0)

def rad_to_deg(radians: float) -> float:
    return radians * (180.0 / math.pi)
